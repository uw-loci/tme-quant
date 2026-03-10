from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import io

from pycurvelets.tumor_annotation_from_HE import (
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

# Synthetic-fixture geometry constants used across tests.
# Ring radii define an annulus centered in the image:
#   expected analytic area ~= pi*(R_OUTER^2 - R_INNER^2) ~= 7037 px.
R_OUTER = 62.0
R_INNER = 40.0
NUCLEI_RADIUS = 35.0

# Tolerances/expectations for mask geometry checks:
# - pixel range brackets the synthetic annulus area with room for morphology/smoothing
# - centroid tolerance allows <=1 px discretization drift from thresholding
EXPECTED_TRUE_PIXELS_MIN = 6200
EXPECTED_TRUE_PIXELS_MAX = 7600
EXPECTED_COMPONENT_COUNT = 1
CENTROID_TOLERANCE_PX = 1.0


def _save_float_image(path: Path, image: np.ndarray) -> None:
    arr = np.clip(image, 0.0, 1.0)
    io.imsave(str(path), np.round(arr * 255.0).astype(np.uint8), check_contrast=False)


def _synthetic_he_image(shape: tuple[int, int] = (180, 180)) -> np.ndarray:
    rows, cols = np.indices(shape)
    he = np.zeros((shape[0], shape[1], 3), dtype=np.float64)
    # Light pink-ish background to mimic HE tissue background.
    he[:, :, :] = np.array([0.96, 0.93, 0.93], dtype=np.float64)

    center_r, center_c = shape[0] / 2.0, shape[1] / 2.0
    ring = (
        ((rows - center_r) ** 2 + (cols - center_c) ** 2 <= R_OUTER**2)
        & ((rows - center_r) ** 2 + (cols - center_c) ** 2 >= R_INNER**2)
    )
    # Red-ish ring to emulate collagen-like signal in HSV segmentation branch.
    he[ring, 0] = 0.88
    he[ring, 1] = 0.23
    he[ring, 2] = 0.22

    # Cyan center cluster to emulate nuclei-like signal.
    nuclei_cluster = ((rows - center_r) ** 2 + (cols - center_c) ** 2) <= NUCLEI_RADIUS**2
    he[nuclei_cluster, 0] = 0.32
    he[nuclei_cluster, 1] = 0.72
    he[nuclei_cluster, 2] = 0.95

    return np.clip(he, 0.0, 1.0)


def test_tumor_annotation_from_he_outputs_binary_mask_and_saves(tmp_path):
    he_dir = tmp_path / "he"
    shg_dir = tmp_path / "shg"
    he_dir.mkdir()
    shg_dir.mkdir()

    filename = "demo_HE.tif"
    he_image = _synthetic_he_image()
    _save_float_image(he_dir / filename, he_image)

    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_dir),
        HEfilename=filename,
        # Non-integer-ish ppm exercises morphology kernels used by conversion code.
        pixelpermicron=1.5,
        # Kept for MATLAB-compat parameter surface (currently not used by algorithm).
        areaThreshold=150.0,
        SHGfilepath=str(shg_dir),
    )
    mask, debug = tumor_annotation_from_he(params, save_output=True, return_debug=True)

    assert mask.shape == he_image.shape[:2]
    assert mask.dtype == np.bool_
    assert "mask_temp" in debug

    # Validate basic geometry on the synthetic sample.
    true_pixels = int(mask.sum())
    assert EXPECTED_TRUE_PIXELS_MIN <= true_pixels <= EXPECTED_TRUE_PIXELS_MAX

    labels, num_labels = ndimage.label(mask)
    assert num_labels == EXPECTED_COMPONENT_COUNT

    ys, xs = np.nonzero(mask)
    centroid_row = float(np.mean(ys))
    centroid_col = float(np.mean(xs))
    expected_row = he_image.shape[0] / 2.0
    expected_col = he_image.shape[1] / 2.0
    assert abs(centroid_row - expected_row) <= CENTROID_TOLERANCE_PX
    assert abs(centroid_col - expected_col) <= CENTROID_TOLERANCE_PX

    # Filename intentionally matches legacy MATLAB naming behavior:
    # appending ".tif" after replacing HE->SHG can produce ".tif.tif".
    expected_name = "mask for demo_SHG.tif.tif"
    out_path = shg_dir / "CA_Boundary" / expected_name
    assert out_path.exists()


def test_tumor_annotation_from_he_deterministic_with_fallback_save_path(tmp_path):
    he_dir = tmp_path / "he_fallback"
    he_dir.mkdir()

    filename = "fallback_HE.tif"
    he_image = _synthetic_he_image(shape=(164, 164))
    _save_float_image(he_dir / filename, he_image)

    params = {
        "HEfilepath": str(he_dir),
        "HEfilename": filename,
        "pixelpermicron": 1.2,
        # Kept for MATLAB-compat parameter surface (currently not used by algorithm).
        "areaThreshold": 120.0,
        "SHGfilepath": "",
    }

    mask1 = tumor_annotation_from_he(params, save_output=True, return_debug=False)
    mask2 = tumor_annotation_from_he(params, save_output=False, return_debug=False)

    np.testing.assert_array_equal(mask1, mask2)

    expected_name = "mask for fallback_SHG.tif.tif"
    out_path = he_dir / "CA_Boundary" / expected_name
    assert out_path.exists()
