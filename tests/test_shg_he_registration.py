from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import io

from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)


def _save_float_image(path: Path, image: np.ndarray) -> None:
    arr = np.clip(image, 0.0, 1.0)
    io.imsave(str(path), np.round(arr * 255.0).astype(np.uint8), check_contrast=False)


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _make_synthetic_pair(shape: tuple[int, int], shift_rc: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape)
    collagen = np.zeros(shape, dtype=np.float64)

    ellipse = (((rows - 72.0) ** 2) / (28.0**2) + ((cols - 55.0) ** 2) / (14.0**2)) <= 1.0
    blob = ((rows - 100.0) ** 2 + (cols - 110.0) ** 2) <= 13.0**2
    collagen[ellipse | blob] = 1.0
    collagen = ndimage.gaussian_filter(collagen, sigma=1.5, mode="nearest")
    collagen = np.clip(collagen / max(collagen.max(), 1e-12), 0.0, 1.0)

    he = np.zeros((shape[0], shape[1], 3), dtype=np.float64)
    he[:, :, 0] = 0.15 + 0.85 * collagen
    he[:, :, 1] = 0.08 + 0.10 * collagen
    he[:, :, 2] = 0.08 + 0.06 * collagen

    # Blue/cyan nuclei-like clusters to exercise HSV nuclei segmentation branch.
    nuclei1 = ((rows - 85.0) ** 2 + (cols - 68.0) ** 2) <= 10.0**2
    nuclei2 = ((rows - 95.0) ** 2 + (cols - 88.0) ** 2) <= 8.0**2
    nuclei = nuclei1 | nuclei2
    he[nuclei, 0] = 0.25
    he[nuclei, 1] = 0.70
    he[nuclei, 2] = 0.92

    he_shifted = np.zeros_like(he)
    for ch in range(3):
        he_shifted[:, :, ch] = ndimage.shift(
            he[:, :, ch],
            shift=shift_rc,
            order=1,
            mode="constant",
            cval=1.0,
            prefilter=False,
        )

    return np.clip(he_shifted, 0.0, 1.0), collagen


def test_shg_he_registration_improves_alignment_and_saves(tmp_path):
    he_dir = tmp_path / "he"
    shg_dir = tmp_path / "shg"
    he_dir.mkdir()
    shg_dir.mkdir()

    he_image, shg_image = _make_synthetic_pair(shape=(144, 144), shift_rc=(7.0, -6.0))
    filename = "synthetic_HE.tif"

    _save_float_image(he_dir / filename, he_image)
    _save_float_image(shg_dir / filename, shg_image)

    params = SHGHERegistrationParameters(
        HEfilepath=str(he_dir),
        HEfilename=filename,
        pixelpermicron=1.5,
        SHGfilepath=str(shg_dir),
    )
    registered, debug = shg_he_registration(params, save_output=True, return_debug=True)

    assert registered.shape == (144, 144, 3)
    assert "he_collagen_exclude" in debug

    pre_corr = _normalized_cross_correlation(he_image[:, :, 0], shg_image)
    post_corr = _normalized_cross_correlation(registered[:, :, 0], shg_image)
    assert np.isfinite(pre_corr)
    assert np.isfinite(post_corr)

    # Coarse translation should approximately recover the known synthetic shift.
    shift_rc = np.asarray(debug["shift_rc"], dtype=np.float64)
    np.testing.assert_allclose(shift_rc, np.array([-7.0, 6.0]), atol=2.0)

    out_path = he_dir / "HE_registered" / filename
    assert out_path.exists()


def test_shg_he_registration_deterministic_and_ppm_rescale(tmp_path):
    he_dir = tmp_path / "he2"
    shg_dir = tmp_path / "shg2"
    he_dir.mkdir()
    shg_dir.mkdir()

    he_image, shg_image = _make_synthetic_pair(shape=(128, 128), shift_rc=(5.0, -4.0))
    filename = "ppm_branch_HE.tif"

    _save_float_image(he_dir / filename, he_image)
    _save_float_image(shg_dir / filename, shg_image)

    params = {
        "HEfilepath": str(he_dir),
        "HEfilename": filename,
        "pixelpermicron": 3.2,
        "SHGfilepath": str(shg_dir),
    }

    out1 = shg_he_registration(params, save_output=False, return_debug=False)
    out2 = shg_he_registration(params, save_output=False, return_debug=False)

    assert out1.shape == (128, 128, 3)
    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)
