"""Smoke tests for the modules moved out of the napari plugin.

segmentation and preprocessing moved from napari_curvealign into pycurvelets
so the Fiji and napari plugins can share them. These tests only check that
each entry point runs headless on synthetic data and returns sensibly shaped
results. The plugin's TACS histogram stays in napari-curvealign; this library's
boundary angle is get_tif_boundary.get_relative_angle.
"""

import numpy as np
import pytest
from pycurvelets import preprocessing, segmentation


def _two_squares(shape=(128, 128)):
    img = np.zeros(shape, dtype=np.float64)
    img[20:50, 20:50] = 200.0
    img[70:110, 60:100] = 200.0
    return img


def test_moved_modules_do_not_import_napari():
    import sys

    assert "napari" not in sys.modules
    assert not hasattr(preprocessing, "HAS_IMAGEJ")


@pytest.mark.skipif(not segmentation.HAS_SKIMAGE, reason="scikit-image not installed")
def test_threshold_segmentation_finds_both_objects():
    options = segmentation.SegmentationOptions(
        method=segmentation.SegmentationMethod.THRESHOLD,
        min_area=50,
        remove_border_objects=True,
    )
    labels = segmentation.segment_image(_two_squares(), options)
    assert labels.shape == (128, 128)
    assert labels.max() == 2

    rois = segmentation.masks_to_roi_data(labels, min_area=50)
    assert len(rois) == 2
    assert all(len(np.asarray(roi["coordinates"])) >= 3 for roi in rois)

    boundary_rois = segmentation.create_tumor_boundary_rois(labels, inner_distance=3, outer_distance=8)
    assert len(boundary_rois) > 0


def test_check_available_methods_reports_threshold():
    available = segmentation.check_available_methods()
    assert available["threshold"] == segmentation.HAS_SKIMAGE


def test_to_2d_grayscale_reduces_rgb():
    rgb = np.dstack([_two_squares().astype(np.uint8)] * 3)
    gray = preprocessing.to_2d_grayscale_for_curvelets(rgb)
    assert gray.shape == (128, 128)


def test_preprocess_image_gaussian_and_threshold():
    options = preprocessing.PreprocessingOptions(
        apply_gaussian=True,
        gaussian_sigma=1.0,
        apply_threshold=True,
        threshold_method=preprocessing.ThresholdMethod.OTSU,
    )
    out = preprocessing.preprocess_image(_two_squares(), options)
    assert out.shape == (128, 128)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_load_image_with_bioformats_reads_tiff(tmp_path):
    from skimage import io

    path = tmp_path / "squares.tif"
    io.imsave(path, _two_squares().astype(np.uint16), check_contrast=False)
    data, metadata = preprocessing.load_image_with_bioformats(str(path))
    assert np.squeeze(data).shape == (128, 128)
    assert "source" in metadata
