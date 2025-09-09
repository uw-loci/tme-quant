import pytest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycurvelets.new_curv import new_curv

# By default, skip curvelops-dependent tests (e.g., on CI). Enable locally with:
#   TMEQ_RUN_CURVELETS=1 pytest -q
if os.environ.get("TMEQ_RUN_CURVELETS") != "1":
    pytest.skip(
        "curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
        allow_module_level=True,
    )

# Optional: attempt import; skip module if curvelet backend is missing
try:
    from pycurvelets.new_curv import new_curv  # type: ignore
except Exception:
    pytest.skip(
        "curvelops not available; skipping new_curv tests", allow_module_level=True
    )


@pytest.fixture(scope="module")
def standard_test_image():
    """Load a standard test image once for the module."""
    img_path = os.path.join(os.path.dirname(__file__), "test_images", "real1.tif")
    img = plt.imread(img_path, format="TIF")
    return img


def test_new_curv_generates_nonempty_curvelets(standard_test_image):
    """
    Verify that `new_curv` returns a non-empty list of curvelets with valid angles
    and center coordinates for typical parameters.
    """
    img = standard_test_image
    in_curves, ct, inc = new_curv(img, {"keep": 0.01, "scale": 1, "radius": 3})

    assert isinstance(in_curves, list)
    assert len(in_curves) > 0

    angles = np.asarray([c["angle"] for c in in_curves], dtype=float)
    centers = np.asarray([c["center"] for c in in_curves], dtype=float)

    assert angles.ndim == 1
    assert centers.ndim == 2 and centers.shape[1] == 2
    assert np.isfinite(angles).all()
    assert (angles >= 0).all() and (angles < 180).all()
    assert (centers >= 0).all()
    assert centers[:, 0].max() < img.shape[0]
    assert centers[:, 1].max() < img.shape[1]


def test_new_curv_matches_matlab_reference(standard_test_image):
    """
    Compare the curvelet centers and angles against a MATLAB reference CSV.
    Checks absolute tolerance and ensures mismatched elements are <1%.
    Absolute tolerance for centers: 50 px
    Absolute tolerance for angles: 30 deg
    """
    img = standard_test_image
    in_curves, ct, inc = new_curv(img, {"keep": 0.01, "scale": 1, "radius": 3})

    ref_csv = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "new_curv_test_files",
        "test_new_curvs_real1.csv",
    )

    if not os.path.exists(ref_csv):
        assert inc == 5.625
        return

    df = pd.read_csv(ref_csv)
    ref_centers = df[["center_0", "center_1"]].to_numpy(dtype=float)
    ref_angles = df["angle"].to_numpy(dtype=float)

    pred_centers = np.asarray([d["center"] for d in in_curves], dtype=float)
    pred_angles = np.asarray([d["angle"] for d in in_curves], dtype=float)

    ref_sort_idx = np.lexsort((ref_centers[:, 1], ref_centers[:, 0]))
    pred_sort_idx = np.lexsort((pred_centers[:, 1], pred_centers[:, 0]))

    ref_centers_sorted = ref_centers[ref_sort_idx]
    pred_centers_sorted = pred_centers[pred_sort_idx]
    ref_angles_sorted = ref_angles[ref_sort_idx]
    pred_angles_sorted = pred_angles[pred_sort_idx]

    # Absolute tolerance of 50 pixels
    np.testing.assert_allclose(
        pred_centers_sorted,
        ref_centers_sorted,
        rtol=0,
        atol=50,
        err_msg="Curvelet centers differ from MATLAB reference",
    )

    # Absolute tolerance of 30 pixels
    np.testing.assert_allclose(
        pred_angles_sorted,
        ref_angles_sorted,
        rtol=0,
        atol=30,
        err_msg="Curvelet angles differ from MATLAB reference",
    )

    angle_mismatches = np.abs(pred_angles_sorted - ref_angles_sorted) > 30
    pct_angle_mismatch = 100 * angle_mismatches.sum() / len(angle_mismatches)
    assert (
        pct_angle_mismatch < 1
    ), f"Too many angle mismatches: {pct_angle_mismatch:.2f}%"

    center_mismatches = np.any(
        np.abs(pred_centers_sorted - ref_centers_sorted) > 5, axis=1
    )
    pct_center_mismatch = 100 * center_mismatches.sum() / len(center_mismatches)
    assert (
        pct_center_mismatch < 1
    ), f"Too many center mismatches: {pct_center_mismatch:.2f}%"


@pytest.mark.parametrize(
    "params",
    [
        {"keep": 0.01, "scale": 1, "radius": 3},
        {"keep": 0.1, "scale": 2, "radius": 5},
    ],
)
def test_new_curv_output_consistency(standard_test_image, params):
    """
    Ensure different curvelet parameters produce non-empty, valid outputs.
    """
    img = standard_test_image
    in_curves, ct, inc = new_curv(img, params)

    assert len(in_curves) > 0
    angles = np.asarray([c["angle"] for c in in_curves], dtype=float)
    centers = np.asarray([c["center"] for c in in_curves], dtype=float)

    assert np.isfinite(angles).all()
    assert centers.ndim == 2 and centers.shape[1] == 2
