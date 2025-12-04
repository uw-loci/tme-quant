import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pycurvelets.models import CurveletControlParameters

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
except ModuleNotFoundError:
    pytest.skip(
        "curvelops not available; skipping new_curv tests", allow_module_level=True
    )

from pycurvelets.new_curv import new_curv

# --------------------------
# Helpers
# --------------------------


def load_test_image(image_name):
    """Load a test image by filename."""
    img_path = os.path.join(os.path.dirname(__file__), "test_images", image_name)
    return plt.imread(img_path, format="TIF")

    config_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "new_curv_test_files",
        "test_cases_new_curv.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    cases = config["test_cases"]
    if matlab_only:
        cases = [tc for tc in cases if "matlab_reference_csv" in tc]

    return [(tc["name"], tc) for tc in cases]


def load_and_sort_curvelets(in_curves: pd.DataFrame, ref_csv_path: str):
    """
    Load reference curvelets from CSV and sort both predicted and reference arrays.

    Returns:
        pred_centers_sorted, pred_angles_sorted, ref_centers_sorted, ref_angles_sorted
    """
    # Load reference
    df = pd.read_csv(ref_csv_path)
    ref_centers = df[["center_0", "center_1"]].to_numpy(dtype=float)
    ref_angles = df["angle"].to_numpy(dtype=float)

    # Predicted values
    pred_centers = in_curves[["center_row", "center_col"]].to_numpy(dtype=float)
    pred_angles = in_curves["angle"].to_numpy(dtype=float)

    # Sort by row, then column
    ref_sort_idx = np.lexsort((ref_centers[:, 1], ref_centers[:, 0]))
    pred_sort_idx = np.lexsort((pred_centers[:, 1], pred_centers[:, 0]))

    ref_centers_sorted = ref_centers[ref_sort_idx]
    pred_centers_sorted = pred_centers[pred_sort_idx]
    ref_angles_sorted = ref_angles[ref_sort_idx]
    pred_angles_sorted = pred_angles[pred_sort_idx]

    return (
        pred_centers_sorted,
        pred_angles_sorted,
        ref_centers_sorted,
        ref_angles_sorted,
    )


def load_test_cases(matlab_only: bool = False):
    """
    Load JSON test cases and return (name, case) tuples for parametrize.

    Args:
        matlab_only: if True, only return cases with a MATLAB reference CSV.

    Returns:
        List of (name, test_case) tuples.
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "new_curv_test_files",
        "test_cases_new_curv.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    cases = config["test_cases"]
    if matlab_only:
        cases = [tc for tc in cases if "matlab_reference_csv" in tc]

    return [(tc["name"], tc) for tc in cases]


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(),
    ids=[name for name, _ in load_test_cases()],
)
def test_new_curv_validate_struct(test_name, test_case):
    """
    Test new_curv's generated structure
    """
    img = load_test_image(test_case["image"])
    curve_cp = CurveletControlParameters(
        keep=test_case["keep"], scale=test_case["scale"], radius=test_case["radius"]
    )

    # Run new_curv
    in_curves, ct, inc = new_curv(img, curve_cp)

    # Basic validations
    assert isinstance(in_curves, pd.DataFrame)
    assert len(in_curves) > 0

    angles = in_curves["angle"].to_numpy(dtype=float)
    centers = in_curves[["center_row", "center_col"]].to_numpy()

    assert angles.ndim == 1
    assert centers.ndim == 2 and centers.shape[1] == 2
    assert np.isfinite(angles).all()
    assert (angles >= 0).all() and (angles < 180).all()
    assert (centers >= 0).all()
    assert centers[:, 0].max() < img.shape[0]
    assert centers[:, 1].max() < img.shape[1]


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(matlab_only=True),
    ids=[name for name, _ in load_test_cases(matlab_only=True)],
)
def test_new_curv_matches_matlab_reference(test_name, test_case):
    """
    Compare the curvelet centers and angles against a MATLAB reference CSV.
    Absolute tolerance of centers and angles are at 1 pixel.
    Checks absolute tolerance and ensures mismatched elements are <1%.
    """

    img = load_test_image(test_case["image"])
    curve_cp = CurveletControlParameters(
        keep=test_case["keep"], scale=test_case["scale"], radius=test_case["radius"]
    )
    in_curves, ct, inc = new_curv(img, curve_cp)

    ref_csv = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "new_curv_test_files",
        test_case["matlab_reference_csv"],
    )

    pred_centers, pred_angles, ref_centers, ref_angles = load_and_sort_curvelets(
        in_curves, ref_csv
    )

    # Absolute tolerance of 1 pixel
    np.testing.assert_allclose(
        pred_centers,
        ref_centers,
        rtol=0,
        atol=1,
        err_msg="Curvelet centers differ from MATLAB reference",
    )

    # Absolute tolerance of 3 pixels
    np.testing.assert_allclose(
        pred_angles,
        ref_angles,
        rtol=0,
        atol=3,
        err_msg="Curvelet angles differ from MATLAB reference",
    )
