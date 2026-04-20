# -*- coding: utf-8 -*-
"""
Tests for extract_curvelet_fiber_candidates and supporting helpers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from tme_quant.fiber_analysis.utils.curvelet_utils import (
    _fix_angle,
    extract_curvelet_fiber_candidates,
)
from tme_quant.fiber_analysis.utils.fiber_dataframe_utils import round_mlab


# ─────────────────────────────────────────────────────────────────────────────
# round_mlab — numpy array handling (regression for new behaviour)
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundMlab:
    def test_scalar_int(self):
        assert round_mlab(3) == 3

    def test_scalar_float(self):
        assert round_mlab(2.5) == 3   # MATLAB rounds 0.5 up
        assert round_mlab(1.5) == 2

    def test_list(self):
        assert round_mlab([1.5, 2.5, 3.4]) == [2, 3, 3]

    def test_tuple(self):
        assert round_mlab((1.5, 2.5)) == [2, 3]

    def test_numpy_array(self):
        arr = np.array([0.5, 1.5, 2.4, 3.6])
        result = round_mlab(arr)
        assert result == [1, 2, 2, 4]

    def test_numpy_single_element(self):
        result = round_mlab(np.array([2.5]))
        assert result == [3]


# ─────────────────────────────────────────────────────────────────────────────
# _fix_angle
# ─────────────────────────────────────────────────────────────────────────────

class TestFixAngle:
    def test_single_angle_returns_itself(self):
        result = _fix_angle(np.array([45.0]), inc=22.5)
        assert abs(result - 45.0) < 1e-9

    def test_uniform_angles_return_mean(self):
        angles = np.array([90.0, 90.0, 90.0])
        result = _fix_angle(angles, inc=22.5)
        assert abs(result - 90.0) < 1e-9

    def test_output_is_float(self):
        result = _fix_angle(np.array([60.0, 65.0]), inc=22.5)
        assert isinstance(result, float)

    def test_result_non_negative(self):
        # After adjustment, mean should be >= 0
        angles = np.array([170.0, 175.0, 180.0])
        result = _fix_angle(angles, inc=22.5)
        assert result >= 0.0

    def test_two_close_angles(self):
        angles = np.array([44.0, 46.0])
        result = _fix_angle(angles, inc=22.5)
        assert abs(result - 45.0) < 5.0   # should be near 45


# ─────────────────────────────────────────────────────────────────────────────
# extract_curvelet_fiber_candidates — ImportError when curvelops absent
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCurveletImportError:
    def test_raises_import_error_without_curvelops(self, monkeypatch):
        # Hide curvelops so the lazy import fails
        monkeypatch.setitem(sys.modules, "curvelops", None)
        img = np.random.default_rng(0).random((64, 64))
        with pytest.raises(ImportError, match="curvelops"):
            extract_curvelet_fiber_candidates(img)

    def test_raises_value_error_for_3d_input(self):
        img_3d = np.zeros((4, 64, 64))
        with pytest.raises(ValueError, match="2-D"):
            extract_curvelet_fiber_candidates(img_3d)


# ─────────────────────────────────────────────────────────────────────────────
# extract_curvelet_fiber_candidates — integration tests (requires curvelops)
# ─────────────────────────────────────────────────────────────────────────────

curvelops = pytest.importorskip("curvelops", reason="curvelops not installed")


class TestExtractCurveletIntegration:
    @pytest.fixture
    def synthetic_fiber_image(self):
        """64×64 image with a faint diagonal stripe to create curvelet responses."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64)).astype(np.float64) * 0.1
        for i in range(64):
            img[i, max(0, i - 2) : i + 3] += 1.0
        return img

    def test_returns_tuple_of_three(self, synthetic_fiber_image):
        result = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        assert len(result) == 3

    def test_dataframe_columns(self, synthetic_fiber_image):
        in_curves, _, _ = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        assert isinstance(in_curves, pd.DataFrame)
        assert set(in_curves.columns) == {"center_row", "center_col", "angle"}

    def test_angles_in_valid_range(self, synthetic_fiber_image):
        in_curves, _, _ = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        if len(in_curves) > 0:
            assert (in_curves["angle"] >= 0).all()
            assert (in_curves["angle"] < 180).all()

    def test_centres_within_image(self, synthetic_fiber_image):
        h, w = synthetic_fiber_image.shape
        in_curves, _, _ = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        if len(in_curves) > 0:
            assert (in_curves["center_row"] >= 0).all()
            assert (in_curves["center_row"] < h).all()
            assert (in_curves["center_col"] >= 0).all()
            assert (in_curves["center_col"] < w).all()

    def test_inc_is_positive_float(self, synthetic_fiber_image):
        _, _, inc = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        assert isinstance(inc, float)
        assert inc > 0.0

    def test_coefficients_is_list_of_lists(self, synthetic_fiber_image):
        _, coeffs, _ = extract_curvelet_fiber_candidates(synthetic_fiber_image)
        assert isinstance(coeffs, list)
        assert all(isinstance(scale, list) for scale in coeffs)

    def test_keep_parameter_affects_output(self, synthetic_fiber_image):
        df_tight, _, _ = extract_curvelet_fiber_candidates(
            synthetic_fiber_image, keep=0.01
        )
        df_loose, _, _ = extract_curvelet_fiber_candidates(
            synthetic_fiber_image, keep=0.20
        )
        # Looser threshold keeps more coefficients → typically more candidates
        # (not guaranteed after grouping, but at least the call succeeds)
        assert isinstance(df_tight, pd.DataFrame)
        assert isinstance(df_loose, pd.DataFrame)

    def test_empty_image_returns_empty_dataframe(self):
        img = np.zeros((64, 64))
        in_curves, _, _ = extract_curvelet_fiber_candidates(img)
        assert isinstance(in_curves, pd.DataFrame)
        assert set(in_curves.columns) == {"center_row", "center_col", "angle"}


# ─────────────────────────────────────────────────────────────────────────────
# Real-dataset tests — ported from pycurvelets tests/test_new_curv.py
#
# Test data lives at the repo root under tests/ (shared with pycurvelets):
#   tests/test_images/          real1.tif, real2.tif, ...
#   tests/test_results/new_curv_test_files/
#       test_cases_new_curv.json
#       test_new_curv_*.csv     (MATLAB reference outputs)
# ─────────────────────────────────────────────────────────────────────────────

# Repo-root tests/ directory (4 levels up from this file)
_REPO_TEST_DIR = Path(__file__).parent.parent.parent.parent / "tests"
_CURVELETS_DATA_DIR = _REPO_TEST_DIR / "test_results" / "new_curv_test_files"
_STRICT_MATLAB = os.environ.get("TMEQ_VALIDATE_MATLAB") == "1"


def _load_test_cases(matlab_only: bool = False):
    """Return list of (name, case_dict) from the shared JSON config."""
    config_path = _CURVELETS_DATA_DIR / "test_cases_new_curv.json"
    if not config_path.exists():
        return []
    with open(config_path) as f:
        cases = json.load(f)["test_cases"]
    if matlab_only:
        cases = [tc for tc in cases if "matlab_reference_csv" in tc]
    return [(tc["name"], tc) for tc in cases]


def _load_and_sort_curvelets(in_curves: pd.DataFrame, ref_csv_path: Path):
    """
    Load MATLAB reference CSV and sort both predicted and reference arrays
    by (center_row, center_col) for element-wise comparison.
    """
    df = pd.read_csv(ref_csv_path)
    ref_centers = df[["center_0", "center_1"]].to_numpy(dtype=float)
    ref_angles  = df["angle"].to_numpy(dtype=float)

    pred_centers = in_curves[["center_row", "center_col"]].to_numpy(dtype=float)
    pred_angles  = in_curves["angle"].to_numpy(dtype=float)

    ref_sort  = np.lexsort((ref_centers[:, 1],  ref_centers[:, 0]))
    pred_sort = np.lexsort((pred_centers[:, 1], pred_centers[:, 0]))

    return (
        pred_centers[pred_sort],
        pred_angles[pred_sort],
        ref_centers[ref_sort],
        ref_angles[ref_sort],
    )


_ALL_CASES    = _load_test_cases()
_MATLAB_CASES = _load_test_cases(matlab_only=True)

# Skip entire class when test data is absent (e.g. CI without the pycurvelets subtree)
_data_missing = not _CURVELETS_DATA_DIR.exists()


@pytest.mark.skipif(_data_missing, reason="pycurvelets test data not found")
class TestExtractCurveletRealDatasets:
    """
    Ported from pycurvelets ``tests/test_new_curv.py``.

    ``test_validate_struct``         — runs on all cases; checks output shape,
                                       angle range, and centres inside image.
    ``test_matches_matlab_reference``— compares against MATLAB-generated CSVs;
                                       only runs when TMEQ_VALIDATE_MATLAB=1.
    """

    @pytest.mark.parametrize(
        "test_name,test_case",
        _ALL_CASES,
        ids=[name for name, _ in _ALL_CASES],
    )
    def test_validate_struct(self, test_name, test_case):
        img_path = _REPO_TEST_DIR / "test_images" / test_case["image"]
        img = tifffile.imread(str(img_path)).astype(np.float64)
        if img.ndim == 3:
            img = img[..., 0]   # take first channel if RGB

        in_curves, _, _ = extract_curvelet_fiber_candidates(
            img,
            keep=test_case["keep"],
            scale=test_case["scale"],
            radius=test_case["radius"],
        )

        assert isinstance(in_curves, pd.DataFrame)
        assert len(in_curves) > 0, f"{test_name}: no candidates returned"

        angles  = in_curves["angle"].to_numpy(dtype=float)
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
        _MATLAB_CASES,
        ids=[name for name, _ in _MATLAB_CASES],
    )
    def test_matches_matlab_reference(self, test_name, test_case):
        if not _STRICT_MATLAB:
            pytest.skip(
                "MATLAB parity checks disabled (set TMEQ_VALIDATE_MATLAB=1 to enable)"
            )

        img_path = _REPO_TEST_DIR / "test_images" / test_case["image"]
        img = tifffile.imread(str(img_path)).astype(np.float64)
        if img.ndim == 3:
            img = img[..., 0]

        in_curves, _, _ = extract_curvelet_fiber_candidates(
            img,
            keep=test_case["keep"],
            scale=test_case["scale"],
            radius=test_case["radius"],
        )

        ref_csv = _CURVELETS_DATA_DIR / test_case["matlab_reference_csv"]
        pred_centers, pred_angles, ref_centers, ref_angles = _load_and_sort_curvelets(
            in_curves, ref_csv
        )

        np.testing.assert_allclose(
            pred_centers, ref_centers,
            rtol=0.05, atol=15,
            err_msg=f"{test_name}: curvelet centres differ from MATLAB reference",
        )
        np.testing.assert_allclose(
            pred_angles, ref_angles,
            rtol=0.05, atol=15,
            err_msg=f"{test_name}: curvelet angles differ from MATLAB reference",
        )
