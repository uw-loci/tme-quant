# -*- coding: utf-8 -*-
"""
Tests for fiber_analysis.utils.fiber_dataframe_utils.

Mirrors the intent of pycurvelets test_get_ct.py and test_process_image.py
for the tme_quant-native adaptation of process_fibers.
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running directly or via pytest from the project root
# ---------------------------------------------------------------------------
_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from unittest.mock import patch

from tme_quant.fiber_analysis.config import FiberFeatureParams
from tme_quant.fiber_analysis.utils.fiber_dataframe_utils import (
    build_fiber_structure_from_curvelets,
    compute_fiber_density_and_alignment,
    flatten_numeric,
    round_mlab,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_fiber_df():
    """15 synthetic fibres arranged in a 5×3 grid."""
    rng = np.random.default_rng(42)
    rows = np.tile(np.arange(0, 150, 30), 3).astype(float)
    cols = np.repeat(np.arange(0, 90, 30), 5).astype(float)
    angles = rng.uniform(0, 180, size=15)
    return pd.DataFrame({"center_row": rows, "center_col": cols, "angle": angles})


@pytest.fixture
def default_params():
    return FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32)


@pytest.fixture
def aligned_fiber_df():
    """10 fibres all pointing at ~45°, clustered together."""
    n = 10
    rows = np.linspace(10, 50, n)
    cols = np.linspace(10, 50, n)
    angles = np.full(n, 45.0) + np.random.default_rng(7).uniform(-1, 1, n)
    return pd.DataFrame({"center_row": rows, "center_col": cols, "angle": angles})


# ---------------------------------------------------------------------------
# round_mlab
# ---------------------------------------------------------------------------

class TestRoundMlab:
    def test_half_rounds_away_from_zero_positive(self):
        assert round_mlab(0.5) == 1

    def test_half_rounds_away_from_zero_negative(self):
        # floor(-0.5 + 0.5) = floor(0) = 0 — same behaviour as MATLAB
        assert round_mlab(-0.5) == 0

    def test_integer_unchanged(self):
        assert round_mlab(3) == 3

    def test_list_input(self):
        result = round_mlab([0.5, 1.5, 2.4])
        assert result == [1, 2, 2]

    def test_float_down(self):
        assert round_mlab(2.3) == 2

    def test_float_up(self):
        assert round_mlab(2.7) == 3


# ---------------------------------------------------------------------------
# compute_fiber_density_and_alignment — structure checks
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_two_dataframes(self, small_fiber_df, default_params):
        density_df, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        assert isinstance(density_df, pd.DataFrame)
        assert isinstance(alignment_df, pd.DataFrame)

    def test_row_count_matches_input(self, small_fiber_df, default_params):
        density_df, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        assert len(density_df) == len(small_fiber_df)
        assert len(alignment_df) == len(small_fiber_df)

    def test_density_has_nine_columns(self, small_fiber_df, default_params):
        density_df, _ = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        assert density_df.shape[1] == 9

    def test_alignment_has_nine_columns(self, small_fiber_df, default_params):
        _, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        assert alignment_df.shape[1] == 9

    def test_density_column_names_reflect_params(self, small_fiber_df):
        params = FiberFeatureParams(minimum_nearest_fibers=3, minimum_box_size=16)
        density_df, _ = compute_fiber_density_and_alignment(small_fiber_df, params)
        assert "distance_to_nearest_3_fibers" in density_df.columns
        assert "distance_to_nearest_6_fibers" in density_df.columns
        assert "fibers_within_box_density16" in density_df.columns
        assert "fibers_within_box_density32" in density_df.columns

    def test_alignment_column_names_reflect_params(self, small_fiber_df):
        params = FiberFeatureParams(minimum_nearest_fibers=3, minimum_box_size=16)
        _, alignment_df = compute_fiber_density_and_alignment(small_fiber_df, params)
        assert "alignment_of_nearest_3_fibers" in alignment_df.columns
        assert "fiber_alignment_in_box_16" in alignment_df.columns

    def test_mean_std_columns_present(self, small_fiber_df, default_params):
        density_df, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        assert "distance_to_nearest_fiber_mean" in density_df.columns
        assert "distance_to_nearest_fiber_std" in density_df.columns
        assert "alignment_mean" in alignment_df.columns
        assert "alignment_std" in alignment_df.columns


# ---------------------------------------------------------------------------
# compute_fiber_density_and_alignment — value checks
# ---------------------------------------------------------------------------

class TestOutputValues:
    def test_density_nonnegative(self, small_fiber_df, default_params):
        density_df, _ = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        knn_cols = [c for c in density_df.columns if "nearest" in c]
        vals = density_df[knn_cols].values
        # Non-NaN values must be non-negative
        assert np.all(vals[~np.isnan(vals)] >= 0)

    def test_alignment_in_unit_interval(self, small_fiber_df, default_params):
        _, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        vals = alignment_df.values
        finite = vals[~np.isnan(vals)]
        assert np.all(finite >= 0.0)
        # Allow 1e-10 tolerance for floating-point rounding in circ_r
        assert np.all(finite <= 1.0 + 1e-10)

    def test_aligned_fibres_have_high_alignment(self, aligned_fiber_df, default_params):
        """Fibres all pointing the same direction should yield high R."""
        _, alignment_df = compute_fiber_density_and_alignment(
            aligned_fiber_df, default_params
        )
        # The kNN alignment for 2 nearest fibres should be close to 1.0
        col = f"alignment_of_nearest_{default_params.minimum_nearest_fibers}_fibers"
        r_vals = alignment_df[col].dropna().values
        assert r_vals.mean() > 0.90, f"Expected high alignment, got {r_vals.mean():.3f}"

    def test_mean_is_mean_of_knn_columns(self, small_fiber_df, default_params):
        _, alignment_df = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        knn_cols = [c for c in alignment_df.columns if "nearest" in c]
        expected_mean = alignment_df[knn_cols].mean(axis=1)
        pd.testing.assert_series_equal(
            alignment_df["alignment_mean"],
            expected_mean,
            check_names=False,
            atol=1e-10,
        )

    def test_box_density_at_least_one(self, small_fiber_df, default_params):
        """Each fibre's box must contain at least itself."""
        density_df, _ = compute_fiber_density_and_alignment(
            small_fiber_df, default_params
        )
        box_cols = [c for c in density_df.columns if "box_density" in c]
        for col in box_cols:
            assert (density_df[col] >= 1).all(), f"Box {col} has zero-count rows"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_dataframe_raises(self, default_params):
        with pytest.raises(ValueError, match="non-empty"):
            compute_fiber_density_and_alignment(pd.DataFrame(), default_params)

    def test_missing_angle_column_raises(self, default_params):
        df = pd.DataFrame({"center_row": [1.0], "center_col": [1.0]})
        with pytest.raises(ValueError, match="angle"):
            compute_fiber_density_and_alignment(df, default_params)

    def test_pycurvelets_column_aliases_accepted(self, default_params):
        """center_1 / center_2 column aliases from pycurvelets are handled."""
        df = pd.DataFrame({
            "center_1": np.arange(5, dtype=float) * 10,
            "center_2": np.zeros(5),
            "angle":    np.linspace(0, 90, 5),
        })
        density_df, alignment_df = compute_fiber_density_and_alignment(df, default_params)
        assert len(density_df) == 5
        assert len(alignment_df) == 5

    def test_single_fiber_returns_nan_knn(self, default_params):
        """With only 1 fibre, kNN distances are NaN (not enough neighbours)."""
        df = pd.DataFrame({"center_row": [10.0], "center_col": [20.0], "angle": [45.0]})
        density_df, alignment_df = compute_fiber_density_and_alignment(df, default_params)
        knn_cols = [c for c in density_df.columns if "nearest_2" in c]
        assert np.isnan(density_df[knn_cols[0]].iloc[0])


# ---------------------------------------------------------------------------
# FiberFeatureParams
# ---------------------------------------------------------------------------

class TestFiberFeatureParams:
    def test_defaults(self):
        p = FiberFeatureParams()
        assert p.minimum_nearest_fibers == 2
        assert p.minimum_box_size == 32
        assert p.fiber_midpoint_estimate == 1

    def test_to_dict(self):
        p = FiberFeatureParams(minimum_nearest_fibers=4, minimum_box_size=64)
        d = p.to_dict()
        assert d["minimum_nearest_fibers"] == 4
        assert d["minimum_box_size"] == 64
        assert "fiber_midpoint_estimate" in d


# ---------------------------------------------------------------------------
# flatten_numeric
# ---------------------------------------------------------------------------

class TestFlattenNumeric:
    def test_numeric_dtype_series(self):
        s = pd.Series([1.0, 2.5, 3.7])
        result = flatten_numeric(s)
        np.testing.assert_array_almost_equal(result, [1.0, 2.5, 3.7])
        assert result.dtype == float

    def test_object_dtype_single_element_arrays(self):
        s = pd.Series([np.array([1.0]), np.array([2.0]), np.array([3.0])])
        result = flatten_numeric(s)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_mixed_scalars_and_arrays(self):
        s = pd.Series([1.0, np.array([2.0]), [3.0]])
        result = flatten_numeric(s)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_integer_series_converted_to_float(self):
        s = pd.Series([1, 2, 3])
        result = flatten_numeric(s)
        assert result.dtype == float

    def test_output_is_1d(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = flatten_numeric(s)
        assert result.ndim == 1


# ---------------------------------------------------------------------------
# build_fiber_structure_from_curvelets — structure tests (mocked inner call)
# ---------------------------------------------------------------------------

def _make_synthetic_fiber_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "center_row": rng.uniform(0, 200, n),
        "center_col": rng.uniform(0, 200, n),
        "angle":      rng.uniform(0, 180, n),
        "width":      rng.uniform(1, 5, n),
    })


_MOCK_COEFFICIENTS = object()  # sentinel


class TestBuildFiberStructureFromCurvelets:
    """All tests mock extract_curvelet_fiber_candidates so curvelops is not required."""

    @pytest.fixture
    def mock_extractor(self):
        fiber_df = _make_synthetic_fiber_df(20)
        # Patch at the source module — build_fiber_structure_from_curvelets does a
        # lazy local import so patching the source attribute is the correct target.
        with patch(
            "tme_quant.fiber_analysis.utils.curvelet_utils"
            ".extract_curvelet_fiber_candidates",
            return_value=(fiber_df, _MOCK_COEFFICIENTS, None),
        ) as m:
            yield m, fiber_df

    @pytest.fixture
    def mock_extractor_empty(self):
        empty_df = pd.DataFrame(columns=["center_row", "center_col", "angle", "width"])
        with patch(
            "tme_quant.fiber_analysis.utils.curvelet_utils"
            ".extract_curvelet_fiber_candidates",
            return_value=(empty_df, _MOCK_COEFFICIENTS, None),
        ) as m:
            yield m

    def test_returns_four_tuple(self, mock_extractor):
        _, fiber_df = mock_extractor
        image = np.zeros((64, 64), dtype=np.float32)
        result = build_fiber_structure_from_curvelets(image)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_output_types(self, mock_extractor):
        _, _ = mock_extractor
        image = np.zeros((64, 64), dtype=np.float32)
        fiber_structure, density_df, alignment_df, coefficients = (
            build_fiber_structure_from_curvelets(image)
        )
        assert isinstance(fiber_structure, pd.DataFrame)
        assert isinstance(density_df, pd.DataFrame)
        assert isinstance(alignment_df, pd.DataFrame)

    def test_density_alignment_have_9_columns(self, mock_extractor):
        _, _ = mock_extractor
        image = np.zeros((64, 64), dtype=np.float32)
        _, density_df, alignment_df, _ = build_fiber_structure_from_curvelets(image)
        assert density_df.shape[1] == 9
        assert alignment_df.shape[1] == 9

    def test_empty_fiber_structure_returns_empty_dfs(self, mock_extractor_empty):
        image = np.zeros((64, 64), dtype=np.float32)
        fiber_structure, density_df, alignment_df, coefficients = (
            build_fiber_structure_from_curvelets(image)
        )
        assert len(fiber_structure) == 0
        assert len(density_df) == 0
        assert len(alignment_df) == 0
        assert coefficients is _MOCK_COEFFICIENTS

    def test_default_feature_params_applied(self, mock_extractor):
        _, _ = mock_extractor
        image = np.zeros((64, 64), dtype=np.float32)
        # No feature_params supplied — should not raise
        _, density_df, _, _ = build_fiber_structure_from_curvelets(image)
        assert "distance_to_nearest_2_fibers" in density_df.columns

    def test_custom_params_respected(self, mock_extractor):
        _, _ = mock_extractor
        image = np.zeros((64, 64), dtype=np.float32)
        params = FiberFeatureParams(minimum_nearest_fibers=3, minimum_box_size=16)
        _, density_df, _, _ = build_fiber_structure_from_curvelets(
            image, feature_params=params
        )
        assert "distance_to_nearest_3_fibers" in density_df.columns


# ---------------------------------------------------------------------------
# Real-dataset tests — ported from pycurvelets tests/test_get_ct.py
#
# Data: tests/test_results/process_image_test_files/
#   real1_fiber_structure.csv  — curvelet fiber candidates (center_1/center_2/angle)
#   real1_density_df.csv       — reference density output (9 cols, no header)
#   real1_alignment_df.csv     — reference alignment output (9 cols, no header)
# ---------------------------------------------------------------------------

_REPO_TEST_DIR  = pathlib.Path(__file__).parent.parent.parent.parent / "tests"
_CT_DATA_DIR    = _REPO_TEST_DIR / "test_results" / "process_image_test_files"
_CT_DATA_MISSING = not _CT_DATA_DIR.exists()


@pytest.mark.skipif(_CT_DATA_MISSING, reason="pycurvelets test data not found")
class TestBuildFiberStructureRealData:
    """
    Validates compute_fiber_density_and_alignment against MATLAB reference outputs
    using the real1 fiber structure.  Does not require curvelops — uses the
    pre-extracted fiber_structure CSV as input to compute_fiber_density_and_alignment
    directly (bypassing the curvelet candidate extraction step).

    Ported from pycurvelets ``tests/test_get_ct.py``.
    """

    @pytest.fixture(scope="class")
    def real1_data(self):
        fiber_path   = _CT_DATA_DIR / "real1_fiber_structure.csv"
        density_path = _CT_DATA_DIR / "real1_density_df.csv"
        align_path   = _CT_DATA_DIR / "real1_alignment_df.csv"

        fiber_df      = pd.read_csv(fiber_path)
        ref_density   = pd.read_csv(density_path, header=None)
        ref_alignment = pd.read_csv(align_path, header=None)
        return fiber_df, ref_density, ref_alignment

    def test_fiber_structure_has_required_columns(self, real1_data):
        fiber_df, _, _ = real1_data
        required = {"angle"}
        # Accepts center_1/center_2 or center_row/center_col
        has_coords = (
            {"center_row", "center_col"}.issubset(fiber_df.columns)
            or {"center_1", "center_2"}.issubset(fiber_df.columns)
        )
        assert has_coords
        assert required.issubset(fiber_df.columns)

    def test_density_shape_matches_reference(self, real1_data):
        fiber_df, ref_density, _ = real1_data
        params = FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32)
        density_df, _ = compute_fiber_density_and_alignment(fiber_df, params)
        assert density_df.shape[0] == ref_density.shape[0], (
            f"Row count: got {density_df.shape[0]}, expected {ref_density.shape[0]}"
        )
        assert density_df.shape[1] == ref_density.shape[1] == 9

    def test_alignment_shape_matches_reference(self, real1_data):
        fiber_df, _, ref_alignment = real1_data
        params = FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32)
        _, alignment_df = compute_fiber_density_and_alignment(fiber_df, params)
        assert alignment_df.shape[0] == ref_alignment.shape[0]
        assert alignment_df.shape[1] == ref_alignment.shape[1] == 9

    def test_density_values_match_reference(self, real1_data):
        fiber_df, ref_density, _ = real1_data
        params = FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32)
        density_df, _ = compute_fiber_density_and_alignment(fiber_df, params)
        np.testing.assert_allclose(
            density_df.values,
            ref_density.values,
            rtol=0.05,
            atol=15,
            err_msg="density_df values differ from MATLAB reference",
        )

    def test_alignment_values_match_reference(self, real1_data):
        fiber_df, _, ref_alignment = real1_data
        params = FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32)
        _, alignment_df = compute_fiber_density_and_alignment(fiber_df, params)
        np.testing.assert_allclose(
            alignment_df.values,
            ref_alignment.values,
            rtol=0.05,
            atol=0.2,
            err_msg="alignment_df values differ from MATLAB reference",
        )
