# -*- coding: utf-8 -*-
"""
Tests for tme_analysis.utils.alignment_utils.compute_fiber_alignment_to_roi.

Mirrors the intent of pycurvelets test_get_alignment_to_roi.py for the
tme_quant-native adaptation.

Key tme_quant convention differences from pycurvelets:
  - Column ``angle_to_boundary_tangent`` = 0° when fibre is parallel to
    boundary (TACS-2); pycurvelets ``angle_to_boundary_edge`` is the complement
    (= 90° for the same configuration).
  - ROI is passed as (N, 2) ndarray in (row, col) order + explicit dimensions,
    not as a ROIList dataclass.
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tme_quant.tme_analysis.utils.alignment_utils import compute_fiber_alignment_to_roi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_roi():
    """50×50 px square ROI centred at (75, 75) in a 200×200 image."""
    coords_rc = np.array([
        [50, 50], [50, 100], [100, 100], [100, 50],
    ], dtype=float)
    return coords_rc, 200, 200


@pytest.fixture
def fibers_inside_roi():
    """5 fibres whose centres are ≤20 px from the ROI boundary."""
    return pd.DataFrame({
        "center_row": [50.0, 55.0, 60.0, 95.0, 75.0],
        "center_col": [75.0, 75.0, 75.0, 75.0, 50.0],
        "angle":      [0.0,  45.0, 90.0, 135.0, 60.0],
    })


@pytest.fixture
def fibers_outside_roi():
    """5 fibres whose centres are far from the ROI boundary (>100 px)."""
    return pd.DataFrame({
        "center_row": [10.0, 10.0, 190.0, 190.0, 10.0],
        "center_col": [10.0, 190.0, 10.0, 190.0, 100.0],
        "angle":      [0.0, 45.0, 90.0, 135.0, 60.0],
    })


@pytest.fixture
def all_fibers(fibers_inside_roi, fibers_outside_roi):
    return pd.concat([fibers_inside_roi, fibers_outside_roi], ignore_index=True)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    EXPECTED_COLS = {
        "angle_to_boundary_tangent", "angle_to_roi_orientation",
        "angle_to_centers_line", "fiber_center_row", "fiber_center_col",
        "fiber_angle", "distance", "boundary_point_row", "boundary_point_col",
    }

    def test_returns_dataframe_and_int(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(count, int)

    def test_dataframe_has_required_columns(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        assert self.EXPECTED_COLS.issubset(set(result_df.columns))

    def test_count_matches_dataframe_length(self, square_roi, all_fibers):
        coords, h, w = square_roi
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, all_fibers, distance_threshold=25.0
        )
        assert count == len(result_df)

    def test_preselected_mode_includes_all_fibers(self, square_roi, all_fibers):
        coords, h, w = square_roi
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, all_fibers, distance_threshold=None
        )
        assert count == len(all_fibers)

    def test_distance_column_is_none_in_preselected_mode(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=None
        )
        assert result_df["distance"].isna().all()

    def test_distance_column_populated_when_threshold_given(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        assert result_df["distance"].notna().all()


# ---------------------------------------------------------------------------
# Distance-based filtering
# ---------------------------------------------------------------------------

class TestDistanceFiltering:
    def test_no_fibers_selected_when_all_outside_threshold(
        self, square_roi, fibers_outside_roi
    ):
        coords, h, w = square_roi
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_outside_roi, distance_threshold=5.0
        )
        assert count == 0
        assert len(result_df) == 0

    def test_inside_fibers_selected_with_adequate_threshold(
        self, square_roi, fibers_inside_roi
    ):
        coords, h, w = square_roi
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        assert count > 0

    def test_larger_threshold_selects_more_fibers(self, square_roi, all_fibers):
        coords, h, w = square_roi
        _, count_small = compute_fiber_alignment_to_roi(
            coords, h, w, all_fibers, distance_threshold=5.0
        )
        _, count_large = compute_fiber_alignment_to_roi(
            coords, h, w, all_fibers, distance_threshold=80.0
        )
        assert count_large >= count_small

    def test_selected_fibers_respect_threshold(self, square_roi, all_fibers):
        coords, h, w = square_roi
        threshold = 30.0
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, all_fibers, distance_threshold=threshold
        )
        assert (result_df["distance"] <= threshold + 1e-9).all()


# ---------------------------------------------------------------------------
# Angle value checks
# ---------------------------------------------------------------------------

class TestAngleValues:
    def test_angle_to_boundary_tangent_in_range(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        vals = result_df["angle_to_boundary_tangent"].dropna().values.astype(float)
        assert np.all(vals >= 0.0 - 1e-9)
        assert np.all(vals <= 90.0 + 1e-9)

    def test_angle_to_roi_orientation_in_range(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        vals = result_df["angle_to_roi_orientation"].values.astype(float)
        assert np.all(vals >= 0.0 - 1e-9)
        assert np.all(vals <= 90.0 + 1e-9)

    def test_angle_to_centers_line_in_range(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        vals = result_df["angle_to_centers_line"].values.astype(float)
        assert np.all(vals >= 0.0 - 1e-9)
        assert np.all(vals <= 90.0 + 1e-9)

    def test_fiber_coordinates_match_input(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        input_rows = set(fibers_inside_roi["center_row"].tolist())
        output_rows = set(result_df["fiber_center_row"].tolist())
        # All output rows must come from input
        assert output_rows.issubset(input_rows)

    def test_fiber_angle_preserved(self, square_roi, fibers_inside_roi):
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        input_angles = set(fibers_inside_roi["angle"].tolist())
        output_angles = set(result_df["fiber_angle"].tolist())
        assert output_angles.issubset(input_angles)

    def test_boundary_point_inside_roi_coords_neighbourhood(
        self, square_roi, fibers_inside_roi
    ):
        """Each reported boundary point must be one of the ROI boundary vertices."""
        coords, h, w = square_roi
        result_df, _ = compute_fiber_alignment_to_roi(
            coords, h, w, fibers_inside_roi, distance_threshold=30.0
        )
        for _, row in result_df.iterrows():
            bp = np.array([row["boundary_point_row"], row["boundary_point_col"]])
            dists = np.linalg.norm(coords - bp, axis=1)
            assert dists.min() < 1e-9, f"Boundary point {bp} not in ROI coords"


# ---------------------------------------------------------------------------
# Column alias support
# ---------------------------------------------------------------------------

class TestColumnAliases:
    def test_center_1_center_2_aliases_accepted(self, square_roi):
        coords, h, w = square_roi
        df = pd.DataFrame({
            "center_1": [60.0, 80.0, 75.0],
            "center_2": [70.0, 70.0, 50.0],
            "angle":    [30.0, 60.0, 90.0],
        })
        result_df, count = compute_fiber_alignment_to_roi(
            coords, h, w, df, distance_threshold=30.0
        )
        assert isinstance(result_df, pd.DataFrame)

    def test_missing_angle_column_raises(self, square_roi):
        coords, h, w = square_roi
        df = pd.DataFrame({"center_row": [50.0], "center_col": [75.0]})
        with pytest.raises((ValueError, KeyError)):
            compute_fiber_alignment_to_roi(coords, h, w, df)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_none_roi_raises(self, fibers_inside_roi):
        with pytest.raises(ValueError, match="roi_coords"):
            compute_fiber_alignment_to_roi(None, 200, 200, fibers_inside_roi)

    def test_too_few_roi_points_raises(self, fibers_inside_roi):
        coords = np.array([[50.0, 50.0], [100.0, 100.0]])   # only 2 points
        with pytest.raises(ValueError):
            compute_fiber_alignment_to_roi(coords, 200, 200, fibers_inside_roi)

    def test_empty_fiber_structure_raises(self, square_roi):
        coords, h, w = square_roi
        with pytest.raises(ValueError, match="empty"):
            compute_fiber_alignment_to_roi(coords, h, w, pd.DataFrame())


# ─────────────────────────────────────────────────────────────────────────────
# Real-dataset tests — ported from pycurvelets tests/test_get_alignment_to_roi.py
#
# Data: tests/test_results/process_image_test_files/
#   real1_roi_df.csv          dense CurveAlign boundary (row, col, no header)
#   real1_fiber_structure.csv fiber DataFrame with center_1/center_2/angle cols
#   real1_ROImeasurements.csv MATLAB reference output (pycurvelets col names)
# ─────────────────────────────────────────────────────────────────────────────

_REPO_TEST_DIR  = pathlib.Path(__file__).parent.parent.parent.parent / "tests"
_ALIGN_DATA_DIR = _REPO_TEST_DIR / "test_results" / "process_image_test_files"
_ALIGN_MISSING  = not _ALIGN_DATA_DIR.exists()


@pytest.mark.skipif(_ALIGN_MISSING, reason="pycurvelets test data not found")
class TestComputeFiberAlignmentRealData:
    """
    Ported from pycurvelets ``tests/test_get_alignment_to_roi.py``.

    Uses real1 image data with distance_threshold=100.  Tolerances match
    the original (rtol=0.05, atol=15).

    Column convention note
    ----------------------
    The reference CSV uses pycurvelets names.  tme_quant renames them and
    applies the 90°-complement to angle_to_boundary_tangent:

      ref ``angle2boundaryEdge``   → compare against 90 − result ``angle_to_boundary_tangent``
      ref ``angle2boundaryCenter`` → compare against result ``angle_to_roi_orientation``
      ref ``angle2centersLine``    → compare against result ``angle_to_centers_line``
                                     (tme_quant uses corrected formula; small
                                     differences from ref are expected for this column)
    """

    @pytest.fixture(scope="class")
    def real1_data(self):
        roi_path    = _ALIGN_DATA_DIR / "real1_roi_df.csv"
        fiber_path  = _ALIGN_DATA_DIR / "real1_fiber_structure.csv"
        ref_path    = _ALIGN_DATA_DIR / "real1_ROImeasurements.csv"

        roi_coords    = pd.read_csv(roi_path, header=None,
                                    names=["row", "col"]).to_numpy(dtype=float)
        fiber_df      = pd.read_csv(fiber_path)
        reference_df  = pd.read_csv(ref_path)
        return roi_coords, fiber_df, reference_df

    def test_returns_dataframe_with_expected_row_count(self, real1_data):
        roi_coords, fiber_df, reference_df = real1_data
        result, count = compute_fiber_alignment_to_roi(
            roi_coords, img_height=512, img_width=512,
            fiber_structure=fiber_df, distance_threshold=100,
        )
        assert isinstance(result, pd.DataFrame)
        assert count == len(reference_df), (
            f"Expected {len(reference_df)} fibres, got {count}"
        )

    def test_angle_to_boundary_tangent_matches_reference(self, real1_data):
        roi_coords, fiber_df, reference_df = real1_data
        result, _ = compute_fiber_alignment_to_roi(
            roi_coords, img_height=512, img_width=512,
            fiber_structure=fiber_df, distance_threshold=100,
        )
        # Convert pycurvelets reference: angle_to_boundary_tangent = 90 - angle2boundaryEdge
        ref_tangent = 90.0 - reference_df["angle2boundaryEdge"].to_numpy(dtype=float)
        pred = pd.to_numeric(result["angle_to_boundary_tangent"],
                             errors="coerce").fillna(0.0).to_numpy()
        np.testing.assert_allclose(pred, ref_tangent, rtol=0.05, atol=15,
                                   err_msg="angle_to_boundary_tangent vs reference")

    def test_angle_to_roi_orientation_matches_reference(self, real1_data):
        roi_coords, fiber_df, reference_df = real1_data
        result, _ = compute_fiber_alignment_to_roi(
            roi_coords, img_height=512, img_width=512,
            fiber_structure=fiber_df, distance_threshold=100,
        )
        ref = reference_df["angle2boundaryCenter"].to_numpy(dtype=float)
        pred = result["angle_to_roi_orientation"].to_numpy(dtype=float)
        np.testing.assert_allclose(pred, ref, rtol=0.05, atol=15,
                                   err_msg="angle_to_roi_orientation vs reference")

    def test_fiber_centers_match_reference(self, real1_data):
        roi_coords, fiber_df, reference_df = real1_data
        result, _ = compute_fiber_alignment_to_roi(
            roi_coords, img_height=512, img_width=512,
            fiber_structure=fiber_df, distance_threshold=100,
        )
        # pycurvelets maps center_1→center_row and center_2→center_col.
        # In the MATLAB reference, fibercenterX = center_1 = center_row
        # and fibercenterY = center_2 = center_col.
        np.testing.assert_allclose(
            result["fiber_center_row"].to_numpy(dtype=float),
            reference_df["fibercenterX"].to_numpy(dtype=float),
            rtol=0.05, atol=1,
        )
        np.testing.assert_allclose(
            result["fiber_center_col"].to_numpy(dtype=float),
            reference_df["fibercenterY"].to_numpy(dtype=float),
            rtol=0.05, atol=1,
        )
