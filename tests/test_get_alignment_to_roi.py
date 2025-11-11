"""
Test module for get_alignment_to_roi function.

This module tests the conversion of MATLAB's getAlignment2ROI.m to Python,
verifying that fiber alignment measurements relative to ROIs are calculated correctly.
"""

import os
import numpy as np
import pandas as pd
import pytest

from pycurvelets.get_alignment_to_roi import get_alignment_to_roi
from pycurvelets.models.models import ROIList, Fiber
from pycurvelets.utils.math import flatten_numeric


def load_test_data():
    """Load test data files for ROI alignment testing."""
    base_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "process_image_test_files",
    )

    # Load ROI coordinates
    roi_coords_path = os.path.join(base_path, "real1_roi_df.csv")
    roi_coords_df = pd.read_csv(roi_coords_path, header=None, names=["row", "col"])
    roi_coords = list(zip(roi_coords_df["row"].values, roi_coords_df["col"].values))

    # Load fiber structure
    fiber_structure_path = os.path.join(base_path, "real1_fiber_structure.csv")
    fiber_df = pd.read_csv(fiber_structure_path)

    # Convert to list of Fiber objects
    fiber_list = []
    for _, row in fiber_df.iterrows():
        fiber_list.append(
            Fiber(
                center_row=row["center_1"],
                center_col=row["center_2"],
                angle=row["angle"],
            )
        )

    # Load expected output
    expected_output_path = os.path.join(base_path, "real1_ROImeasurements.csv")
    expected_df = pd.read_csv(expected_output_path)

    return roi_coords, fiber_list, expected_df


def test_get_alignment_to_roi_with_distance_threshold():
    """Test get_alignment_to_roi with distance threshold mode."""
    roi_coords, fiber_list, expected_df = load_test_data()

    # Create ROIList with single ROI
    roi_list = ROIList(
        coordinates=[roi_coords],
        image_width=512,
        image_height=512,
    )

    distance_threshold = 100

    # Call the function
    result, result_fiber_count = get_alignment_to_roi(
        roi_list, [fiber_list], distance_threshold
    )

    # Verify result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"

    # Check that we have the expected columns
    expected_columns = [
        "angle_to_boundary_edge",
        "angle_to_boundary_center",
        "angle_to_center_line",
        "fiber_center_x",
        "fiber_center_y",
        "fiber_angle",
        "distance",
    ]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"

    # Check number of rows matches expected
    assert len(result) == len(
        expected_df
    ), f"Expected {len(expected_df)} rows, got {len(result)}"

    # Compare values with tolerance for floating point
    # Use a reasonable tolerance for numerical differences
    rtol = 1e-3  # 0.1% relative tolerance
    atol = 1e-3  # 0.1% absolute tolerance for angles

    for col in expected_columns:
        if col in result.columns and col in expected_df.columns:
            # Use numpy allclose for floating point comparison
            assert np.allclose(
                flatten_numeric(result[col]),
                flatten_numeric(expected_df[col]),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ), f"Values don't match for column: {col}"

    assert result_fiber_count == expected_df.shape[0]


def test_get_alignment_to_roi_empty_fiber_structure():
    """Test that function raises error with empty fiber structure."""
    roi_coords = [(10, 10), (20, 10), (20, 20), (10, 20)]
    roi_list = ROIList(
        coordinates=[roi_coords],
        image_width=100,
        image_height=100,
    )

    with pytest.raises(ValueError, match="fiber_structure cannot be None or empty"):
        get_alignment_to_roi(roi_list, [], 10)


def test_get_alignment_to_roi_none_roi():
    """Test that function raises error with None ROI."""
    fiber_list = [Fiber(center_row=15, center_col=15, angle=45)]

    with pytest.raises(ValueError, match="roi_list cannot be None"):
        get_alignment_to_roi(None, [fiber_list], 10)


if __name__ == "__main__":
    # Run tests
    test_get_alignment_to_roi_with_distance_threshold()
    print("✓ test_get_alignment_to_roi_with_distance_threshold passed")

    test_get_alignment_to_roi_empty_fiber_structure()
    print("✓ test_get_alignment_to_roi_empty_fiber_structure passed")

    test_get_alignment_to_roi_none_roi()
    print("✓ test_get_alignment_to_roi_none_roi passed")

    print("\nAll tests passed!")
