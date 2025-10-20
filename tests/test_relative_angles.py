import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image

# Import the new Python API
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import curvealign_py as ca
from curvealign_py import Boundary, CurveAlignOptions, Curvelet

"""
Refactored test suite for relative angle calculations using the new Python API.

These tests verify that boundary analysis and relative angle calculations produce
results that are compatible with the original MATLAB CurveAlign implementation by
testing exact angle values with appropriate tolerances.
"""


def load_coords_csv(csv_path: str) -> np.ndarray:
    """
    Load coordinate data from CSV file.
    
    Expected format: CSV with Y,X columns (MATLAB convention)
    Returns: numpy array with shape (N, 2) as [Y, X] coordinates
    """
    # Use regex to match any whitespace (spaces or tabs)
    # This handles mixed whitespace in headers and data
    df = pd.read_csv(csv_path, sep=r'\s+', engine='python')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Handle different possible column names
    if 'Y' in df.columns and 'X' in df.columns:
        coords = df[['Y', 'X']].values
    elif len(df.columns) >= 2:
        # Assume first two columns are Y, X
        coords = df.iloc[:, :2].values
    else:
        raise ValueError(f"Cannot parse coordinate data from {csv_path}: found columns {df.columns.tolist()}")
    
    return coords.astype(float)


def create_boundary_from_coords(coords: np.ndarray) -> Boundary:
    """Create a Boundary object from coordinate array."""
    return Boundary(
        kind="polygon",
        data=coords,
        spacing_xy=(1.0, 1.0)  # Assume pixel spacing
    )


def create_test_curvelet(center_row: int, center_col: int, angle_deg: float) -> Curvelet:
    """Create a test curvelet with specified properties."""
    return Curvelet(
        center_row=center_row,
        center_col=center_col,
        angle_deg=angle_deg,
        weight=1.0
    )


def test_load_coords_1():
    """
    Test loading coordinates from sample_coords.csv.
    
    This tests the coordinate loading functionality with a simple CSV file.
    """
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "relative_angle_test_files",
        "sample_coords.csv",
    )
    
    result = load_coords_csv(csv_path)
    expected = np.array([[143.0, 205.0], [143.0, 206.0], [142.0, 206.0]])
    
    # Check that coordinates are loaded correctly
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    assert result.shape[1] == 2, "Should have 2 columns (Y, X)"


def test_load_coords_2():
    """
    Test loading coordinates from boundary_coords.csv.
    
    This tests coordinate loading with a more complex CSV file.
    Only checks the first 25 rows for consistency with original test.
    """
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "relative_angle_test_files",
        "boundary_coords.csv",
    )
    
    result = load_coords_csv(csv_path)
    expected = np.array([
        [143, 205], [143, 206], [142, 206], [141, 206], [140, 206],
        [139, 206], [138, 206], [137, 206], [136, 206], [135, 206],
        [134, 206], [134, 207], [133, 207], [132, 207], [131, 207],
        [130, 207], [130, 208], [129, 208], [128, 208], [127, 208],
        [127, 209], [126, 209], [125, 209], [124, 209], [124, 210],
    ])
    
    # Check first 25 rows match expected values
    assert np.allclose(result[:25], expected), "First 25 coordinates should match expected values"
    assert result.shape[0] >= 25, "Should have at least 25 coordinate pairs"


@pytest.mark.parametrize("center_row,center_col,angle_deg,expected_angle,test_description", [
    (145, 430, 14.0625, 89.6518396, "Fiber at [145, 430] - nearly perpendicular (angle2boundaryEdge: 89.65°, angle2boundaryCenter: 87.54°, angle2centersLine: 26.44°)"),
    (94, 473, 75.9375, 78.26128691, "Fiber at [94, 473] - (angle2boundaryEdge: 78.26°, angle2boundaryCenter: 25.66°, angle2centersLine: 32.82°)"),
    (167, 414, 2.8125, 81.65541942, "Fiber at [167, 414] - angle2boundaryEdge=81.66°"),
    (72, 249, 105.6696429, 65.29739494, "Fiber at [72, 249] - angle2boundaryEdge=65.30°"),
    (394, 221, 95.15625, 35.48753723, "Fiber at [394, 221] - angle2boundaryEdge=35.49°"),
    (420, 197, 92.8125, 37.83128723, "Fiber at [420, 197] - angle2boundaryEdge=37.83°"),
])
def test_relative_angles_parametrized(center_row, center_col, angle_deg, expected_angle, test_description):
    """
    Test relative angle calculation for various fibers against MATLAB reference values.
    
    This parametrized test verifies that the new Python API produces angle calculations
    compatible with the original MATLAB implementation. The expected_angle corresponds
    to the MATLAB angle2boundaryEdge value.
    """
    # Load boundary coordinates
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "relative_angle_test_files",
        "boundary_coords.csv",
    )
    coords = load_coords_csv(csv_path)
    boundary = create_boundary_from_coords(coords)
    
    # Create test curvelet with exact MATLAB parameters
    test_curvelet = create_test_curvelet(
        center_row=center_row,
        center_col=center_col,
        angle_deg=angle_deg
    )
    
    # Perform boundary analysis with parameters matching MATLAB
    metrics = ca.core.processors.measure_boundary_alignment(
        curvelets=[test_curvelet],
        boundary=boundary,
        dist_thresh=1000,  # Large threshold to ensure inclusion
    )
    
    # Verify we got results
    assert len(metrics.relative_angles) > 0, f"{test_description}: Should get relative angle measurements"
    
    # Get the computed relative angle
    relative_angle = metrics.relative_angles[0]
    
    # Allow for algorithm differences but should be in reasonable range (0.5° tolerance)
    assert abs(relative_angle - expected_angle) < 0.5, \
        f"{test_description}: Relative angle should be near MATLAB reference {expected_angle}°, got {relative_angle}°"
    
    # Ensure it's in valid range
    assert 0 <= relative_angle <= 90, f"{test_description}: Relative angle should be in [0,90], got {relative_angle}"


def test_matlab_reference_angles_comprehensive():
    """
    Test multiple fibers against MATLAB reference values simultaneously.
    
    This validates that the new Python API produces relative angles that are
    reasonably close to the original MATLAB angle2boundaryEdge values.
    """
    # Load boundary coordinates
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "relative_angle_test_files",
        "boundary_coords.csv",
    )
    coords = load_coords_csv(csv_path)
    boundary = create_boundary_from_coords(coords)
    
    # Test cases with MATLAB reference values (angle2boundaryEdge)
    test_cases = [
        {"position": (145, 430), "angle": 14.0625, "matlab_ref": 89.6518396},
        {"position": (94, 473), "angle": 75.9375, "matlab_ref": 78.26128691},
        {"position": (167, 414), "angle": 2.8125, "matlab_ref": 81.65541942},
        {"position": (72, 249), "angle": 105.6696429, "matlab_ref": 65.29739494},
        {"position": (394, 221), "angle": 95.15625, "matlab_ref": 35.48753723},
        {"position": (420, 197), "angle": 92.8125, "matlab_ref": 37.83128723},
    ]
    
    # Test each case
    tolerance = 0.5  # Allow 0.5° tolerance for algorithm differences
    results = []
    
    for i, case in enumerate(test_cases):
        test_curvelet = create_test_curvelet(
            case["position"][0], 
            case["position"][1], 
            case["angle"]
        )
        
        metrics = ca.core.processors.measure_boundary_alignment(
            curvelets=[test_curvelet],
            boundary=boundary,
            dist_thresh=1000,
        )
        
        assert len(metrics.relative_angles) > 0, f"Case {i+1}: Should get angle measurement"
        
        relative_angle = metrics.relative_angles[0]
        matlab_ref = case["matlab_ref"]
        
        # Check that it's reasonably close to MATLAB reference
        diff = abs(relative_angle - matlab_ref)
        assert diff < tolerance, \
            f"Case {i+1}: Angle diff too large. Python={relative_angle:.2f}°, MATLAB={matlab_ref:.2f}°, diff={diff:.2f}°"
        
        # Ensure valid range
        assert 0 <= relative_angle <= 90, f"Case {i+1}: Invalid angle range {relative_angle}°"
        
        results.append({
            "position": case["position"],
            "python_angle": relative_angle,
            "matlab_ref": matlab_ref,
            "difference": diff
        })
    
    # Print summary for debugging (will be visible if test fails)
    avg_diff = sum(r["difference"] for r in results) / len(results)
    max_diff = max(r["difference"] for r in results)
    
    print(f"\nMATLAB Compatibility Summary:")
    print(f"Average difference: {avg_diff:.2f}°")
    print(f"Maximum difference: {max_diff:.2f}°")
    print(f"All {len(results)} test cases passed within {tolerance}° tolerance")


def test_boundary_analysis_integration():
    """Test basic boundary analysis workflow functionality."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_results", "relative_angle_test_files", "boundary_coords.csv")
    coords = load_coords_csv(csv_path)
    boundary = create_boundary_from_coords(coords)
    
    test_curvelets = [
        create_test_curvelet(145, 430, 14.0625),
        create_test_curvelet(94, 473, 75.9375),
        create_test_curvelet(167, 414, 2.8125),
    ]
    
    metrics = ca.core.processors.measure_boundary_alignment(test_curvelets, boundary, dist_thresh=1000)
    
    # Basic validation
    assert len(metrics.relative_angles) <= len(test_curvelets)
    assert len(metrics.distances) == len(metrics.relative_angles)
    assert all(0 <= angle <= 90 for angle in metrics.relative_angles)
    assert all(dist >= 0 for dist in metrics.distances)


def test_distance_thresholding():
    """Test that distance thresholding works correctly."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_results", "relative_angle_test_files", "boundary_coords.csv")
    coords = load_coords_csv(csv_path)
    boundary = create_boundary_from_coords(coords)
    
    far_curvelet = create_test_curvelet(0, 0, 45.0)  # Top-left corner
    
    metrics_small = ca.core.processors.measure_boundary_alignment([far_curvelet], boundary, dist_thresh=10)
    metrics_large = ca.core.processors.measure_boundary_alignment([far_curvelet], boundary, dist_thresh=1000)
    
    # Large threshold should include more curvelets
    assert len(metrics_large.relative_angles) >= len(metrics_small.relative_angles)


def test_empty_inputs():
    """Test handling of empty inputs."""
    csv_path = os.path.join(os.path.dirname(__file__), "test_results", "relative_angle_test_files", "boundary_coords.csv")
    coords = load_coords_csv(csv_path)
    boundary = create_boundary_from_coords(coords)
    
    metrics = ca.core.processors.measure_boundary_alignment([], boundary, dist_thresh=100)
    
    # Should handle empty input gracefully
    assert len(metrics.relative_angles) == 0
    assert len(metrics.distances) == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__])