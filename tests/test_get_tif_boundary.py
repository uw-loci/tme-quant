import os
import csv
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycurvelets.get_tif_boundary import get_tif_boundary


@pytest.fixture
def test_data():
    """Loads all test inputs from disk."""
    base_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "get_tif_boundary_test_files",
    )

    # --- Load boundary coordinates ---
    coord_files = [
        os.path.join(base_path, f)
        for f in [
            "real1_boundary1_coords.csv",
            "real1_boundary2_coords.csv",
            "real1_boundary3_coords.csv",
        ]
    ]
    coords = {}
    for i, f in enumerate(coord_files, start=1):
        with open(f, newline="") as csvfile:
            reader = csv.reader(csvfile)
            coords[f"csv{i}"] = [
                tuple(np.array(list(map(float, row))) - 1) for row in reader
            ]

    # --- Load image ---
    img_path = os.path.join(os.path.dirname(__file__), "test_images", "real1.tif")
    img = plt.imread(img_path, format="TIF")

    # --- Load curvelet object data ---
    obj = {}
    curvelet_file = os.path.join(base_path, "real1_curvelets.csv")
    with open(curvelet_file, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            obj[i] = {
                "center": (float(row["center_1"]) - 1, float(row["center_2"]) - 1),
                "angle": float(row["angle"]),
                "weight": float(row["weight"]),
            }

    return coords, img, obj


def test_get_tif_boundary_output_structure(test_data):
    """Tests if get_tif_boundary produces outputs with correct shapes and types."""
    coords, img, obj = test_data

    dist_thresh = 100
    min_dist = []

    result_mat, result_mat_names, num_img_points, result_df = get_tif_boundary(
        coords, img, obj, dist_thresh, min_dist
    )

    # --- Type checks ---
    assert isinstance(result_mat, np.ndarray)
    assert isinstance(result_mat_names, list)
    assert isinstance(num_img_points, (int, float))
    assert isinstance(result_df, pd.DataFrame)

    # --- Shape consistency ---
    n_curvs = len(obj)
    assert result_mat.shape[0] == n_curvs
    assert result_mat.shape[1] == len(result_mat_names)
    assert list(result_df.columns) == result_mat_names
    assert result_df.shape == result_mat.shape

    # --- Value sanity checks ---
    # Allow NaN, but check numeric columns exist
    numeric_cols = [
        "nearest_boundary_distance",
        "nearest_region_distance",
        "nearest_boundary_angle",
    ]
    for col in numeric_cols:
        assert col in result_df.columns
        assert pd.api.types.is_numeric_dtype(result_df[col])

    # --- Basic numerical range sanity ---
    # (Example: distances should not be negative)
    assert np.all(result_df["nearest_boundary_distance"].dropna() >= 0)

    print("\nSample result_df:\n", result_df.head())


def test_get_tif_boundary_returns_meaningful_points(test_data):
    """Check if function returns non-empty, plausible data."""
    coords, img, obj = test_data
    dist_thresh = 100
    min_dist = []

    _, _, num_img_points, df = get_tif_boundary(coords, img, obj, dist_thresh, min_dist)

    # Should detect at least some image points
    assert num_img_points > 0
    # Should have at least one valid non-NaN row
    assert df.notna().any().any()


def test_get_tif_boundary_vs_expected_file(test_data):
    coords, img, obj = test_data
    dist_thresh = 100
    min_dist = []

    # Run your function
    _, _, _, df = get_tif_boundary(coords, img, obj, dist_thresh, min_dist)
    df[["boundary_point_row", "boundary_point_col"]] = df[
        ["boundary_point_col", "boundary_point_row"]
    ]

    # Load expected output CSV
    base_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "get_tif_boundary_test_files",
    )
    expected_file = os.path.join(base_path, "real1_get_tif_boundary_output.csv")
    expected_df = pd.read_csv(expected_file)
    expected_df[["bndryPtRow", "bndryPtCol"]] -= 1

    # --- Compare only the values (ignore column names) ---
    assert df.shape == expected_df.shape, "DataFrames shape mismatch"

    # Convert both to NumPy arrays for comparison
    actual_values = df.to_numpy()
    expected_values = expected_df.to_numpy()

    # Use np.isclose with NaN equality for numerical comparison
    comparison = np.isclose(
        actual_values, expected_values, rtol=1e-6, atol=1e-8, equal_nan=True
    )

    # comparison: boolean array from np.isclose
    mismatch_idx = np.argwhere(~comparison)

    for row, col_idx in mismatch_idx:
        col_name = df.columns[col_idx]  # or df.columns[col_idx]
        print(
            f"Mismatch at row {row}, column '{col_name}': "
            f"actual={actual_values[row, col_idx]}, expected={expected_values[row, col_idx]}"
        )

    # Optional assertion
    assert comparison.all(), f"{len(mismatch_idx)} mismatches found."
