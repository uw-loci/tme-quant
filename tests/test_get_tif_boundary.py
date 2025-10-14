import os
import csv
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycurvelets.get_tif_boundary import get_tif_boundary


@pytest.fixture
def test_data():
    """
    Load all test inputs (image, boundary coordinates, and curvelet data) from disk.
    Converts MATLAB 1-based coordinates to Python 0-based indexing.
    """
    base_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "get_tif_boundary_test_files",
    )

    # load boundary coordinates
    coord_files = [
        os.path.join(base_path, fname)
        for fname in [
            "real1_boundary1_coords.csv",
            "real1_boundary2_coords.csv",
            "real1_boundary3_coords.csv",
        ]
    ]

    coords = {}
    for i, path in enumerate(coord_files, start=1):
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            coords[f"csv{i}"] = [
                tuple(np.array(list(map(float, row))) - 1) for row in reader
            ]

    # load test image
    img_path = os.path.join(os.path.dirname(__file__), "test_images", "real1.tif")
    img = plt.imread(img_path, format="TIF")

    # load curvelet metadata
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
    """Verify output types, shapes, and numeric consistency."""
    coords, img, obj = test_data
    dist_thresh = 100

    result_mat, result_mat_names, num_img_points, result_df = get_tif_boundary(
        coords, img, obj, dist_thresh, min_dist=[]
    )

    # check type
    assert isinstance(result_mat, np.ndarray), "result_mat must be a NumPy array"
    assert isinstance(result_mat_names, list), "result_mat_names must be a list"
    assert isinstance(num_img_points, (int, float)), "num_img_points must be numeric"
    assert isinstance(result_df, pd.DataFrame), "result_df must be a pandas DataFrame"

    # shape consistency
    n_curvs = len(obj)
    assert result_mat.shape == result_df.shape, "Matrix and DataFrame shape mismatch"
    assert result_mat.shape[0] == n_curvs, "Row count must match number of curvelets"
    assert result_mat.shape[1] == len(result_mat_names), "Column count mismatch"
    assert list(result_df.columns) == result_mat_names, "Column names inconsistent"

    # column sanity check
    numeric_cols = [
        "nearest_boundary_distance",
        "nearest_region_distance",
        "nearest_boundary_angle",
    ]
    for col in numeric_cols:
        assert col in result_df.columns, f"Missing expected column: {col}"
        assert pd.api.types.is_numeric_dtype(
            result_df[col]
        ), f"Column {col} not numeric"

    # numerical sanity check
    distances = result_df["nearest_boundary_distance"].dropna()
    assert (distances >= 0).all(), "Boundary distances should be non-negative"

    print("\nSample result_df:\n", result_df.head())


def test_get_tif_boundary_returns_meaningful_points(test_data):
    """Ensure the function detects valid image points and returns usable data."""
    coords, img, obj = test_data
    dist_thresh = 100

    _, _, num_img_points, df = get_tif_boundary(
        coords, img, obj, dist_thresh, min_dist=[]
    )

    assert num_img_points > 0, "No image points detected"
    assert df.notna().any().any(), "DataFrame appears empty or entirely NaN"


def test_get_tif_boundary_matches_expected_file(test_data):
    """Compare computed output against the expected reference CSV file."""
    coords, img, obj = test_data
    dist_thresh = 100

    # get output
    _, _, _, df = get_tif_boundary(coords, img, obj, dist_thresh, min_dist=[])

    # swap row/col to match MATLAB convention, round, convert to float for NaN-safe comparison
    df[["boundary_point_row", "boundary_point_col"]] = df[
        ["boundary_point_col", "boundary_point_row"]
    ].round()

    # load reference
    base_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "get_tif_boundary_test_files",
    )
    expected_file = os.path.join(base_path, "real1_get_tif_boundary_output.csv")
    expected_df = pd.read_csv(expected_file)

    # convert MATLAB 1-based to Python 0-based, round
    expected_df[["bndryPtRow", "bndryPtCol"]] = (
        expected_df[["bndryPtRow", "bndryPtCol"]] - 1
    ).round()

    # check shape consistency
    assert (
        df.shape == expected_df.shape
    ), f"Shape mismatch: expected {expected_df.shape}, got {df.shape}"

    # convert all to float to handle NaN and pd.NA safely
    actual = df.to_numpy(dtype=float)
    expected = expected_df.to_numpy(dtype=float)

    # numerical comparison with tolerance
    comparison = np.isclose(actual, expected, rtol=1e-2, atol=1e-2, equal_nan=True)
    mismatch_idx = np.argwhere(~comparison)

    if len(mismatch_idx) > 0:
        print("\n⚠️ Mismatched values detected:")
        for row, col_idx in mismatch_idx:
            col_name = df.columns[col_idx]
            print(
                f"  Row {row}, column '{col_name}': "
                f"actual={actual[row, col_idx]:.6f}, expected={expected[row, col_idx]:.6f}"
            )

    assert comparison.all(), f"{len(mismatch_idx)} mismatched entries found."
