import os
import pytest
import pandas as pd
from pycurvelets.relative_angle.get_relative_angles import (
    get_relative_angles,
    load_coords,
)
import numpy as np

# Load boundary coordinates
csv_path = os.path.join(
    os.path.dirname(__file__),
    "test_results",
    "relative_angle_test_files",
    "boundary_coords.csv",
)
coords = load_coords(csv_path)

# Load Excel file with fiber info and expected angles
xl_path = os.path.join(
    os.path.dirname(__file__),
    "test_results",
    "relative_angle_test_files",
    "real1_BoundaryMeasurements.xlsx",
)
df = pd.read_excel(xl_path, sheet_name=1)

# Build test cases dynamically
test_cases = []
for _, row in df.iterrows():
    # Find the index of the boundary point in coords
    boundary_coord = [row["boundaryPointRow"], row["boundaryPointCol"]]
    matches = np.where((coords == boundary_coord).all(axis=1))[0]
    if len(matches) == 0:
        # If not found, skip this fiber
        pytest.skip(f"No boundary match found for fiber at {boundary_coord}")
    index2object = matches[0]

    test_cases.append(
        (
            row["fibercenterRow"],
            row["fibercenterCol"],
            row["fiberangleList"],
            index2object,
            {
                "angle2boundaryEdge": row["angle2boundaryEdge"],
                "angle2boundaryCenter": row["angle2boundaryCenter"],
                "angle2centersLine": row["angle2centersLine"],
            },
        )
    )


@pytest.mark.parametrize(
    "fiber_row,fiber_col,fiber_angle,index2object,expected_angles", test_cases
)
def test_get_relative_angles(
    fiber_row, fiber_col, fiber_angle, index2object, expected_angles
):
    """
    Tests the functions of get_relative_angles given the values from boundary_coords.csv
    and real1_BoundaryMeasurements.xlsx. The index2object value was identified by finding
    the row the boundary center coordinate found in boundary_coords.csv.

    Error margin is 0.5
    """
    ROI = {
        "coords": coords,
        "imageWidth": 512,
        "imageHeight": 512,
        "index2object": index2object,
    }

    object_data = {"center": [fiber_row, fiber_col], "angle": fiber_angle}

    angles, _ = get_relative_angles(ROI, object_data, angle_option=0, fig_flag=False)

    for key, expected in expected_angles.items():
        assert (
            abs(angles[key] - expected) < 0.5
        ), f"{key} mismatch: got {angles[key]:.4f}, expected {expected:.4f}"
