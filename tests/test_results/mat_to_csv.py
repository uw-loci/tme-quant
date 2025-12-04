import argparse
from scipy.io import loadmat
import pandas as pd
import numpy as np


def extract_field(field):
    """
    Safely extract a MATLAB struct field that may be nested ndarray.
    Returns a flat numpy array for 'center' or a float for 'angle'.
    """
    # Squeeze to remove singleton dimensions
    if isinstance(field, np.ndarray):
        field = field.squeeze()
    # If it's still an ndarray with size 1, extract the scalar
    if isinstance(field, np.ndarray) and field.size == 1:
        field = field.item()
    return field


def mat_to_csv(mat_file_path, csv_file_path):
    mat_data = loadmat(mat_file_path)

    if "inCurvs" not in mat_data:
        raise KeyError("Expected 'inCurvs' in .mat file")

    in_curvs_array = mat_data["inCurvs"]

    # Flatten to 1D array
    in_curvs_flat = in_curvs_array.ravel()

    centers_list = []
    angles_list = []

    for entry in in_curvs_flat:
        # Entry is a numpy.void with 'center' and 'angle' fields
        center = extract_field(entry["center"])
        angle = extract_field(entry["angle"])

        # Ensure center is a 1D array with 2 elements
        if isinstance(center, np.ndarray):
            center = center.flatten()
        else:
            center = np.array([center])

        # Ensure angle is a scalar
        if isinstance(angle, np.ndarray):
            angle = angle.item()

        centers_list.append(center)
        angles_list.append(angle)

    centers = np.array(centers_list)
    angles = np.array(angles_list)

    df = pd.DataFrame(
        {"center_0": centers[:, 0], "center_1": centers[:, 1], "angle": angles}
    )

    df.to_csv(csv_file_path, index=False)
    print(f"Saved CSV to {csv_file_path} with {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat file to CSV")
    parser.add_argument("mat_file", help="Path to input .mat file")
    parser.add_argument("csv_file", help="Path to output CSV file")
    args = parser.parse_args()

    mat_to_csv(args.mat_file, args.csv_file)
