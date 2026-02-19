import argparse
from scipy.io import loadmat
import numpy as np


def mat_to_csv(mat_file_path, csv_file_path):
    """
    Convert a MATLAB .mat file to CSV format.
    Handles various data structures and saves without column names.
    """
    mat_data = loadmat(mat_file_path)

    # Filter out MATLAB metadata keys (start with __)
    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]

    if not data_keys:
        raise ValueError("No data found in .mat file")

    # Get the first (and usually only) data variable
    key = data_keys[0]
    data = mat_data[key]

    # Handle different data structures
    if isinstance(data, np.ndarray):
        # Check if it's a structured array (MATLAB struct)
        if data.dtype.names:
            # Structured array - extract fields
            rows = []
            for entry in data.ravel():
                row = []
                for field_name in data.dtype.names:
                    field_value = entry[field_name]
                    # Squeeze and flatten
                    if isinstance(field_value, np.ndarray):
                        field_value = field_value.flatten()
                        if field_value.size == 1:
                            field_value = field_value.item()
                    row.extend(np.atleast_1d(field_value))
                rows.append(row)
            data_array = np.array(rows)
        else:
            # Regular array
            data_array = (
                data.reshape(-1, data.shape[-1])
                if data.ndim > 1
                else data.reshape(-1, 1)
            )
    else:
        # Scalar or other type
        data_array = np.atleast_2d(data)

    # Save to CSV without header
    np.savetxt(csv_file_path, data_array, delimiter=",", fmt="%g")
    print(f"Saved CSV to {csv_file_path} with shape {data_array.shape}")


def mat_to_new_curv_csv(mat_file_path, csv_file_path, debug=False):
    """
    Convert a MATLAB .mat file from new_curv output to CSV format.
    Extracts center (as center_0, center_1) and angle columns and saves as CSV.

    Args:
        mat_file_path: Path to input .mat file
        csv_file_path: Path to output CSV file
        debug: If True, print field names and first entry
    """
    mat_data = loadmat(mat_file_path)

    # Filter out MATLAB metadata keys (start with __)
    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]

    if not data_keys:
        raise ValueError("No data found in .mat file")

    # Get the first (and usually only) data variable
    key = data_keys[0]
    data = mat_data[key]

    if debug:
        print(f"Variable name: {key}")
        print(f"Data type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"Array shape: {data.shape}")
            print(f"Array dtype: {data.dtype}")
            if data.dtype.names:
                print(f"Field names: {data.dtype.names}")
                if len(data) > 0:
                    print(f"First entry: {data[0]}")

    # Handle structured array (MATLAB struct)
    if isinstance(data, np.ndarray) and data.dtype.names:
        rows = []
        field_names = data.dtype.names

        # Look for 'center' and 'angle' fields
        center_field = "center" if "center" in field_names else None
        angle_field = "angle" if "angle" in field_names else None

        if not (center_field and angle_field):
            raise ValueError(
                f"Could not find 'center' and 'angle' fields. Available: {field_names}"
            )

        for entry in data.ravel():
            center = entry[center_field].flatten()
            angle = entry[angle_field].flatten()[0]

            # center should be [row, col] or [center_0, center_1]
            if len(center) >= 2:
                center_0 = center[0]
                center_1 = center[1]
            else:
                raise ValueError(
                    f"Expected center to have at least 2 elements, got {center}"
                )

            rows.append([center_0, center_1, angle])

        if not rows:
            raise ValueError("Could not extract data from .mat file")

        data_array = np.array(rows)
    else:
        raise ValueError("Expected structured array")

    # Save to CSV with header
    with open(csv_file_path, "w") as f:
        f.write("center_0,center_1,angle\n")
        np.savetxt(f, data_array, delimiter=",", fmt="%g")

    print(
        f"Saved new_curv CSV to {csv_file_path} with shape {data_array.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat file to CSV")
    parser.add_argument("mat_file", help="Path to input .mat file")
    parser.add_argument("csv_file", help="Path to output CSV file")
    parser.add_argument(
        "--new-curv",
        action="store_true",
        help="Use new_curv format (center as [row, col], angle)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug information"
    )
    args = parser.parse_args()

    if args.new_curv:
        mat_to_new_curv_csv(args.mat_file, args.csv_file, debug=args.debug)
    else:
        mat_to_csv(args.mat_file, args.csv_file)
