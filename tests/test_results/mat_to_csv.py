import argparse
from scipy.io import loadmat
import numpy as np


def mat_to_csv(mat_file_path, csv_file_path, swap_last_cols=False):
    """
    Convert a MATLAB .mat file to CSV format.
    Handles various data structures and saves without column names.
    
    Args:
        mat_file_path: Path to input .mat file
        csv_file_path: Path to output CSV file
        swap_last_cols: If True, swap the last two columns (row/col coordinates)
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

    # Swap last two columns if requested (for row/col coordinates)
    if swap_last_cols and data_array.shape[1] >= 2:
        data_array[:, [-2, -1]] = data_array[:, [-1, -2]]
        print(f"Swapped last two columns")

    # Save to CSV without header
    np.savetxt(csv_file_path, data_array, delimiter=",", fmt="%g")
    print(f"Saved CSV to {csv_file_path} with shape {data_array.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATLAB .mat file to CSV")
    parser.add_argument("mat_file", help="Path to input .mat file")
    parser.add_argument("csv_file", help="Path to output CSV file")
    parser.add_argument("--swap-last-cols", action="store_true", help="Swap the last two columns (row/col coordinates)")
    args = parser.parse_args()

    mat_to_csv(args.mat_file, args.csv_file, swap_last_cols=args.swap_last_cols)
