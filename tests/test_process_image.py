"""
Test module for process_image function.

This module tests the end-to-end image processing pipeline,
verifying that fiber features are computed correctly.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pycurvelets.models import (
    AdvancedAnalysisOptions,
    BoundaryParameters,
    CurveletControlParameters,
    FeatureControlParameters,
    FiberAnalysisParameters,
    ImageInputParameters,
    OutputControlParameters,
)

# By default, skip curvelops-dependent tests (e.g., on CI). Enable locally with:
#   TMEQ_RUN_CURVELETS=1 pytest -q
if os.environ.get("TMEQ_RUN_CURVELETS") != "1":
    pytest.skip(
        "curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
        allow_module_level=True,
    )

# Optional: attempt import; skip module if curvelet backend is missing
try:
    from pycurvelets.process_image import process_image
except ModuleNotFoundError:
    pytest.skip(
        "curvelops not available; skipping process_image tests", allow_module_level=True
    )


# --------------------------
# Helpers
# --------------------------


def load_test_image(image_name):
    """Load a test image by filename."""
    img_path = os.path.join(os.path.dirname(__file__), "test_images", image_name)
    return plt.imread(img_path, format="TIF")


def load_test_cases():
    """
    Load JSON test cases and return (name, case) tuples for parametrize.

    Returns:
        List of (name, test_case) tuples.
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        "test_results",
        "process_image_test_files",
        "test_cases_process_image.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    cases = config["test_cases"]
    return [(tc["name"], tc) for tc in cases]


def load_boundary_data_from_json(boundary_params_dict, img_name):
    """
    Load boundary data from CSV files referenced in JSON.

    Converts "from_file:" markers in the JSON into actual numpy arrays.
    Supports paths like:
      - from_file:shared_data/real1/boundary1_coords.csv (relative to test_results/)
      - from_file:real1_boundary1_coords.csv (legacy, from get_tif_boundary_test_files/)
    If coordinates are not provided, auto-loads them from shared_data/{img_name}/.
    """
    if boundary_params_dict is None:
        return None

    boundary_params_dict = boundary_params_dict.copy()

    # Load coordinates if they reference files, or auto-load from shared_data if not provided
    if "coordinates" in boundary_params_dict and boundary_params_dict["coordinates"]:
        coords_dict = boundary_params_dict["coordinates"]
        loaded_coords = {}

        for key, value in coords_dict.items():
            if isinstance(value, str) and value.startswith("from_file:"):
                filename = value.replace("from_file:", "")
                # Determine base path based on filename format
                if filename.startswith("shared_data/"):
                    # New format: from_file:shared_data/real1/boundary1_coords.csv
                    filepath = os.path.join(
                        os.path.dirname(__file__),
                        "test_results",
                        filename,
                    )
                else:
                    # Legacy format: from_file:real1_boundary1_coords.csv
                    filepath = os.path.join(
                        os.path.dirname(__file__),
                        "test_results",
                        "get_tif_boundary_test_files",
                        filename,
                    )
                loaded_coords[key] = np.loadtxt(filepath, delimiter=",")
            else:
                loaded_coords[key] = value

        boundary_params_dict["coordinates"] = loaded_coords
    else:
        # Auto-load coordinates from shared_data/{img_name}/ if not explicitly provided
        coords_dict = {}
        for i in range(
            1, 4
        ):  # Try loading boundary1_coords, boundary2_coords, boundary3_coords
            coord_filename = f"boundary{i}_coords.csv"
            coord_filepath = os.path.join(
                os.path.dirname(__file__),
                "test_results",
                "shared_data",
                img_name,
                coord_filename,
            )
            if os.path.exists(coord_filepath):
                coords_dict[f"csv{i}"] = np.loadtxt(coord_filepath, delimiter=",")

        if coords_dict:
            boundary_params_dict["coordinates"] = coords_dict

    # Load boundary_img if it references a file
    if "boundary_img" in boundary_params_dict and boundary_params_dict["boundary_img"]:
        value = boundary_params_dict["boundary_img"]
        if isinstance(value, str) and value.startswith("from_file:"):
            filename = value.replace("from_file:", "")
            filepath = os.path.join(
                os.path.dirname(__file__),
                "test_results",
                filename,
            )
            boundary_params_dict["boundary_img"] = np.loadtxt(filepath, delimiter=",")

    return BoundaryParameters(**boundary_params_dict)


# --------------------------
# Tests
# --------------------------


@pytest.mark.parametrize(
    "test_name,test_case",
    load_test_cases(),
    ids=[name for name, _ in load_test_cases()],
)
def test_process_image_returns_fiber_features(test_name, test_case, tmp_path):
    """
    Test that process_image returns fiber features dataframe that matches expected results.
    Runs process_image with parameters from JSON test case.
    """
    # Load image
    img = load_test_image(test_case["image_params"]["img"])

    # Build ImageInputParameters from JSON
    image_params = ImageInputParameters(
        img=img,
        img_name=test_case["image_params"]["img_name"],
        slice_num=1,
        num_sections=1,
    )

    # Build FiberAnalysisParameters from JSON
    fiber_params_dict = test_case["fiber_params"].copy()
    fiber_params_dict["fire_directory"] = str(tmp_path)
    fiber_params = FiberAnalysisParameters(**fiber_params_dict)

    # Build OutputControlParameters from JSON
    output_params_dict = test_case["output_params"].copy()
    output_params_dict["output_directory"] = str(tmp_path)
    output_params = OutputControlParameters(**output_params_dict)

    # Build BoundaryParameters from JSON (may be None)
    # This handles loading CSV files referenced with "from_file:" markers
    # and auto-loads coordinates from shared_data/{img_name}/ if not explicitly provided
    boundary_params = load_boundary_data_from_json(
        test_case["boundary_params"], test_case["image_params"]["img_name"]
    )

    # Build AdvancedAnalysisOptions from JSON
    advanced_options = AdvancedAnalysisOptions(**test_case["advanced_options"])

    # Run process_image with parameters from test case
    results = process_image(
        image_params=image_params,
        fiber_params=fiber_params,
        output_params=output_params,
        boundary_params=boundary_params,
        advanced_options=advanced_options,
    )

    # Check that results were returned
    assert results is not None, "process_image should return results"
    assert "fib_feat_df" in results, "results should contain fib_feat_df"

    fib_feat_df = results["fib_feat_df"]

    # Check that we have results
    assert isinstance(fib_feat_df, pd.DataFrame), "fib_feat_df should be a DataFrame"
    assert len(fib_feat_df) > 0, "fib_feat_df should not be empty"

    # Compare with reference CSV if specified in test case
    if "matlab_reference_csv" in test_case:
        reference_csv_name = test_case["matlab_reference_csv"]
        reference_csv_path = os.path.join(
            os.path.dirname(__file__),
            "test_results",
            "process_image_test_files",
            reference_csv_name,
        )

        if os.path.exists(reference_csv_path):
            expected_fib_feat = pd.read_csv(reference_csv_path, header=None)

            # Check dimensions match
            assert len(fib_feat_df) == len(
                expected_fib_feat
            ), f"Row count mismatch: {len(fib_feat_df)} vs {len(expected_fib_feat)}"

            # Compare values (skip first column if it's an index/fiber_key)
            fib_feat_values = (
                fib_feat_df.iloc[:, 1:].values
                if fib_feat_df.shape[1] > 1
                else fib_feat_df.values
            )
            expected_values = (
                expected_fib_feat.iloc[:, 1:].values
                if expected_fib_feat.shape[1] > 1
                else expected_fib_feat.values
            )

            # Allow for floating point tolerance
            np.testing.assert_allclose(
                fib_feat_values,
                expected_values,
                rtol=0.05,
                atol=15,
                equal_nan=True,
                err_msg=f"Fiber features differ from reference {reference_csv_name}",
            )
            print(f"✓ Results match reference CSV: {reference_csv_name}")
        else:
            print(f"⚠ Reference CSV not found: {reference_csv_name}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
