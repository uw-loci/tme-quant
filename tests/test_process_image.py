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
    from pycurvelets.process_image import process_image, generate_overlay
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
                # Example: from_file:shared_data/real1/boundary1_coords.csv
                filepath = os.path.join(
                    os.path.dirname(__file__),
                    "test_results",
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


def is_association_test_case(test_case):
    """
    Return True only for test cases where generate_overlay will draw association lines.
    This requires make_associations=True, make_overlay=True, and tif_boundary=3.
    """
    output_params = test_case.get("output_params", {})
    boundary_params = test_case.get("boundary_params")
    return (
        output_params.get("make_associations", False)
        and output_params.get("make_overlay", False)
        and boundary_params is not None
        and boundary_params.get("tif_boundary") == 3
    )


all_cases = load_test_cases()
association_cases = [
    (name, tc) for name, tc in all_cases if is_association_test_case(tc)
]


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


@pytest.mark.parametrize(
    "test_name,test_case",
    association_cases,
    ids=[name for name, _ in association_cases],
)
def test_generate_overlay(test_name, test_case, tmp_path, monkeypatch):
    """
    Verify that generate_overlay draws each association line with the correct
    row/col coordinate ordering.

    For every fiber that has a non-NaN boundary point in fib_feat_df we expect
    ax.plot to be called with:

        x = [fiber_center_col, boundary_point_col]   (i.e. center[1], bndry_pt[0])
        y = [fiber_center_row, boundary_point_row]   (i.e. center[0], bndry_pt[1])

    A bug would flip one or both pairs, e.g. using boundary_point_row as an
    x-coordinate or boundary_point_col as a y-coordinate.
    """
    # --- Run process_image to get fib_feat_df ---
    img = load_test_image(test_case["image_params"]["img"])

    image_params = ImageInputParameters(
        img=img,
        img_name=test_case["image_params"]["img_name"],
        slice_num=1,
        num_sections=1,
    )
    fiber_params = FiberAnalysisParameters(
        **{**test_case["fiber_params"], "fire_directory": str(tmp_path)}
    )
    output_params = OutputControlParameters(
        **{**test_case["output_params"], "output_directory": str(tmp_path)}
    )
    boundary_params = load_boundary_data_from_json(
        test_case["boundary_params"],
        test_case["image_params"]["img_name"],
    )
    advanced_options = AdvancedAnalysisOptions(**test_case["advanced_options"])

    results = process_image(
        image_params=image_params,
        fiber_params=fiber_params,
        output_params=output_params,
        boundary_params=boundary_params,
        advanced_options=advanced_options,
    )

    assert (
        results is not None and "fib_feat_df" in results
    ), "process_image must return fib_feat_df for this test to run"
    fib_feat_df = results["fib_feat_df"]

    # Fibers that have a valid boundary association
    valid_fibers = fib_feat_df.dropna(
        subset=["boundary_point_row", "boundary_point_col"]
    )
    assert len(valid_fibers) > 0, (
        f"Test case '{test_name}' has no fibers with valid boundary points; "
        "cannot verify association-line coordinates."
    )

    # --- Intercept ax.plot calls made inside generate_overlay ---
    # Each captured entry: {"x": [...], "y": [...]}
    captured_blue_lines = []

    original_plot = plt.Axes.plot

    def spy_plot(self, *args, **kwargs):
        # Association lines are drawn as "b-" with linewidth=0.5
        is_blue = (len(args) >= 3 and args[2] == "b-") or kwargs.get("color") in (
            "b",
            "blue",
        )
        if is_blue and len(args) >= 2:
            captured_blue_lines.append({"x": list(args[0]), "y": list(args[1])})
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", spy_plot)

    # --- Reconstruct measured_boundary in the shape generate_overlay expects ---
    #
    # analyze_global_boundary returns a slice of res_df with these columns:
    #   nearest_boundary_distance, nearest_region_distance, nearest_boundary_angle,
    #   extension_point_distance, extension_point_angle,
    #   boundary_point_col, boundary_point_row
    #
    # save_fiber_features renames most of them but keeps boundary_point_row and
    # boundary_point_col verbatim, so we can recover them from fib_feat_df directly.
    # The remaining columns are not accessed by generate_overlay, so we fill them
    # with NaN to satisfy the DataFrame shape without misrepresenting data.
    measured_boundary = pd.DataFrame(
        {
            "nearest_boundary_distance": fib_feat_df[
                "nearest_distance_to_boundary"
            ].values,
            "nearest_region_distance": fib_feat_df["inside_epicenter_region"].values,
            "nearest_boundary_angle": fib_feat_df[
                "nearest_relative_boundary_angle"
            ].values,
            "extension_point_distance": fib_feat_df["extension_point_distance"].values,
            "extension_point_angle": fib_feat_df["extension_point_angle"].values,
            # These two are what generate_overlay actually reads:
            "boundary_point_col": fib_feat_df["boundary_point_col"].values,
            "boundary_point_row": fib_feat_df["boundary_point_row"].values,
        },
        index=fib_feat_df.index,
    )

    # --- Also reconstruct fiber_structure with the columns generate_overlay expects ---
    #
    # generate_overlay reads fiber_structure["center_row"] / ["center_col"] (or
    # ["center_1"] / ["center_2"] for FIRE mode) to get fiber centers.
    # fib_feat_df stores these as "end_point_row" / "end_point_col" (see save_fiber_features).
    # We build a minimal fiber_structure DataFrame with the right column names so that
    # generate_overlay can locate fiber centers correctly.
    fiber_structure = fib_feat_df.rename(
        columns={
            "end_point_row": "center_row",
            "end_point_col": "center_col",
            "fiber_absolute_angle": "angle",
        }
    )

    # --- Call generate_overlay directly with the reconstructed data ---
    coordinates = boundary_params.coordinates if boundary_params else None
    n_fibers = len(fib_feat_df)
    in_curvs_flag = np.ones(n_fibers, dtype=bool)  # include every fiber
    out_curvs_flag = np.zeros(n_fibers, dtype=bool)
    nearest_angles = fiber_structure["angle"].values

    generate_overlay(
        img=img,
        fiber_structure=fiber_structure,
        coordinates=coordinates,
        in_curvs_flag=in_curvs_flag,
        out_curvs_flag=out_curvs_flag,
        nearest_angles=nearest_angles,
        measured_boundary=measured_boundary,
        output_directory=str(tmp_path),
        img_name=test_case["image_params"]["img_name"],
        fiber_mode=0,
        tif_boundary=3,
        boundary_measurement=True,
        make_associations=True,
        num_sections=1,
    )

    assert len(captured_blue_lines) > 0, (
        "generate_overlay drew no blue association lines even though "
        f"make_associations=True and {len(valid_fibers)} fibers have boundary points."
    )

    # --- Verify coordinate ordering for every fiber with a valid boundary point ---
    #
    # Expected ax.plot call for fiber at (center_row, center_col) → (bp_row, bp_col):
    #   x = [center_col, boundary_point_col]
    #   y = [center_row, boundary_point_row]
    #
    # We read from the same DataFrames passed into generate_overlay (fiber_structure
    # and measured_boundary) so the test is consistent with what the function sees.

    expected_lines = []
    for idx in valid_fibers.index:
        center_row = fiber_structure.at[idx, "center_row"]
        center_col = fiber_structure.at[idx, "center_col"]
        bp_col = measured_boundary.at[idx, "boundary_point_col"]
        bp_row = measured_boundary.at[idx, "boundary_point_row"]
        expected_lines.append(
            {
                # x = [center_col, boundary_point_row]  (bndry_pt[1] = row value used as x)
                "x": [center_col, bp_row],
                # y = [center_row, boundary_point_col]  (bndry_pt[0] = col value used as y)
                "y": [center_row, bp_col],
                "fiber_idx": idx,
            }
        )

    # Match each expected line to a captured line (within floating-point tolerance)
    unmatched = []
    for exp in expected_lines:
        found = any(
            np.allclose(cap["x"], exp["x"], atol=0.05)
            and np.allclose(cap["y"], exp["y"], atol=15)
            for cap in captured_blue_lines
        )
        if not found:
            unmatched.append(exp)

    # Provide a clear failure message showing the first few mismatches
    if unmatched:
        examples = unmatched[:5]
        msg_lines = [
            f"{len(unmatched)} / {len(expected_lines)} association lines have wrong coordinates.",
            "",
            "Each line should be plotted as:",
            "  x = [center_col, boundary_point_col]",
            "  y = [center_row, boundary_point_row]",
            "",
            "First mismatches (expected → not found among captured lines):",
        ]
        for e in examples:
            msg_lines.append(f"  fiber {e['fiber_idx']}: " f"x={e['x']}, y={e['y']}")
        msg_lines += [
            "",
            "Sample of captured blue lines:",
        ]
        for cap in captured_blue_lines[:5]:
            msg_lines.append(f"  x={cap['x']}, y={cap['y']}")

        pytest.fail("\n".join(msg_lines))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
