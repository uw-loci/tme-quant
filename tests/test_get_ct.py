import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pycurvelets.models import CurveletControlParameters, FeatureControlParameters

# By default, skip curvelops-dependent tests (e.g., on CI). Enable locally with:
#   TMEQ_RUN_CURVELETS=1 pytest -q
if os.environ.get("TMEQ_RUN_CURVELETS") != "1":
    pytest.skip(
        "curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
        allow_module_level=True,
    )

# Optional: attempt import; skip module if curvelet backend is missing
try:
    from pycurvelets.get_ct import get_ct
except ModuleNotFoundError:
    pytest.skip(
        "curvelops not available; skipping get_ct tests", allow_module_level=True
    )

from pycurvelets.get_ct import get_ct


@pytest.fixture(scope="module")
def standard_test_image():
    """Load a standard test image once for the module."""
    img_path = os.path.join(os.path.dirname(__file__), "test_images", "real1.tif")
    img = plt.imread(img_path, format="TIF")
    return img


@pytest.fixture(scope="module")
def standard_control_parameters():
    """Create standard control parameters for testing."""
    advanced_options = {
        "exclude_fibers_in_mask_flag": 1,
        "curvelets_group_radius": 10,
        "selected_scale": 1,
        "heatmap_STD_filter_size": 16,
        "heatmap_SQUARE_max_filter_size": 12,
        "heatmap_GAUSSIAN_disc_filter_sigma": 4,
        "plot_rgb_flag": 0,
        "minimum_nearest_fibers": 2,
        "minimum_box_size": 32,
        "fiber_midpoint_estimate": 1,
        "min_dist": [],
    }

    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=advanced_options["minimum_nearest_fibers"],
        minimum_box_size=advanced_options["minimum_box_size"],
        fiber_midpoint_estimate=advanced_options["fiber_midpoint_estimate"],
    )

    curve_cp = CurveletControlParameters(
        keep=0.05,
        scale=advanced_options["selected_scale"],
        radius=advanced_options["curvelets_group_radius"],
    )

    return curve_cp, feature_cp


def test_get_ct_matches_expected_results(
    standard_test_image, standard_control_parameters
):
    """
    Test that get_ct produces density_df and alignment_df that match expected results.
    The expected CSV files contain raw feature arrays (9 columns), while get_ct returns
    summary statistics (mean and std). This test computes the expected mean/std from
    the raw data and compares with get_ct output.
    """
    img = standard_test_image
    curve_cp, feature_cp = standard_control_parameters

    # Run get_ct
    fiber_structure, density_df, alignment_df = get_ct(img, curve_cp, feature_cp)

    # Load expected raw results (9 columns: 4 nearest_fibers + 2 + 3 box_sizes)
    test_results_dir = os.path.join(
        os.path.dirname(__file__), "test_results", "process_image_test_files"
    )

    expected_density_raw = pd.read_csv(
        os.path.join(test_results_dir, "real1_density_df.csv"), header=None
    )
    expected_alignment_raw = pd.read_csv(
        os.path.join(test_results_dir, "real1_alignment_df.csv"), header=None
    )

    # Check that we have results
    assert len(fiber_structure) > 0, "fiber_structure should not be empty"
    assert len(density_df) > 0, "density_df should not be empty"
    assert len(alignment_df) > 0, "alignment_df should not be empty"

    # Compute expected mean and std from first 4 columns (nearest_fibers features)
    # This matches what process_fibers does: mean/std of first len(nearest_fibers) columns
    expected_density_mean = expected_density_raw.iloc[:, :4].mean(axis=1).values
    expected_density_std = expected_density_raw.iloc[:, :4].std(axis=1, ddof=0).values

    expected_alignment_mean = expected_alignment_raw.iloc[:, :4].mean(axis=1).values
    expected_alignment_std = (
        expected_alignment_raw.iloc[:, :4].std(axis=1, ddof=0).values
    )

    # Create expected dataframes with same structure as actual output
    expected_density = pd.DataFrame(
        {"density_mean": expected_density_mean, "density_std": expected_density_std}
    )
    expected_alignment = pd.DataFrame(
        {
            "alignment_mean": expected_alignment_mean,
            "alignment_std": expected_alignment_std,
        }
    )

    # Check dimensions match
    assert len(density_df) == len(
        expected_density
    ), f"density_df length mismatch: {len(density_df)} vs {len(expected_density)}"
    assert len(alignment_df) == len(
        expected_alignment
    ), f"alignment_df length mismatch: {len(alignment_df)} vs {len(expected_alignment)}"

    density_values = density_df.values
    expected_density_values = expected_density.values
    alignment_values = alignment_df.values
    expected_alignment_values = expected_alignment.values

    # Compare directly without sorting
    # The order from get_ct matches the order in the CSV files
    np.testing.assert_allclose(
        density_values,
        expected_density_values,
        rtol=1e-3,
        atol=1e-3,
        err_msg="density_df values differ from expected results",
    )

    np.testing.assert_allclose(
        alignment_values,
        expected_alignment_values,
        rtol=1e-2,
        atol=0.05,
        err_msg="alignment_df values differ from expected results",
    )


def test_get_ct_returns_valid_structure(
    standard_test_image, standard_control_parameters
):
    """
    Test that get_ct returns valid data structures with expected properties.
    """
    img = standard_test_image
    curve_cp, feature_cp = standard_control_parameters

    fiber_structure, density_df, alignment_df = get_ct(img, curve_cp, feature_cp)

    # Check fiber_structure
    assert isinstance(fiber_structure, pd.DataFrame)
    assert len(fiber_structure) > 0
    assert "angle" in fiber_structure.columns
    assert "center_row" in fiber_structure.columns
    assert "center_col" in fiber_structure.columns

    # Check density_df
    assert isinstance(density_df, pd.DataFrame)
    assert len(density_df) > 0
    assert density_df.shape[1] == 2, "density_df should have 2 columns (mean, std)"
    assert "density_mean" in density_df.columns
    assert "density_std" in density_df.columns

    # Check alignment_df
    assert isinstance(alignment_df, pd.DataFrame)
    assert len(alignment_df) > 0
    assert alignment_df.shape[1] == 2, "alignment_df should have 2 columns (mean, std)"
    assert "alignment_mean" in alignment_df.columns
    assert "alignment_std" in alignment_df.columns

    # Check that all values are finite
    assert np.isfinite(density_df.values).all(), "density_df contains non-finite values"
    assert np.isfinite(
        alignment_df.values
    ).all(), "alignment_df contains non-finite values"
