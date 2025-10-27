import numpy as np
import os
from scipy.io import loadmat

from pycurvelets.models import FeatureControlParameters
from pycurvelets.process_fibers import process_fibers


def get_fire(
    img_name: str,
    fire_directory: str,
    fiber_mode: int,
    feature_cp: FeatureControlParameters,
):
    """
    Loads fiber output from CT-Fire and converts it into a structured format usable by CurveAlign.

    Parameters
    ----------
    img_name : str
        Name of the image we are processing.
    fire_directory : str
        Directory containing FIRE fiber results (optional; used instead of curvelets).
    fiber_mode : int
        Determines fiber processing method:
        0 = curvelet, 1/2/3 = FIRE variants
    feature_cp : FeatureControlParameters
        Control parameters for extracted features (e.g., minimum fibers, box size).

    Returns
    -------
    fiber_structure: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``angle``: orientation angle in degrees
        - ``center_row``: row coordinate of the curvelet center
        - ``center_col``: column coordinate of the curvelet center
        - ``width``: width of the curvelet
    density_df: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``density_mean``: mean density of fibers
        - ``density_std``: standard deviation of density of fibers
    alignment_df: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``alignment_mean``: mean alignment of fibers
        - ``alignment_std``: standard deviation of alignment of fibers
    """

    ct_fire_name = f"ct_fire_out{img_name}.mat"

    # Build full path
    file_path = os.path.join(fire_directory, "ct_fire_out", ct_fire_name)

    # Load .mat file
    fiber_structure_mat = loadmat(file_path)

    width_option_flag = 0  # default
    try:
        fiber_cp = fiber_structure_mat["cP"]
        width_max = fiber_cp["widMAX"][0, 0]  # MATLAB scalar to Python
        width_cp = fiber_cp["widcon"][0, 0]  # nested struct

        width_min_max_fiber_width = width_cp["wid_mm"][0, 0]
        width_min_pts = width_cp["wid_mp"][0, 0]
        width_confidence_region = width_cp["wid_sigma"][0, 0]  # default +/- 1 sigma
        width_option = width_cp["wid_opt"][0, 0]

        # Check widMAX
        if width_max < width_min_max_fiber_width:
            print(
                f"Please make sure the maximum fiber width is correct. Using default min maximum width {width_min_max_fiber_width}."
            )
            width_threshold = width_min_max_fiber_width
        else:
            width_threshold = width_max

        width_option_flag = 1
        print("Use advanced width calculation method.")

    except Exception as e:
        print(str(e))
        width_option_flag = 0
        print("Use the original width calculation method.")

    fiber_structure = fiber_structure_mat["data"]
    length_limit = fiber_structure_mat["cP"]["LL1"][0, 0]
    density_df, alignment_df = process_fibers(
        fiber_structure=fiber_structure, feature_cp=feature_cp
    )

    if len(fiber_structure) == 0:
        return fiber_structure, density_df, alignment_df

    fiber_number = len(fiber_structure_mat["Fai"][0, 0])
    X = fiber_structure["Xai"]

    # Now process segments:

    return fiber_structure, density_df, alignment_df
