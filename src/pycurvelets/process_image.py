from curvelops import FDCT2D, curveshow, fdct2d_wrapper
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
import time
import new_curv
import get_ct
from pycurvelets.models import CurveletControlParameters, FeatureControlParameters


def process_image(
    img,
    img_name,
    output_directory,
    keep,
    coordinates,
    distance_threshold,
    make_associations,
    make_map,
    make_overlay,
    make_feature_file,
    slice_num,
    tif_boundary,
    boundary_img,
    fire_directory,
    fiber_mode,
    num_sections,
    advanced_options,
):
    """
    process_image - Process images for fiber/curvelet analysis.

    This function provides three main analysis modes:
        1. Boundary analysis: Compare fiber angles to boundary angles and generate statistics.
        2. Absolute angle analysis: Return absolute fiber angles and statistics.
        3. Optionally use FIRE results if `fire_directory` is provided.

    Parameters
    ----------
    img : ndarray
        2D image array of size [M, N] to be analyzed.
    img_name : str
        Name of the image (without path), used for labeling outputs.
    output_directory : str
        Directory where results will be saved.
    keep : float
        Percentage of curvelet coefficients to keep for analysis.
    coordinates : ndarray
        Coordinates of a boundary for evaluation.
    distance_threshold : float
        Distance from the boundary to evaluate fibers/curvelets.
    make_associations : bool
        Whether to generate association plots between boundary and fibers.
    make_map : bool
        Whether to generate heatmap.
    make_overlay : bool
        Whether to generate curvelet overlay figure.
    make_feature_file : bool
        Whether to generate feature file.
    slice_num : int
        Slice number within a stack (if analyzing multiple slices).
    tif_boundary : bool
        Flag indicating whether the boundary input is a TIFF file (True) or a list of coordinates (False).
    boundary_img : ndarray
        Boundary image used for overlay outputs.
    fire_directory : str
        Directory containing FIRE fiber results (optional; used instead of curvelets).
    fiber_mode : str
        Determines fiber processing method:
        0 = curvelet, 1/2/3 = FIRE variants
    num_sections : int
        Number of sections within a stack (optional; used instead of curvelets).
    advanced_options : dict
        Dictionary containing advanced interface controls

        **Required Keys**
            - minimum_nearest_fibers: int
            Number of nearest fibers to consider.
            - minimum_box_size: int
            Minimum box size in pixels.
            - fiber_midpoint_estimate: int
            Estimate of the midpoint of the fibers.

        **Optional Keys**
            - exclude_fibers_in_mask_flag : int
                1 = exclude fibers inside the boundary mask, 0 = keep them.
            - curvelets_group_radius : float
                Radius to group nearby curvelets.
            - selected_scale : int
                Curvelet scale used for analysis. Must be in the range
                `[2, ceil(log2(min(M, N)) - 4)]`, where M and N are the image dimensions.
                Default is the 2nd finest scale: `ceil(log2(min(M, N)) - 3) - 1`.
            - heatmap_STD_filter_size : int
                Size of standard deviation filter for heatmap (default 24).
            - heatmap_SQUARE_max_filter_size : int
                Size of max filter for heatmap (default 12).
            - heatmap_GAUSSIAN_disc_filter_sigma : float
                Sigma for Gaussian disc filter (default 4).

    Returns
    -------
    bool
        Returns True if the image was processed successfully.

    Notes
    -----
    This function can generate:
        - Histogram data of fiber angles relative to boundary.
        - Reconstructed curvelet image (if curvelet transform is used).
        - Compass plot values.
        - Distances to boundary for each fiber/curvelet.
        - Statistical summaries of angles and correlations.
        - Filtered spatial correlation maps of curvelet angles.
    """

    # Get screen size for figure position
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    exclude_fibers_in_mask_flag = advanced_options["exclude_fibers_in_mask_flag"]
    img_name_length = len(img_name)
    img_name_plain = img_name  # plain image name, without slice number
    if num_sections > 1:
        img_name = f"{img_name[:img_name_length]}_s{slice_num}"

    print(f"Image name: {img_name_plain}")

    if num_sections > 1:
        print(f"Slide number: {slice_num}")

    # Check if we are measuring with respect to boundary
    boundary_measurement = coordinates.size > 0

    start_time = time.perf_counter()
    # Feature control structure initialized: feature_cp
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=advanced_options["minimum_nearest_fibers"],
        minimum_box_size=advanced_options["minimum_box_size"],
        fiber_midpoint_estimate=advanced_options["fiber_midpoint_estimate"],
    )

    # Add lower limit for distance threshold
    min_dist = advanced_options["min_dist"]

    # Get features that are only based on fibers
    if fiber_mode == 0:
        print("Computing curvelet transform.")
        curve_cp = CurveletControlParameters(
            keep=keep,
            scale=advanced_options["selected_scale"],
            radius=advanced_options["curvelets_group_radius"],
        )
        # Call getCT
    else:
        print("Reading CT-FIRE database.")
        # Call getFIRE

    return True
