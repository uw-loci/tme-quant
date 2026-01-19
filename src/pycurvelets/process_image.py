from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage.draw import polygon, polygon2mask
from skimage.measure import regionprops, label, find_contours
import tkinter as tk
import time
from multiprocessing import Pool
from typing import Optional

from curvelops import FDCT2D, curveshow, fdct2d_wrapper

from pycurvelets.get_alignment_to_roi import get_alignment_to_roi
from pycurvelets.models import (
    AdvancedAnalysisOptions,
    BoundaryParameters,
    CurveletControlParameters,
    FeatureControlParameters,
    FiberAnalysisParameters,
    ImageInputParameters,
    OutputControlParameters,
    ROIList,
)
from pycurvelets.get_ct import get_ct
from pycurvelets.get_fire import get_fire
from pycurvelets.get_tif_boundary import get_tif_boundary
from pycurvelets.utils.misc import format_df_to_excel
from pycurvelets.utils.visualization import draw_curvs, draw_map


def process_image(
    image_params: ImageInputParameters,
    fiber_params: FiberAnalysisParameters,
    output_params: OutputControlParameters,
    boundary_params: Optional[BoundaryParameters] = None,
    advanced_options: Optional[AdvancedAnalysisOptions] = None,
):
    """
    process_image - Process images for fiber/curvelet analysis.

    This function provides three main analysis modes:
        1. Boundary analysis: Compare fiber angles to boundary angles and generate statistics.
        2. Absolute angle analysis: Return absolute fiber angles and statistics.
        3. Optionally use FIRE results if `fiber_params.fire_directory` is provided.

    Parameters
    ----------
    image_params : ImageInputParameters
        Image input parameters including:
        - img : ndarray - 2D image array of size [M, N] to be analyzed
        - img_name : str - Name of the image (without path), used for labeling outputs
        - slice_num : int - Slice number within a stack (default 1)
        - num_sections : int - Number of sections within a stack (default 1)
    
    fiber_params : FiberAnalysisParameters
        Fiber analysis method parameters including:
        - fiber_mode : int - Determines fiber processing method (0=curvelet, 1/2/3=FIRE)
        - keep : float - Percentage of curvelet coefficients to keep (default 0.05)
        - fire_directory : str - Directory containing FIRE fiber results (optional)
    
    output_params : OutputControlParameters
        Output generation control flags including:
        - output_directory : str - Directory where results will be saved
        - make_associations : bool - Generate association plots between boundary and fibers
        - make_map : bool - Generate heatmap
        - make_overlay : bool - Generate curvelet overlay figure
        - make_feature_file : bool - Generate feature file
    
    boundary_params : BoundaryParameters, optional
        Boundary analysis parameters including:
        - coordinates : dict - Coordinates of a boundary for evaluation
        - distance_threshold : float - Distance from boundary to evaluate fibers/curvelets
        - tif_boundary : int - Boundary type (0=none, 1/2=CSV, 3=TIFF)
        - boundary_img : ndarray - Boundary image used for overlay outputs
    
    advanced_options : AdvancedAnalysisOptions, optional
        Advanced interface controls including:
        - exclude_fibers_in_mask_flag : int - Exclude fibers inside boundary mask
        - curvelets_group_radius : float - Radius to group nearby curvelets
        - selected_scale : int - Curvelet scale for analysis
        - min_dist : list/float - Minimum distance to boundary
        - minimum_nearest_fibers : int - Number of nearest fibers to consider
        - minimum_box_size : int - Minimum box size in pixels
        - fiber_midpoint_estimate : int - Estimate of fiber midpoint
        - heatmap_STD_filter_size : int - STD filter size for heatmap
        - heatmap_SQUARE_max_filter_size : int - Max filter size for heatmap
        - heatmap_GAUSSIAN_disc_filter_sigma : float - Gaussian sigma for heatmap

    Returns
    -------
    dict or None
        Returns a dictionary containing processing results if successful:
        - 'fib_feat_df': pd.DataFrame (if make_feature_file is True)
            Fiber features dataframe with all computed features
        Returns None if processing failed.

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

    # Extract parameters from dataclasses
    img = image_params.img
    img_name = image_params.img_name
    slice_num = image_params.slice_num
    num_sections = image_params.num_sections
    
    fiber_mode = fiber_params.fiber_mode
    keep = fiber_params.keep
    fire_directory = fiber_params.fire_directory
    
    output_directory = output_params.output_directory
    make_associations = output_params.make_associations
    make_map = output_params.make_map
    make_overlay = output_params.make_overlay
    make_feature_file = output_params.make_feature_file
    
    # Initialize defaults if not provided
    if boundary_params is None:
        boundary_params = BoundaryParameters()
    if advanced_options is None:
        advanced_options = AdvancedAnalysisOptions()
    
    coordinates = boundary_params.coordinates
    distance_threshold = boundary_params.distance_threshold
    tif_boundary = boundary_params.tif_boundary
    boundary_img = boundary_params.boundary_img
    
    # Initialize parameters
    params = initialize_parameters(
        img_name=img_name,
        num_sections=num_sections,
        slice_num=slice_num,
        coordinates=coordinates,
        advanced_options=advanced_options,
        tif_boundary=tif_boundary,
        boundary_img=boundary_img,
    )

    exclude_fibers_in_mask_flag = params["exclude_fibers_in_mask_flag"]
    img_name_plain = params["img_name_plain"]
    img_name = params["img_name"]
    boundary_measurement = params["boundary_measurement"]
    feature_cp = params["feature_cp"]
    min_dist = params["min_dist"]

    # Initialize attributes for fiber features
    fiber_structure = pd.DataFrame()
    fiber_key = []
    total_length_list = []
    end_length_list = []
    curvature_list = []
    width_list = []
    density_df = pd.DataFrame()
    alignment_df = pd.DataFrame()
    curvelet_coefficients = None

    # Get features that are only based on fibers
    t_fiber_start = time.perf_counter()
    if fiber_mode == 0:
        print("Computing curvelet transform.")
        curve_cp = CurveletControlParameters(
            keep=keep,
            scale=advanced_options.selected_scale if isinstance(advanced_options, AdvancedAnalysisOptions) else advanced_options["selected_scale"],
            radius=advanced_options.curvelets_group_radius if isinstance(advanced_options, AdvancedAnalysisOptions) else advanced_options["curvelets_group_radius"],
        )
        # Call getCT
        fiber_structure, density_df, alignment_df, curvelet_coefficients = get_ct(
            img, curve_cp, feature_cp
        )
    else:
        print("Reading CT-FIRE database.")
        # Call getFIRE
        # Add slice name used in CT-FIRE output
        fiber_structure, density_df, alignment_df = get_fire(
            img_name_plain, fire_directory, fiber_mode, feature_cp
        )
    t_fiber_end = time.perf_counter()
    print(f"⏱️  Fiber extraction took: {t_fiber_end - t_fiber_start:.2f}s")

    if fiber_structure.empty:
        return None

    # Initialize variables that may be set in boundary analysis
    measured_boundary = None
    nearest_angles = None
    in_curvs_flag = None
    bins = np.arange(2.5, 180, 5)  # Default bins for histogram

    # Get features correlating fibers to boundaries
    if boundary_measurement:
        print("Analyzing boundary.")
        t_boundary_start = time.perf_counter()

        if tif_boundary == 3:
            # Extract boundary coordinates from mask if not provided
            if coordinates is None and boundary_img is not None:
                print("Extracting boundary coordinates from mask image...")
                coordinates = extract_boundary_coords_from_mask(boundary_img)
            
            # Process ROIs in parallel only if coordinates are available
            if coordinates is not None and len(coordinates) > 0:
                (
                    roi_measurement_details,
                    roi_summary_details,
                    save_boundary_width_measurements,
                ) = process_tiff_boundary_rois(
                    coordinates=coordinates,
                    fiber_structure=fiber_structure,
                    distance_threshold=distance_threshold,
                    img=img,
                    output_directory=output_directory,
                    img_name=img_name,
                )
            else:
                # No ROI processing when coordinates not available
                print("Warning: No boundary coordinates available for ROI processing")
                roi_measurement_details = None
                roi_summary_details = None
                save_boundary_width_measurements = None

            # Analyze global fiber-boundary relationships
            boundary_results = analyze_global_boundary(
                coordinates=coordinates,
                boundary_img=boundary_img,
                fiber_structure=fiber_structure,
                distance_threshold=distance_threshold,
                min_dist=min_dist,
                exclude_fibers_in_mask_flag=exclude_fibers_in_mask_flag,
            )

            nearest_angles = boundary_results["nearest_angles"]
            in_curvs_flag = boundary_results["in_curvs_flag"]
            out_curvs_flag = boundary_results["out_curvs_flag"]
            distances = boundary_results["distances"]
            measured_boundary = boundary_results["measured_boundary"]
            bins = boundary_results["bins"]
        elif tif_boundary == 1 or tif_boundary == 2:
            # TODO: implement getBoundary
            return None

    else:
        # No boundary measurement - analyze absolute fiber angles
        print("No boundary analysis - using absolute fiber angles")

        if fiber_mode == 0:
            # Curvelet mode
            angles = fiber_structure["angle"].values
            distances = np.full(len(fiber_structure), np.nan)
            in_curvs_flag = np.ones(len(fiber_structure), dtype=bool)
            out_curvs_flag = np.zeros(len(fiber_structure), dtype=bool)
        else:
            # FIRE mode
            in_curvs_flag = np.ones(len(fiber_structure), dtype=bool)
            angles = fiber_structure["angle"].values
            distances = np.full(len(fiber_structure), np.nan)

        measured_bndry = 0
        num_im_pts = img.shape[0] * img.shape[1]  # count all pixels if no boundary
        bins = np.arange(2.5, 180, 5)  # 2.5:5:177.5

    # Save fiber features if make_feature_file is enabled
    fib_feat_df = None
    if make_feature_file:
        fib_feat_df = save_fiber_features(
            fiber_structure=fiber_structure,
            fiber_key=fiber_key,
            total_length_list=total_length_list,
            end_length_list=end_length_list,
            curvature_list=curvature_list,
            width_list=width_list,
            density_df=density_df,
            alignment_df=alignment_df,
            measured_boundary=measured_boundary,
            boundary_measurement=boundary_measurement,
            output_directory=output_directory,
            img_name=img_name,
            num_sections=num_sections,
            slice_num=slice_num,
        )

    # Create and save histogram of angles
    save_histogram(
        fiber_structure=fiber_structure,
        nearest_angles=nearest_angles,
        in_curvs_flag=in_curvs_flag,
        boundary_measurement=boundary_measurement,
        tif_boundary=tif_boundary,
        bins=bins,
        output_directory=output_directory,
        img_name=img_name,
    )

    # Inverse curvelet transform (only for curvelet mode)
    if fiber_mode == 0:
        generate_reconstruction(
            img=img,
            curvelet_coefficients=curvelet_coefficients,
            output_directory=output_directory,
            img_name=img_name,
            num_sections=num_sections,
        )

    # Create overlay figure if requested
    if make_overlay:
        generate_overlay(
            img=img,
            fiber_structure=fiber_structure,
            coordinates=coordinates,
            in_curvs_flag=(
                in_curvs_flag
                if boundary_measurement
                else np.ones(len(fiber_structure), dtype=bool)
            ),
            out_curvs_flag=(
                out_curvs_flag
                if boundary_measurement
                else np.zeros(len(fiber_structure), dtype=bool)
            ),
            nearest_angles=(
                nearest_angles
                if boundary_measurement and nearest_angles is not None
                else fiber_structure["angle"].values
            ),
            measured_boundary=measured_boundary if boundary_measurement else None,
            output_directory=output_directory,
            img_name=img_name,
            fiber_mode=fiber_mode,
            tif_boundary=tif_boundary,
            boundary_measurement=boundary_measurement,
            make_associations=make_associations,
            num_sections=num_sections,
        )

    # Generate heatmap if requested
    if make_map:
        generate_heatmap(
            img=img,
            fiber_structure=fiber_structure,
            in_curvs_flag=(
                in_curvs_flag
                if boundary_measurement
                else np.ones(len(fiber_structure), dtype=bool)
            ),
            angles=(
                nearest_angles
                if boundary_measurement and nearest_angles is not None
                else fiber_structure["angle"].values
            ),
            distances=distances if boundary_measurement else np.array([]),
            output_directory=output_directory,
            img_name=img_name,
            tif_boundary=tif_boundary,
            boundary_measurement=boundary_measurement,
            num_sections=num_sections,
            advanced_options=advanced_options,
        )

    # plt.show()

    # Return results for testing
    results = {}
    if fib_feat_df is not None:
        results["fib_feat_df"] = fib_feat_df

    return results if results else None


def process_single_roi(
    roi_index, roi_coords, fiber_structure, distance_threshold, img_shape
):
    """
    Process a single ROI - designed for parallel execution.

    Parameters
    ----------
    roi_index : int
        Index of the ROI being processed.
    roi_coords : list
        Coordinates of the ROI boundary.
    fiber_structure : pd.DataFrame
        DataFrame containing fiber information.
    distance_threshold : float
        Distance threshold for fiber selection.
    img_shape : tuple
        Shape of the image (height, width).

    Returns
    -------
    tuple
        (roi_index, roi_measurements, summary_row)
    """
    roi_list = ROIList(
        coordinates=[roi_coords],
        image_width=img_shape[1],
        image_height=img_shape[0],
    )
    roi_coords_array = np.array(roi_coords)
    roi_mask = polygon2mask(
        (roi_list.image_width, roi_list.image_height), roi_coords_array[:, [1, 0]]
    )
    roi_regions = regionprops(label(roi_mask.astype(int)))

    if len(roi_regions) != 1:
        raise ValueError(f"ROI {roi_index} does not correspond to a single region")

    roi_properties = roi_regions[0]
    orientation_degrees = -np.degrees(roi_properties.orientation)
    if orientation_degrees < 0:
        orientation_degrees += 180

    summary_row = {
        "ROI_id": roi_index + 1,
        "center_row": roi_properties.centroid[0],
        "center_col": roi_properties.centroid[1],
        "orientation": orientation_degrees,
        "area": roi_properties.area,
    }

    roi_measurements = None
    fiber_count = 0
    try:
        roi_measurements, fiber_count = get_alignment_to_roi(
            roi_list, fiber_structure, distance_threshold
        )
        print(f"ROI {roi_index}: Found {fiber_count} fibers")
    except Exception as e:
        print(f"ROI {roi_index} was skipped. Error message: {e}")

    return roi_index, roi_measurements, summary_row


def initialize_parameters(
    img_name, num_sections, slice_num, coordinates, advanced_options, 
    tif_boundary=0, boundary_img=None
):
    """
    Initialize parameters and control structures for image processing.

    Parameters
    ----------
    img_name : str
        Image name
    num_sections : int
        Number of sections in stack
    slice_num : int
        Current slice number
    coordinates : dict or None
        Boundary coordinates
    advanced_options : AdvancedAnalysisOptions or dict
        Advanced processing options
    tif_boundary : int, optional
        Boundary type (0=none, 1/2=CSV, 3=TIFF)
    boundary_img : ndarray, optional
        Boundary mask image

    Returns
    -------
    dict
        Dictionary containing initialized parameters:
        - exclude_fibers_in_mask_flag
        - img_name_plain
        - img_name (modified with slice number if needed)
        - boundary_measurement
        - feature_cp
        - min_dist
    """
    # Handle both dict and dataclass
    if isinstance(advanced_options, AdvancedAnalysisOptions):
        exclude_fibers_in_mask_flag = advanced_options.exclude_fibers_in_mask_flag
        minimum_nearest_fibers = advanced_options.minimum_nearest_fibers
        minimum_box_size = advanced_options.minimum_box_size
        fiber_midpoint_estimate = advanced_options.fiber_midpoint_estimate
        min_dist = advanced_options.min_dist
    else:
        exclude_fibers_in_mask_flag = advanced_options["exclude_fibers_in_mask_flag"]
        minimum_nearest_fibers = advanced_options["minimum_nearest_fibers"]
        minimum_box_size = advanced_options["minimum_box_size"]
        fiber_midpoint_estimate = advanced_options["fiber_midpoint_estimate"]
        min_dist = advanced_options.get("min_dist", [])
    
    img_name_length = len(img_name)
    img_name_plain = img_name  # plain image name, without slice number

    if num_sections > 1:
        img_name = f"{img_name[:img_name_length]}_s{slice_num}"

    print(f"Image name: {img_name_plain}")

    if num_sections > 1:
        print(f"Slide number: {slice_num}")

    # Check if we are measuring with respect to boundary
    # Boundary measurement is enabled if coordinates are provided OR if tif_boundary=3 with boundary_img
    boundary_measurement = bool(coordinates) or (tif_boundary == 3 and boundary_img is not None)

    # Feature control structure initialized: feature_cp
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=minimum_nearest_fibers,
        minimum_box_size=minimum_box_size,
        fiber_midpoint_estimate=fiber_midpoint_estimate,
    )

    return {
        "exclude_fibers_in_mask_flag": exclude_fibers_in_mask_flag,
        "img_name_plain": img_name_plain,
        "img_name": img_name,
        "boundary_measurement": boundary_measurement,
        "feature_cp": feature_cp,
        "min_dist": min_dist,
    }


def extract_boundary_coords_from_mask(boundary_img):
    """
    Extract boundary coordinates from a binary mask image.
    
    Parameters
    ----------
    boundary_img : ndarray
        Binary mask image where non-zero values represent the boundary/ROI regions
        
    Returns
    -------
    dict
        Dictionary mapping ROI indices to boundary coordinate arrays.
        Each coordinate array has shape (N, 2) with columns [row, col]
    """
    # Label connected regions in the mask
    labeled_mask = label(boundary_img > 0)
    num_regions = labeled_mask.max()
    
    if num_regions == 0:
        print("Warning: No boundary regions found in mask image")
        return {}
    
    print(f"Found {num_regions} boundary region(s) in mask image")
    
    coordinates = {}
    
    for region_id in range(1, num_regions + 1):
        # Create binary mask for this region
        region_mask = (labeled_mask == region_id).astype(np.uint8)
        
        # Find contours at 0.5 level (boundary between 0 and 1)
        contours = find_contours(region_mask, 0.5)
        
        if len(contours) > 0:
            # Use the longest contour for this region
            longest_contour = max(contours, key=len)
            # Store as [row, col] format
            coordinates[f'ROI_{region_id}'] = longest_contour
            print(f"  ROI {region_id}: {len(longest_contour)} boundary points")
    
    return coordinates


def analyze_global_boundary(
    coordinates,
    boundary_img,
    fiber_structure,
    distance_threshold,
    min_dist,
    exclude_fibers_in_mask_flag,
):
    """
    Analyze global fiber-boundary relationships using get_tif_boundary.

    Parameters
    ----------
    coordinates : dict
        Dictionary of ROI coordinates
    boundary_img : ndarray
        Boundary mask image
    fiber_structure : pd.DataFrame
        Fiber data
    distance_threshold : float
        Distance threshold for fiber selection
    min_dist : list or array
        Minimum distance threshold
    exclude_fibers_in_mask_flag : int
        Flag to exclude fibers inside mask (1=exclude, 0=keep)

    Returns
    -------
    dict
        Dictionary containing:
        - nearest_angles: Angles relative to boundary
        - in_curvs_flag: Boolean mask for included fibers
        - out_curvs_flag: Boolean mask for excluded fibers
        - distances: Distances to boundary
        - measured_boundary: Boundary measurements DataFrame
        - bins: Histogram bins
    """
    print("Calling get_tif_boundary for global fiber-boundary relationships...")

    res_mat, res_mat_names, num_im_pts, res_df = get_tif_boundary(
        coordinates=coordinates,
        img=boundary_img,
        obj=fiber_structure,
        dist_thresh=distance_threshold,
        min_dist=min_dist,
    )

    nearest_angles = res_df["nearest_boundary_angle"]

    if len(min_dist) == 0:
        in_curvs_flag = res_df["nearest_boundary_distance"] <= distance_threshold
    else:
        in_curvs_flag = (res_df["nearest_boundary_distance"] <= distance_threshold) & (
            res_df["nearest_boundary_distance"] > min_dist
        )
    out_curvs_flag = ~in_curvs_flag

    if exclude_fibers_in_mask_flag == 1:
        print(
            f"Excluding fibers in mask. Region distance values: {res_df['nearest_region_distance'].unique()}"
        )
        print(f"Before exclusion: {np.sum(in_curvs_flag)} fibers included")

        if len(min_dist) == 0:
            in_curvs_flag = (
                res_df["nearest_boundary_distance"] <= distance_threshold
            ) & (res_df["nearest_region_distance"] == 0)
        else:
            in_curvs_flag = (
                (res_df["nearest_boundary_distance"] <= distance_threshold)
                & (res_df["nearest_boundary_distance"] > min_dist)
                & (res_df["nearest_region_distance"] == 0)
            )
        out_curvs_flag = ~in_curvs_flag

        print(f"After exclusion: {np.sum(in_curvs_flag)} fibers included")
        print(
            f"Fibers with region_dist==0: {np.sum(res_df['nearest_region_distance'] == 0)}"
        )
        print(
            f"Fibers with region_dist==1: {np.sum(res_df['nearest_region_distance'] == 1)}"
        )

    distances = res_df["nearest_boundary_distance"]
    measured_boundary = res_df[
        [
            "nearest_boundary_distance",
            "nearest_region_distance",
            "nearest_boundary_angle",
            "extension_point_distance",
            "extension_point_angle",
            "boundary_point_row",
            "boundary_point_col",
        ]
    ]

    bins = np.arange(2.5, 90, 5)  # 2.5:5:87.5

    return {
        "nearest_angles": nearest_angles,
        "in_curvs_flag": in_curvs_flag,
        "out_curvs_flag": out_curvs_flag,
        "distances": distances,
        "measured_boundary": measured_boundary,
        "bins": bins,
    }


def process_tiff_boundary_rois(
    coordinates, fiber_structure, distance_threshold, img, output_directory, img_name
):
    """
    Process multiple ROIs for TIFF boundary analysis in parallel.

    Parameters
    ----------
    coordinates : dict
        Dictionary of ROI coordinates
    fiber_structure : pd.DataFrame
        Fiber data
    distance_threshold : float
        Distance threshold for fiber selection
    img : ndarray
        Image array
    output_directory : str
        Output directory
    img_name : str
        Image name

    Returns
    -------
    tuple
        (roi_measurement_details, roi_summary_details, save_boundary_width_measurements)
    """
    print(
        "Calculate relative angles of fibers within the specified distance to each boundary, including"
    )
    print(" 1) angle to the nearest point on the boundary")
    print(" 2) angle to the boundary mask orientation")
    print(" 3) angle to the line connecting the fiber center and boundary mask center")

    roi_number = len(coordinates)

    save_boundary_width_measurements = os.path.join(
        output_directory, f"{img_name}_BoundaryMeasurements.xlsx"
    )
    if os.path.isfile(save_boundary_width_measurements):
        os.remove(save_boundary_width_measurements)

    roi_measurement_details_cols = [
        "angle_to_boundary_edge",
        "angle_to_boundary_center",
        "angle_to_center_line",
        "fiber_center_row",
        "fiber_center_col",
        "fiber_angle_list",
        "distance_list",
        "boundary_point_row",
        "boundary_point_col",
    ]
    roi_measurement_details = pd.DataFrame(columns=roi_measurement_details_cols)

    roi_summary_cols = [
        "name",
        "center_row",
        "center_col",
        "orientation",
        "area",
        "mean_of_angle_to_boundary_edge",
        "mean_of_angle_to_boundary_center",
        "mean_of_angle_to_center_line",
        "number_of_fibers",
    ]
    roi_summary_details = pd.DataFrame(columns=roi_summary_cols)

    # Delete existing file if it exists to start fresh
    if os.path.exists(save_boundary_width_measurements):
        os.remove(save_boundary_width_measurements)

    format_df_to_excel(
        roi_summary_details,
        save_boundary_width_measurements,
        sheet_name="Boundary Summary",
    )

    # Process ROIs in parallel
    print(f"Processing {len(coordinates)} ROIs in parallel...")
    t_roi_start = time.perf_counter()
    with Pool() as pool:
        process_func = partial(
            process_single_roi,
            fiber_structure=fiber_structure,
            distance_threshold=distance_threshold,
            img_shape=img.shape,
        )

        results = pool.starmap(
            process_func,
            [(idx, coords) for idx, coords in enumerate(coordinates.values())],
        )
    t_roi_end = time.perf_counter()
    print(f"⏱️  ROI processing took: {t_roi_end - t_roi_start:.2f}s")

    # Collect results sequentially (file I/O must be sequential)
    for roi_index, roi_measurements, summary_row in results:
        roi_measurement_details, roi_summary_details = concat_roi_df(
            roi_measurements,
            roi_measurement_details,
            roi_summary_details,
            summary_row,
        )

        # Detailed measurements for each individual ROI
        if roi_measurements is not None and not roi_measurements.empty:
            format_df_to_excel(
                roi_measurements,
                save_boundary_width_measurements,
                sheet_name=f"ROI {roi_index + 1} Details",
                mode="a",
            )

        format_df_to_excel(
            roi_summary_details,
            save_boundary_width_measurements,
            sheet_name="Boundary Summary",
            mode="a",
        )

    return (
        roi_measurement_details,
        roi_summary_details,
        save_boundary_width_measurements,
    )


def save_fiber_features(
    fiber_structure,
    fiber_key,
    total_length_list,
    end_length_list,
    curvature_list,
    width_list,
    density_df,
    alignment_df,
    measured_boundary,
    boundary_measurement,
    output_directory,
    img_name,
    num_sections,
    slice_num,
):
    """
    Save fiber features to Excel file.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        Fiber data
    fiber_key : list
        Fiber identifiers
    total_length_list : list
        Total fiber lengths
    end_length_list : list
        End-to-end fiber lengths
    curvature_list : list
        Fiber curvatures
    width_list : list
        Fiber widths
    density_df : pd.DataFrame
        Density measurements
    alignment_df : pd.DataFrame
        Alignment measurements
    measured_boundary : pd.DataFrame or None
        Boundary measurements
    boundary_measurement : bool
        Whether boundary measurement was performed
    output_directory : str
        Output directory
    img_name : str
        Image name
    num_sections : int
        Number of sections in stack
    slice_num : int
        Current slice number

    Returns
    -------
    pd.DataFrame
        Fiber features dataframe with all computed features
    """
    print("Saving fiber features...")

    # Build fiber features DataFrame with descriptive column names
    fib_feat_data = {
        "fiber_key": fiber_key if fiber_key else list(range(len(fiber_structure))),
        "end_point_row": (
            fiber_structure["center_row"]
            if "center_row" in fiber_structure.columns
            else fiber_structure["center_1"]
        ),
        "end_point_col": (
            fiber_structure["center_col"]
            if "center_col" in fiber_structure.columns
            else fiber_structure["center_2"]
        ),
        "fiber_absolute_angle": fiber_structure["angle"],
        "fiber_weight": fiber_structure.get(
            "weight", pd.Series([np.nan] * len(fiber_structure))
        ),
        "total_length": (
            total_length_list if total_length_list else [np.nan] * len(fiber_structure)
        ),
        "end_to_end_length": (
            end_length_list if end_length_list else [np.nan] * len(fiber_structure)
        ),
        "curvature": (
            curvature_list if curvature_list else [np.nan] * len(fiber_structure)
        ),
        "width": width_list if width_list else [np.nan] * len(fiber_structure),
    }

    fib_feat_df = pd.DataFrame(fib_feat_data)

    # Add density columns if available
    if not density_df.empty:
        for col in density_df.columns:
            fib_feat_df[col] = density_df[col].values

    # Add alignment columns if available
    if not alignment_df.empty:
        for col in alignment_df.columns:
            fib_feat_df[col] = alignment_df[col].values

    # Add boundary-related features if boundary measurement was performed
    if boundary_measurement and measured_boundary is not None:
        # Map res_df column names to output column names
        boundary_col_mapping = {
            "nearest_boundary_distance": "nearest_distance_to_boundary",
            "nearest_region_distance": "inside_epicenter_region",
            "nearest_boundary_angle": "nearest_relative_boundary_angle",
            "extension_point_distance": "extension_point_distance",
            "extension_point_angle": "extension_point_angle",
            "boundary_point_row": "boundary_point_row",
            "boundary_point_col": "boundary_point_col",
        }
        for src_col, dest_col in boundary_col_mapping.items():
            if src_col in measured_boundary.columns:
                fib_feat_df[dest_col] = measured_boundary[src_col].values
            else:
                fib_feat_df[dest_col] = np.nan
    else:
        # No boundary - set boundary features to NaN
        boundary_cols = [
            "nearest_distance_to_boundary",
            "inside_epicenter_region",
            "nearest_relative_boundary_angle",
            "extension_point_distance",
            "extension_point_angle",
            "boundary_point_row",
            "boundary_point_col",
        ]
        for col_name in boundary_cols:
            fib_feat_df[col_name] = np.nan

    # Save fiber features to Excel
    if num_sections > 1:
        save_fib_feat = os.path.join(
            output_directory, f"{img_name}_s{slice_num}_FiberFeatures.xlsx"
        )
    else:
        save_fib_feat = os.path.join(output_directory, f"{img_name}_FiberFeatures.xlsx")

    format_df_to_excel(fib_feat_df, save_fib_feat, sheet_name="Fiber Features")
    print(f"Saved fiber features to {save_fib_feat}")

    return fib_feat_df


def generate_reconstruction(
    img, curvelet_coefficients, output_directory, img_name, num_sections
):
    """
    Generate and save inverse curvelet transform (reconstruction).

    Parameters
    ----------
    img : ndarray
        Original image
    curvelet_coefficients : array
        Curvelet coefficients from get_ct
    output_directory : str
        Output directory
    img_name : str
        Image name
    num_sections : int
        Number of sections in stack
    """
    print("Computing inverse curvelet transform.")
    M, N = img.shape
    is_real = 0  # 0 means complex
    ac = 0  # 1 is curvelets, 0 is wavelets
    nbscales = math.floor(math.log2(min(M, N)) - 3)
    nbangles_coarse = 16  # default
    temp_inverse = fdct2d_wrapper.fdct2d_inverse_wrap(
        img.shape[0],
        img.shape[1],
        nbscales,
        nbangles_coarse,
        ac,
        curvelet_coefficients,
    )

    recon = np.real(temp_inverse)

    # Save reconstructed image
    if num_sections > 1:
        save_recon = os.path.join(output_directory, f"{img_name}_reconstructed.tiff")
        # Append mode for multiple sections
        recon_img = Image.fromarray(recon.astype(np.float32))
        if os.path.exists(save_recon):
            recon_img.save(save_recon, append_images=[recon_img], save_all=True)
        else:
            recon_img.save(save_recon)
    else:
        save_recon = os.path.join(output_directory, f"{img_name}_reconstructed.tiff")
        Image.fromarray(recon.astype(np.float32)).save(save_recon)

    print(f"Saved reconstructed image to {save_recon}")


def save_histogram(
    fiber_structure,
    nearest_angles,
    in_curvs_flag,
    boundary_measurement,
    tif_boundary,
    bins,
    output_directory,
    img_name,
):
    """
    Create and save histogram of fiber angles.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        Fiber data with angle column
    nearest_angles : ndarray or None
        Angles relative to boundary (if boundary measurement)
    in_curvs_flag : ndarray or None
        Boolean mask for included fibers
    boundary_measurement : bool
        Whether boundary measurement is enabled
    tif_boundary : int
        Boundary type (0=none, 1/2=CSV, 3=TIFF)
    bins : ndarray
        Histogram bins
    output_directory : str
        Output directory
    img_name : str
        Image name
    """
    # Determine which angles to use
    if boundary_measurement:
        if tif_boundary == 3:
            # Only for tiff boundary - use angles within threshold
            if nearest_angles is not None and in_curvs_flag is not None:
                values = nearest_angles[in_curvs_flag]
            elif nearest_angles is not None:
                values = nearest_angles
            else:
                values = fiber_structure["angle"].values
        else:
            values = (
                nearest_angles
                if nearest_angles is not None
                else fiber_structure["angle"].values
            )
    else:
        values = (
            nearest_angles
            if nearest_angles is not None
            else fiber_structure["angle"].values
        )

    # Create histogram
    n, bin_edges = np.histogram(values, bins=bins)

    # Calculate bin centers (like MATLAB's hist)
    xout = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Prepare histogram data (frequency and bin centers)
    hist_data = np.vstack([n, xout])

    # Save histogram
    save_hist = os.path.join(output_directory, f"{img_name}_hist.csv")

    # Circularly shift by 1 (equivalent to MATLAB circshift)
    temp_hist = np.roll(hist_data, 1, axis=0)

    # Save transposed (rows become columns)
    np.savetxt(save_hist, temp_hist.T, delimiter=",", fmt="%.6f")
    print(f"Saved histogram to {save_hist}")

    # Plot histogram
    counts = hist_data[0]
    bin_centers = hist_data[1]

    plt.figure()
    plt.bar(bin_centers, counts, width=(bin_centers[1] - bin_centers[0]))
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Histogram")


def generate_overlay(
    img,
    fiber_structure,
    coordinates,
    in_curvs_flag,
    out_curvs_flag,
    nearest_angles,
    measured_boundary,
    output_directory,
    img_name,
    fiber_mode,
    tif_boundary,
    boundary_measurement,
    make_associations,
    num_sections,
):
    """
    Generate and save fiber overlay visualization.

    Parameters
    ----------
    img : ndarray
        Original grayscale image
    fiber_structure : pd.DataFrame
        Fiber data
    coordinates : dict
        Boundary coordinates
    in_curvs_flag : ndarray
        Boolean mask for included fibers
    out_curvs_flag : ndarray
        Boolean mask for excluded fibers
    nearest_angles : ndarray
        Fiber angles relative to boundary
    measured_boundary : pd.DataFrame
        Boundary point measurements
    output_directory : str
        Output directory
    img_name : str
        Image name
    fiber_mode : int
        Fiber processing method (0=curvelet, 1/2/3=FIRE)
    tif_boundary : int
        Boundary type (0=none, 1/2=CSV, 3=TIFF)
    boundary_measurement : bool
        Whether boundary measurement is enabled
    make_associations : bool
        Whether to draw association lines
    num_sections : int
        Number of sections in stack
    """
    print("Plotting overlay")

    # Determine fiber line length based on mode
    if fiber_mode == 0:
        fiber_len = 4  # Curvelet mode
    elif fiber_mode == 1:
        fiber_len = 2.5  # CT-FIRE minimum segment length
    else:  # fiber_mode 2 or 3
        fiber_len = 10  # CT-FIRE minimum fiber length

    # Create figure
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # Plot boundaries if present
    if boundary_measurement:
        if tif_boundary < 3:  # CSV boundary
            if coordinates:
                coords_array = np.array(list(coordinates.values())[0])
                ax.plot(coords_array[:, 1], coords_array[:, 0], "y-")
                ax.plot(coords_array[:, 1], coords_array[:, 0], "*y", markersize=3)
        elif tif_boundary == 3:  # TIFF boundary
            for roi_coords in coordinates.values():
                roi_coords_array = np.array(roi_coords)
                ax.plot(
                    roi_coords_array[:, 1],
                    roi_coords_array[:, 0],
                    "y-",
                    linewidth=1,
                )

    # Draw fibers using draw_curvs utility
    marksize = 3
    linewidth = 1

    if tif_boundary == 3:  # tiff boundary
        # Draw fibers that are used (color_flag=0 for green)
        if np.any(in_curvs_flag):
            draw_curvs(
                fiber_structure[in_curvs_flag],
                ax,
                fiber_len,
                color_flag=0,
                angles=nearest_angles[in_curvs_flag],
                mark_size=marksize,
                line_width=linewidth,
                boundary_measurement=boundary_measurement,
            )

        # Draw fibers that are not used (color_flag=1 for red)
        if np.any(out_curvs_flag):
            draw_curvs(
                fiber_structure[out_curvs_flag],
                ax,
                fiber_len,
                color_flag=1,
                angles=nearest_angles[out_curvs_flag],
                mark_size=marksize,
                line_width=linewidth,
                boundary_measurement=boundary_measurement,
            )

        # Draw associations between fibers and boundary points
        if boundary_measurement and make_associations:
            # Get fiber centers
            fiber_centers = (
                fiber_structure[["center_row", "center_col"]].values
                if "center_row" in fiber_structure.columns
                else fiber_structure[["center_1", "center_2"]].values
            )

            # Get included fibers and their boundary points
            in_curvs = fiber_centers[in_curvs_flag]
            in_bndry = measured_boundary[
                ["boundary_point_row", "boundary_point_col"]
            ].values[in_curvs_flag]

            # Plot lines connecting each fiber center to its nearest boundary point
            for center, bndry_pt in zip(in_curvs, in_bndry):
                if not np.isnan(bndry_pt[0]) and not np.isnan(bndry_pt[1]):
                    # Plot line from fiber center to boundary point
                    # MATLAB: plot([center(1,2) bndry(1)], [center(1,1) bndry(2)], 'b')
                    # bndry_pt is [row, col], center is [row, col]
                    ax.plot(
                        [center[1], bndry_pt[1]],  # x: col coordinates
                        [center[0], bndry_pt[0]],  # y: row coordinates
                        "b-",
                        linewidth=0.5,
                    )
    elif tif_boundary == 0:  # No boundary - draw all fibers
        # Draw all fibers in green
        draw_curvs(
            fiber_structure,
            ax,
            fiber_len,
            color_flag=0,  # Green color
            angles=nearest_angles,
            mark_size=marksize,
            line_width=linewidth,
            boundary_measurement=boundary_measurement,
        )
    # TODO: tif_boundary 1, 2 implementation

    print("Saving overlay")

    # Save overlay
    save_overlay = os.path.join(output_directory, f"{img_name}_overlay.tiff")

    plt.tight_layout(pad=0)
    plt.savefig(save_overlay, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved overlay to {save_overlay}")


def generate_heatmap(
    img,
    fiber_structure,
    in_curvs_flag,
    angles,
    distances,
    output_directory,
    img_name,
    tif_boundary,
    boundary_measurement,
    num_sections,
    advanced_options,
):
    """
    Generate and save a heatmap showing fiber alignment patterns.

    Parameters
    ----------
    img : ndarray
        Original grayscale image
    fiber_structure : pd.DataFrame
        Fiber data
    in_curvs_flag : ndarray
        Boolean mask for included fibers
    angles : ndarray
        Fiber angles (relative to boundary if boundary_measurement=True)
    distances : ndarray
        Distances to boundary
    output_directory : str
        Output directory
    img_name : str
        Image name
    tif_boundary : int
        Boundary type (0=none, 1/2=CSV, 3=TIFF)
    boundary_measurement : bool
        Whether boundary measurement is enabled
    num_sections : int
        Number of sections in stack
    advanced_options : dict
        Advanced options including heatmap parameters
    """

    print("Plotting map")

    # Get heatmap parameters
    if isinstance(advanced_options, AdvancedAnalysisOptions):
        map_params = {
            "STDfilter_size": advanced_options.heatmap_STD_filter_size,
            "SQUAREmaxfilter_size": advanced_options.heatmap_SQUARE_max_filter_size,
            "GAUSSIANdiscfilter_sigma": advanced_options.heatmap_GAUSSIAN_disc_filter_sigma,
        }
    else:
        map_params = {
            "STDfilter_size": advanced_options.get("heatmap_STD_filter_size", 24),
            "SQUAREmaxfilter_size": advanced_options.get(
                "heatmap_SQUARE_max_filter_size", 12
            ),
            "GAUSSIANdiscfilter_sigma": advanced_options.get(
                "heatmap_GAUSSIAN_disc_filter_sigma", 4
            ),
        }

    # Select fibers to use for map
    if tif_boundary == 0:  # No boundary
        map_fibers = fiber_structure[in_curvs_flag]
        map_angles = angles[in_curvs_flag]
    elif tif_boundary in [1, 2]:  # CSV boundary
        map_fibers = fiber_structure[in_curvs_flag]
        map_angles = angles[in_curvs_flag] if len(angles) > 0 else angles
    elif tif_boundary == 3:  # TIFF boundary
        map_fibers = fiber_structure[in_curvs_flag]
        map_angles = angles[in_curvs_flag]

    # Create heatmap using draw_map
    raw_map, proc_map = draw_map(
        map_fibers, map_angles, img, boundary_measurement, map_params
    )

    # Create figure and save
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(img, cmap="gray")

    # Create colormap
    if boundary_measurement:
        # Green (aligned), yellow (moderate), red (perpendicular)
        tg = int(10 * 255 / 90)  # Green threshold
        ty = int(45 * 255 / 90)  # Yellow threshold
        tr = int(60 * 255 / 90)  # Red threshold
    else:
        tg = 32
        ty = 64
        tr = 128

    # Create custom colormap
    from matplotlib.colors import ListedColormap

    colors = np.zeros((256, 3))
    colors[tg:ty, 1] = 1.0  # Green
    colors[ty:tr, 0:2] = 1.0  # Yellow
    colors[tr:, 0] = 1.0  # Red
    cmap = ListedColormap(colors)

    # Overlay heatmap with transparency
    ax.imshow(proc_map, cmap=cmap, alpha=0.5)
    ax.axis("off")

    print("Saving map")

    # Save map
    save_map = os.path.join(output_directory, f"{img_name}_procmap.tiff")
    plt.tight_layout(pad=0)
    plt.savefig(save_map, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved heatmap to {save_map}")

    # Save values CSV
    save_values = os.path.join(output_directory, f"{img_name}_values.csv")
    if tif_boundary == 3:  # TIFF boundary
        values_data = np.column_stack([map_angles, distances[in_curvs_flag]])
    elif tif_boundary in [1, 2]:  # CSV boundary
        values_data = (
            np.column_stack([map_angles, distances[in_curvs_flag]])
            if len(distances) > 0
            else map_angles.reshape(-1, 1)
        )
    else:  # No boundary
        values_data = map_angles.reshape(-1, 1)

    np.savetxt(save_values, values_data, delimiter=",")
    print(f"Saved values to {save_values}")


def save_roi_tif_file(
    img, coordinates, fiber_structure, output_directory, img_name, fiber_len=5
):
    """
    Save ROI and fiber positions to a TIFF file.

    Parameters
    ----------
    img : ndarray
        2D image array to display as background.
    coordinates : dict
        Dictionary of ROI coordinates.
    fiber_structure : pd.DataFrame
        DataFrame containing fiber information with columns:
        center_row/center_1, center_col/center_2, angle.
    output_directory : str
        Directory where the TIFF file will be saved.
    img_name : str
        Name of the image (used for output filename).
    fiber_len : int, optional
        Length of fiber lines to draw (default 5).
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # Show all ROIs and fiber locations
    for roi_idx, roi_coords in enumerate(coordinates.values()):
        roi_coords_array = np.array(roi_coords)
        # Plot boundary (X=col, Y=row)
        ax.plot(
            roi_coords_array[:, 1],
            roi_coords_array[:, 0],
            "y-",
            linewidth=1,
        )

        # Add ROI label at center
        center_row = roi_coords_array[:, 0].mean()
        center_col = roi_coords_array[:, 1].mean()
        ax.text(
            center_col,
            center_row,
            str(roi_idx + 1),
            fontweight="bold",
            color="yellow",
            fontsize=7,
        )

    # Plot fiber positions and orientations
    for _, fiber_row in fiber_structure.iterrows():
        if "center_row" in fiber_structure.columns:
            center_row = fiber_row["center_row"]
            center_col = fiber_row["center_col"]
            angle = fiber_row["angle"]
        else:
            center_row = fiber_row["center_1"]
            center_col = fiber_row["center_2"]
            angle = fiber_row["angle"]

        # Calculate fiber line endpoints
        angle_rad = np.deg2rad(angle)
        start_x = center_col + fiber_len * np.cos(angle_rad)
        start_y = center_row - fiber_len * np.sin(angle_rad)
        end_x = center_col - fiber_len * np.cos(angle_rad)
        end_y = center_row + fiber_len * np.sin(angle_rad)

        ax.plot([start_x, end_x], [start_y, end_y], "g-", linewidth=0.5)

    # Save figure
    save_bf_figures = os.path.join(
        output_directory, f"{img_name}_BoundaryFiberPositions.tif"
    )
    plt.tight_layout(pad=0)
    plt.savefig(save_bf_figures, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def concat_roi_df(
    roi_measurements, roi_measurement_details, roi_summary_details, summary_row
):
    """
    Concatenate ROI measurements into accumulating dataframes.

    Parameters
    ----------
    roi_measurements : pd.DataFrame or None
        Detailed measurements for fibers in this ROI. Contains columns like:
        angle_to_boundary_edge, angle_to_boundary_center, etc.
        If None, only summary_row is appended with NaN statistics.
    roi_measurement_details : pd.DataFrame
        Accumulator for all detailed measurements across ROIs (modified in-place).
    roi_summary_details : pd.DataFrame
        Accumulator for summary statistics per ROI (modified in-place).
    summary_row : dict
        Pre-populated summary info for this ROI (name, center_row, center_col,
        orientation, area). Will be augmented with statistics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Updated (roi_measurement_details, roi_summary_details)
    """
    if roi_measurements is not None and not roi_measurements.empty:
        # Append detailed measurements
        if roi_measurement_details.empty:
            roi_measurement_details = roi_measurements.copy()
        else:
            roi_measurement_details = pd.concat(
                [roi_measurement_details, roi_measurements], ignore_index=True
            )

        # Calculate summary statistics from measurements
        summary_row["mean_of_angle_to_boundary_edge"] = roi_measurements[
            "angle_to_boundary_edge"
        ].mean()
        summary_row["mean_of_angle_to_boundary_center"] = roi_measurements[
            "angle_to_boundary_center"
        ].mean()
        summary_row["mean_of_angle_to_center_line"] = roi_measurements[
            "angle_to_center_line"
        ].mean()
        summary_row["number_of_fibers"] = len(roi_measurements)
    else:
        # No measurements for this ROI
        summary_row["mean_of_angle_to_boundary_edge"] = np.nan
        summary_row["mean_of_angle_to_boundary_center"] = np.nan
        summary_row["mean_of_angle_to_center_line"] = np.nan
        summary_row["number_of_fibers"] = 0

    # Append summary row
    if roi_summary_details.empty:
        roi_summary_details = pd.DataFrame([summary_row])
    else:
        roi_summary_details = pd.concat(
            [roi_summary_details, pd.DataFrame([summary_row])], ignore_index=True
        )

    return roi_measurement_details, roi_summary_details


if __name__ == "__main__":
    import csv
    import os
    import matplotlib.pyplot as plt

    base_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "test_results",
        "get_tif_boundary_test_files",
    )

    files = [
        os.path.join(base_path, f)
        for f in [
            "real1_boundary1_coords.csv",
            "real1_boundary2_coords.csv",
            "real1_boundary3_coords.csv",
        ]
    ]
    coords = {}

    for i, f in enumerate(files, start=1):
        with open(f, newline="") as csvfile:
            reader = csv.reader(csvfile)
            # convert each row into a tuple, cast to float or int if needed
            coords[f"csv{i}"] = [tuple(map(float, row)) for row in reader]

    img = plt.imread(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "tests", "test_images", "real1.tif"
        ),
        format="TIF",
    )

    dist_thresh = 100
    min_dist = []
    obj = {}

    # Load curvelet data as DataFrame
    obj = pd.read_csv(os.path.join(base_path, "real1_curvelets.csv"))
    # Convert from MATLAB 1-based to Python 0-based indexing
    obj["center_1"] = obj["center_1"] - 1
    obj["center_2"] = obj["center_2"] - 1

    boundary_img = np.loadtxt(
        os.path.join(base_path, "real1_boundary_img.csv"), delimiter=","
    )

    # Create parameter objects
    image_params = ImageInputParameters(
        img=img,
        img_name="bob",
        slice_num=1,
        num_sections=1,
    )
    
    fiber_params = FiberAnalysisParameters(
        fiber_mode=0,
        keep=0.05,
        fire_directory=os.getcwd(),
    )
    
    output_params = OutputControlParameters(
        output_directory=".",
        make_associations=True,
        make_map=True,
        make_overlay=True,
        make_feature_file=True,
    )
    
    boundary_params = BoundaryParameters(
        coordinates=coords,
        distance_threshold=dist_thresh,
        tif_boundary=3,
        boundary_img=boundary_img,
    )
    
    advanced_options = AdvancedAnalysisOptions(
        exclude_fibers_in_mask_flag=1,
        curvelets_group_radius=10,
        selected_scale=1,
        heatmap_STD_filter_size=16,
        heatmap_SQUARE_max_filter_size=12,
        heatmap_GAUSSIAN_disc_filter_sigma=4,
        minimum_nearest_fibers=2,
        minimum_box_size=32,
        fiber_midpoint_estimate=1,
        min_dist=[],
    )
    
    process_image(
        image_params=image_params,
        fiber_params=fiber_params,
        output_params=output_params,
        boundary_params=boundary_params,
        advanced_options=advanced_options,
    )
