from functools import partial
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage.draw import polygon, polygon2mask
from skimage.measure import regionprops, label
import tkinter as tk
import time
from multiprocessing import Pool

from curvelops import FDCT2D, curveshow, fdct2d_wrapper

from pycurvelets.get_alignment_to_roi import get_alignment_to_roi
from pycurvelets.models import (
    CurveletControlParameters,
    FeatureControlParameters,
    ROIList,
)
from pycurvelets.get_ct import get_ct
from pycurvelets.get_fire import get_fire
from pycurvelets.get_tif_boundary import get_tif_boundary
from pycurvelets.utils.misc import format_df_to_excel


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
    make_associations : int
        Whether to generate association plots between boundary and fibers.
    make_map : int
        Whether to generate heatmap.
    make_overlay : int
        Whether to generate curvelet overlay figure.
    make_feature_file : int
        Whether to generate feature file.
    slice_num : int
        Slice number within a stack (if analyzing multiple slices).
    tif_boundary : int
        Flag indicating whether the boundary input is a TIFF file (True) or a list of coordinates (False).
    boundary_img : ndarray
        Boundary image used for overlay outputs.
    fire_directory : str
        Directory containing FIRE fiber results (optional; used instead of curvelets).
    fiber_mode : int
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
    boundary_measurement = bool(coordinates)

    # Feature control structure initialized: feature_cp
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=advanced_options["minimum_nearest_fibers"],
        minimum_box_size=advanced_options["minimum_box_size"],
        fiber_midpoint_estimate=advanced_options["fiber_midpoint_estimate"],
    )

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

    # Add lower limit for distance threshold
    min_dist = advanced_options["min_dist"]

    # Get features that are only based on fibers
    t_fiber_start = time.perf_counter()
    if fiber_mode == 0:
        print("Computing curvelet transform.")
        curve_cp = CurveletControlParameters(
            keep=keep,
            scale=advanced_options["selected_scale"],
            radius=advanced_options["curvelets_group_radius"],
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
    meas_bndry = None
    angles = None
    in_curvs_flag = None
    bins = np.arange(2.5, 180, 5)  # Default bins for histogram

    # Get features correlating fibers to boundaries
    if boundary_measurement:
        print("Analyzing boundary.")
        t_boundary_start = time.perf_counter()

        if tif_boundary == 3:
            print(
                "Calculate relative angles of fibers within the specified distance to each boundary, including"
            )
            print(" 1) angle to the nearest point on the boundary")
            print(" 2) angle to the boundary mask orientation")
            print(
                " 3) angle to the line connecting the fiber center and boundary mask center"
            )

            roi_number = len(coordinates)
            fiber_list = fiber_structure

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
            t_excel_start = time.perf_counter()

            # Aggregate data for overlay visualization
            all_roi_measurements = []

            for roi_index, roi_measurements, summary_row in results:
                roi_measurement_details, roi_summary_details = concat_roi_df(
                    roi_measurements,
                    roi_measurement_details,
                    roi_summary_details,
                    summary_row,
                )

                # Collect measurements for overlay
                if roi_measurements is not None and not roi_measurements.empty:
                    all_roi_measurements.append(roi_measurements)

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

            # Call get_tif_boundary to get global angles, distances, and boundary points
            # This is needed for the overlay visualization
            # print("Calling get_tif_boundary for global fiber-boundary relationships...")
            # res_mat, res_mat_names, num_im_pts = get_tif_boundary(
            #     coordinates=coordinates,
            #     img=img,
            #     obj=fiber_structure,
            #     dist_thresh=distance_threshold,
            #     min_dist=min_dist,
            # )

            #     # Extract data for overlay
            #     angles = res_mat[:, 2]  # nearest relative boundary angle
            #     distances = res_mat[:, 0]  # nearest boundary distance

            #     # Determine which fibers are within threshold
            #     if min_dist is None or min_dist == 0:
            #         in_curvs_flag = res_mat[:, 0] <= distance_threshold
            #     else:
            #         in_curvs_flag = (res_mat[:, 0] <= distance_threshold) & (
            #             res_mat[:, 0] > min_dist
            #         )

            #     out_curvs_flag = ~in_curvs_flag

            #     # Apply mask exclusion if enabled
            #     if exclude_fibers_in_mask_flag == 1:
            #         if min_dist is None or min_dist == 0:
            #             in_curvs_flag = (res_mat[:, 0] <= distance_threshold) & (
            #                 res_mat[:, 1] == 0
            #             )
            #         else:
            #             in_curvs_flag = (
            #                 (res_mat[:, 0] <= distance_threshold)
            #                 & (res_mat[:, 0] > min_dist)
            #                 & (res_mat[:, 1] == 0)
            #             )
            #         out_curvs_flag = ~in_curvs_flag

            #     # Extract boundary points (columns 6:7 in MATLAB, 0-indexed: 5:7)
            #     meas_bndry = res_mat[:, 5:7]
            #     bins = np.arange(2.5, 90, 5)

            #     # Combine all ROI measurements for overlay
            #     if all_roi_measurements:
            #         combined_measurements = pd.concat(
            #             all_roi_measurements, ignore_index=True
            #         )

            #         # Create in_curvs_flag based on which fibers were measured
            #         in_curvs_flag = np.zeros(len(fiber_structure), dtype=bool)
            #         # Mark fibers that appear in measurements as "in"
            #         if (
            #             "fiber_center_y" in combined_measurements.columns
            #             and "fiber_center_x" in combined_measurements.columns
            #         ):
            #             for _, row in combined_measurements.iterrows():
            #                 # Find matching fiber in fiber_structure
            #                 if "center_row" in fiber_structure.columns:
            #                     mask = (
            #                         fiber_structure["center_row"] == row["fiber_center_y"]
            #                     ) & (fiber_structure["center_col"] == row["fiber_center_x"])
            #                 else:
            #                     mask = (
            #                         fiber_structure["center_1"] == row["fiber_center_y"]
            #                     ) & (fiber_structure["center_2"] == row["fiber_center_x"])
            #                 in_curvs_flag[mask] = True

            #         # Set angles and meas_bndry for overlay
            #         angles = fiber_structure["angle"].values

            #         # Create meas_bndry array (boundary points for each fiber)
            #         meas_bndry = np.full((len(fiber_structure), 2), np.nan)
            #         if "boundary_point_row" in combined_measurements.columns:
            #             for _, row in combined_measurements.iterrows():
            #                 if "center_row" in fiber_structure.columns:
            #                     mask = (
            #                         fiber_structure["center_row"] == row["fiber_center_y"]
            #                     ) & (fiber_structure["center_col"] == row["fiber_center_x"])
            #                 else:
            #                     mask = (
            #                         fiber_structure["center_1"] == row["fiber_center_y"]
            #                     ) & (fiber_structure["center_2"] == row["fiber_center_x"])
            #                 idx = np.where(mask)[0]
            #                 if len(idx) > 0:
            #                     meas_bndry[idx[0]] = [
            #                         row.get("boundary_point_col", np.nan),
            #                         row.get("boundary_point_row", np.nan),
            #                     ]

            #     t_excel_end = time.perf_counter()
            #     print(f"⏱️  Excel writing took: {t_excel_end - t_excel_start:.2f}s")

            #     # Save visualization after all ROIs processed
            #     t_viz_start = time.perf_counter()
            #     save_roi_tif_file(
            #         img=img,
            #         coordinates=coordinates,
            #         fiber_structure=fiber_structure,
            #         output_directory=output_directory,
            #         img_name=img_name,
            #     )
            #     t_viz_end = time.perf_counter()
            #     print(f"⏱️  Visualization took: {t_viz_end - t_viz_start:.2f}s")

            #     t_boundary_end = time.perf_counter()
            #     print(
            #         f"⏱️  Total boundary analysis took: {t_boundary_end - t_boundary_start:.2f}s"
            #     )

            #     # Extract distances and boundary measurements
            #     distances = res_mat[:, 0]  # nearest boundary distance
            #     meas_bndry = res_mat[:, 5:7]  # columns 6:7 in MATLAB (0-indexed: 5:7)

            # elif tif_boundary == 1 or tif_boundary == 2:
            # NEED TO IMPLEMENT getBoundary
            # # Coordinate boundary modes
            # print("Processing boundary with getBoundary method")

            # # Call get_tif_boundary (which handles both tif_boundary 1 and 2)
            # res_mat, res_mat_names, num_im_pts, result_df = get_tif_boundary(
            #     coordinates=coordinates,
            #     boundary_img=boundary_img,
            #     fiber_structure=fiber_structure,
            #     img_name=img_name,
            #     distance_threshold=distance_threshold,
            #     fiber_key=fiber_key,
            #     end_length_list=end_length_list,
            #     fiber_mode=fiber_mode - 1,
            #     min_dist=min_dist,
            # )

            # # Extract angles (column 3, 0-indexed = column 2)
            # angles = res_mat[:, 2]

            # # Determine which fibers are within threshold
            # if min_dist is None or min_dist == 0:
            #     in_curvs_flag = res_mat[:, 0] <= distance_threshold
            # else:
            #     in_curvs_flag = (res_mat[:, 0] <= distance_threshold) & (
            #         res_mat[:, 0] > min_dist
            #     )

            # out_curvs_flag = ~in_curvs_flag

            # # Apply mask exclusion if enabled
            # if exclude_fibers_in_mask_flag == 1:
            #     if min_dist is None or min_dist == 0:
            #         in_curvs_flag = (res_mat[:, 0] <= distance_threshold) & (
            #             res_mat[:, 1] == 0
            #         )
            #     else:
            #         in_curvs_flag = (
            #             (res_mat[:, 0] <= distance_threshold)
            #             & (res_mat[:, 0] > min_dist)
            #             & (res_mat[:, 1] == 0)
            #         )
            # out_curvs_flag = ~in_curvs_flag

        bins = np.arange(2.5, 90, 5)  # 2.5:5:87.5

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

        meas_bndry = 0
        num_im_pts = img.shape[0] * img.shape[1]  # count all pixels if no boundary
        bins = np.arange(2.5, 180, 5)  # 2.5:5:177.5

    # Save fiber features if make_feature_file is enabled
    if make_feature_file:
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
                total_length_list
                if total_length_list
                else [np.nan] * len(fiber_structure)
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
        if (
            boundary_measurement
            and isinstance(meas_bndry, np.ndarray)
            and meas_bndry.ndim == 2
        ):
            boundary_cols = [
                "nearest_distance_to_boundary",
                "inside_epicenter_region",
                "nearest_relative_boundary_angle",
                "extension_point_distance",
                "extension_point_angle",
                "boundary_point_row",
                "boundary_point_col",
            ]
            for i, col_name in enumerate(boundary_cols):
                if i < meas_bndry.shape[1]:
                    fib_feat_df[col_name] = meas_bndry[:, i]
                else:
                    fib_feat_df[col_name] = np.nan
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
            save_fib_feat = os.path.join(
                output_directory, f"{img_name}_FiberFeatures.xlsx"
            )

        format_df_to_excel(fib_feat_df, save_fib_feat, sheet_name="Fiber Features")
        print(f"Saved fiber features to {save_fib_feat}")

    # Create and save histogram of angles
    if boundary_measurement:
        if tif_boundary == 3:
            # Only for tiff boundary - use angles within threshold
            if angles is not None and in_curvs_flag is not None:
                values = angles[in_curvs_flag]
            elif angles is not None:
                values = angles
            else:
                values = fiber_structure["angle"].values
        else:
            values = angles if angles is not None else fiber_structure["angle"].values
    else:
        values = angles if angles is not None else fiber_structure["angle"].values

    # Create histogram
    n, xout = np.histogram(values, bins=bins)

    # Prepare histogram data (frequency and bin centers)
    hist_data = np.vstack([n, xout[:-1]])  # xout has one more element than n

    # Save histogram
    save_hist = os.path.join(output_directory, f"{img_name}_hist.csv")

    # Circularly shift by 1 (equivalent to MATLAB circshift)
    temp_hist = np.roll(hist_data, 1, axis=0)

    # Save transposed (rows become columns)
    np.savetxt(save_hist, temp_hist.T, delimiter=",")
    print(f"Saved histogram to {save_hist}")

    # Inverse curvelet transform (only for curvelet mode)
    if fiber_mode == 0:
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
            save_recon = os.path.join(
                output_directory, f"{img_name}_reconstructed.tiff"
            )
            # Append mode for multiple sections

            recon_img = Image.fromarray(recon.astype(np.float32))
            if os.path.exists(save_recon):
                recon_img.save(save_recon, append_images=[recon_img], save_all=True)
            else:
                recon_img.save(save_recon)
        else:
            save_recon = os.path.join(
                output_directory, f"{img_name}_reconstructed.tiff"
            )

            Image.fromarray(recon.astype(np.float32)).save(save_recon)

        print(f"Saved reconstructed image to {save_recon}")
    else:
        # Cannot do inverse-CT for FIRE mode
        recon = None

        # Create overlay figure if requested
        # if make_overlay:
        #     print("Plotting overlay")

        #     # Determine fiber line length based on mode
        #     if fiber_mode == 0:
        #         fiber_len = 4  # Curvelet mode
        #     elif fiber_mode == 1:
        #         fiber_len = 2.5  # CT-FIRE minimum segment length
        #     else:  # fiber_mode 2 or 3
        #         fiber_len = 10  # CT-FIRE minimum fiber length

        #     # Create figure
        #     fig, ax = plt.subplots(
        #         figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100
        #     )
        #     ax.imshow(img, cmap="gray")
        #     ax.axis("off")

        #     # Plot boundaries if present
        #     if boundary_measurement:
        #         if tif_boundary < 3:  # CSV boundary
        #             if coordinates:
        #                 coords_array = np.array(list(coordinates.values())[0])
        #                 ax.plot(coords_array[:, 1], coords_array[:, 0], "y-")
        #                 ax.plot(coords_array[:, 1], coords_array[:, 0], "*y", markersize=3)
        #         elif tif_boundary == 3:  # TIFF boundary
        #             for roi_coords in coordinates.values():
        #                 roi_coords_array = np.array(roi_coords)
        #                 ax.plot(
        #                     roi_coords_array[:, 1],
        #                     roi_coords_array[:, 0],
        #                     "y-",
        #                     linewidth=1,
        #                 )

        #     # Draw fibers using draw_curvs utility
        #     from pycurvelets.utils.visualization import draw_curvs

        #     marksize = 7
        #     linewidth = 1

        #     # Determine which fibers to draw
        #     if in_curvs_flag is not None and isinstance(in_curvs_flag, np.ndarray):
        #         in_fibers_mask = in_curvs_flag
        #         out_fibers_mask = ~in_curvs_flag
        #         print(
        #             f"Overlay: {np.sum(in_fibers_mask)} fibers IN, {np.sum(out_fibers_mask)} fibers OUT"
        #         )
        #     else:
        #         print("Overlay: in_curvs_flag not set, showing all fibers as IN")
        #         in_fibers_mask = np.ones(len(fiber_structure), dtype=bool)
        #         out_fibers_mask = np.zeros(len(fiber_structure), dtype=bool)

        #     # Get angles
        #     fiber_angles_data = (
        #         angles if angles is not None else fiber_structure["angle"].values
        #     )

        #     # Draw fibers that are used (color_flag=0 for green)
        #     if np.any(in_fibers_mask):
        #         draw_curvs(
        #             fiber_structure[in_fibers_mask],
        #             ax,
        #             fiber_len,
        #             color_flag=0,
        #             angles=fiber_angles_data[in_fibers_mask],
        #             mark_size=marksize,
        #             line_width=linewidth,
        #             boundary_measurement=boundary_measurement,
        #         )

        #     # Draw fibers that are not used (color_flag=1 for red)
        #     if np.any(out_fibers_mask):
        #         draw_curvs(
        #             fiber_structure[out_fibers_mask],
        #             ax,
        #             fiber_len,
        #             color_flag=1,
        #             angles=fiber_angles_data[out_fibers_mask],
        #             mark_size=marksize,
        #             line_width=linewidth,
        #             boundary_measurement=boundary_measurement,
        #         )

        # Draw associations if requested
        if boundary_measurement and make_associations and meas_bndry is not None:
            if isinstance(meas_bndry, np.ndarray) and meas_bndry.ndim == 2:
                fiber_centers = (
                    fiber_structure[["center_row", "center_col"]].values
                    if "center_row" in fiber_structure.columns
                    else fiber_structure[["center_1", "center_2"]].values
                )

                # Debug: check meas_bndry
                valid_count = 0
                for i, (center, bndry_pt) in enumerate(
                    zip(fiber_centers[in_fibers_mask], meas_bndry[in_fibers_mask])
                ):
                    if (
                        len(bndry_pt) >= 2
                        and not np.isnan(bndry_pt[0])
                        and not np.isnan(bndry_pt[1])
                    ):
                        ax.plot(
                            [center[1], bndry_pt[0]],
                            [center[0], bndry_pt[1]],
                            "b-",
                            linewidth=0.5,
                            alpha=0.5,
                        )
                        valid_count += 1
                        if i < 3:  # Debug first 3
                            print(
                                f"Association {i}: fiber ({center[1]:.1f}, {center[0]:.1f}) -> boundary ({bndry_pt[0]:.1f}, {bndry_pt[1]:.1f})"
                            )

                print(
                    f"Drew {valid_count} association lines out of {np.sum(in_fibers_mask)} IN fibers"
                )

        print("Saving overlay")

        # Save overlay
        if num_sections > 1:
            save_overlay = os.path.join(output_directory, f"{img_name}_overlay.tiff")
        else:
            save_overlay = os.path.join(output_directory, f"{img_name}_overlay.tiff")

        plt.tight_layout(pad=0)
        plt.savefig(save_overlay, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"Saved overlay to {save_overlay}")

    return True


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

    # Load curvelet data as DataFrame
    obj = pd.read_csv(os.path.join(base_path, "real1_curvelets.csv"))
    # Convert from MATLAB 1-based to Python 0-based indexing
    obj["center_1"] = obj["center_1"] - 1
    obj["center_2"] = obj["center_2"] - 1

    boundary_img = get_tif_boundary(coords, img, obj, dist_thresh, min_dist)
    boundary_mode = 3
    fiber_mode = 0
    i = 1
    keep = 0.05
    make_association_flag = 1
    make_map_flag = 1
    make_overlay_flag = 1
    num_sections = 1
    output_dir = "."
    path_name = os.getcwd()
    process_image(
        img,
        "bob",
        output_dir,
        keep,
        coords,
        distance_threshold=dist_thresh,
        make_associations=make_association_flag,
        make_map=make_map_flag,
        make_overlay=make_overlay_flag,
        make_feature_file=1,
        slice_num=1,
        tif_boundary=boundary_mode,
        boundary_img=boundary_img,
        fire_directory=path_name,
        fiber_mode=fiber_mode,
        advanced_options=advanced_options,
        num_sections=1,
    )
