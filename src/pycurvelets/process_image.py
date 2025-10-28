from curvelops import FDCT2D, curveshow, fdct2d_wrapper
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage.draw import polygon, polygon2mask
from skimage.measure import regionprops, label
import tkinter as tk
import time

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

    start_time = time.perf_counter()
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
        (
            fiber_structure,
            density_df,
            alignment_df,
        ) = get_ct(img, curve_cp, feature_cp)
    else:
        print("Reading CT-FIRE database.")
        # Call getFIRE
        # Add slice name used in CT-FIRE output
        fiber_structure, density_df, alignment_df = get_fire(
            img_name_plain, fire_directory, fiber_mode, feature_cp
        )

    if fiber_structure.empty:
        return None

    # Get features correlating fibers to boundaries
    if boundary_measurement:
        print("Analyzing boundary.")
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

            format_df_to_excel(
                roi_summary_details,
                save_boundary_width_measurements,
                sheet_name="Boundary Summary",
            )

            for roi_index, roi_coords in enumerate(coordinates.values()):
                roi_list = ROIList(
                    coordinates=roi_coords,
                    image_width=img.shape[1],
                    image_height=img.shape[0],
                )
                roi_coords = np.array(roi_coords)
                roi_mask = polygon2mask(
                    (roi_list.image_width, roi_list.image_height), roi_coords[:, [1, 0]]
                )
                roi_regions = regionprops(label(roi_mask.astype(int)))

                if len(roi_regions) != 1:
                    raise ValueError(
                        f"ROI {roi_index} does not correspond to a single region"
                    )

                roi_properties = roi_regions[0]
                orientation_degrees = -np.degrees(roi_properties.orientation)
                if orientation_degrees < 0:
                    orientation_degrees += 180

                summary_row = {
                    "name": roi_properties.label,
                    "center_row": roi_properties.centroid[0],
                    "center_col": roi_properties.centroid[1],
                    "orientation": orientation_degrees,
                    "area": roi_properties.area,
                }

                roi_measurements = None
                try:
                    roi_measurements = get_alignment_to_roi(
                        roi_list, fiber_structure, distance_threshold
                    )
                except Exception as e:
                    print(f"Boundary {roi_index} was skipped. Error: {str(e)}")

                # concat_roi_df(
                #     roi_measurements, roi_measurement_details, roi_summary_details
                # )

    return True


# def concat_roi_df(roi_measurements, roi_measurement_details, roi_summary_details):
#     if roi_measurements is not None:
#         roi_measurement_details = pd.concat(
#             [
#                 roi_measurement_details,
#                 pd.DataFrame(
#                     {
#                         "angle_to_boundary_edge": [roi_measurements.angle2boundaryEdge],
#                         "angle_to_boundary_center": [
#                             roi_measurements.angle2boundaryCenter
#                         ],
#                         "angle_to_center_line": [roi_measurements.angle2centersLine],
#                         "fiber_center_row": [roi_measurements.fibercenterList[:, 0]],
#                         "fiber_center_col": [roi_measurements.fibercenterList[:, 1]],
#                         "fiber_angle_list": [roi_measurements.fiberangleList],
#                         "distance_list": [roi_measurements.distanceList],
#                         "boundary_point_row": [roi_measurements.boundaryPoints[:, 0]],
#                         "boundary_point_col": [roi_measurements.boundaryPoints[:, 1]],
#                     }
#                 ),
#             ],
#             ignore_index=True,
#         )
#
#         roi_summary_details = pd.concat(
#             [
#                 roi_summary_details,
#                 pd.DataFrame(
#                     {
#                         "name": [i],
#                         "center_row": [center_row],
#                         "center_col": [center_col],
#                         "orientation": [orientation_deg],
#                         "area": [area],
#                         "mean_of_angle_to_boundary_edge": [
#                             np.nanmean(roi_measurements.angle2boundaryEdge)
#                         ],
#                         "mean_of_angle_to_boundary_center": [
#                             np.nanmean(roi_measurements.angle2boundaryCenter)
#                         ],
#                         "mean_of_angle_to_center_line": [
#                             np.nanmean(roi_measurements.angle2centersLine)
#                         ],
#                         "number_of_fibers": [n_fibers],
#                     }
#                 ),
#             ],
#             ignore_index=True,
#         )


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

    with open(os.path.join(base_path, "real1_curvelets.csv"), newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            obj[i] = {
                "center": (float(row["center_1"]) - 1, float(row["center_2"]) - 1),
                "angle": float(row["angle"]),
                "weight": float(row["weight"]),
            }

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
