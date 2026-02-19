import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from pycurvelets.utils.segmentation import get_segment_pixels
from pycurvelets.utils.math import circ_r, find_outline_slope


def get_tif_boundary(coordinates, img, obj, dist_thresh, min_dist):
    """
    get_tif_boundary - Associates boundary coordinates with curvelets/fibers and
    computes relative angle measures.

    Parameters
    ----------
    coordinates : ndarray or dict
        Array of coordinates for boundary endpoints of line segments, or dict of coordinate arrays.
    img : ndarray
        The image being measured.
    obj : pd.DataFrame
        DataFrame containing curvelet/fiber properties.
        Should have columns: center_row/center_1, center_col/center_2, angle.
    dist_thresh : float
        Pixel distance from boundary within which to evaluate curvelets.
    min_dist : float
        Minimum distance to boundary (to exclude fibers on/near boundary).

    Returns
    -------
    res_mat : ndarray
        Result matrix with fiber-boundary relationships.
    res_mat_names : list
        Column names for res_mat.
    num_im_pts : int
        Number of image points.
    """

    img_height, img_width = img.shape[:2]
    img_size = (img_height, img_width)

    # Extract center points from DataFrame
    if "center_row" in obj.columns:
        all_center_points = obj[["center_row", "center_col"]].values
    else:
        all_center_points = obj[["center_1", "center_2"]].values
    all_center_points = np.round(all_center_points).astype(int)

    # collect all boundary points
    coords = np.vstack([coordinates[k] for k in coordinates])

    neighbors = NearestNeighbors(
        n_neighbors=1, algorithm="brute", metric="euclidean"
    ).fit(coords)
    distances, indices = neighbors.kneighbors(all_center_points)
    idx_dist = indices.flatten()
    dist = distances.flatten()

    # Get pixel values at fiber center locations from the boundary image
    # Round coordinates to nearest integer (MATLAB sub2ind behavior)
    center_rows = np.clip(
        np.round(all_center_points[:, 0]).astype(int), 0, img_height - 1
    )
    center_cols = np.clip(
        np.round(all_center_points[:, 1]).astype(int), 0, img_width - 1
    )
    reg_dist = img[center_rows, center_cols]

    step_size = img_width // 20

    linear_indices = np.arange(0, img_height * img_width, step_size)

    # Convert linear indices to (row, col)
    cols, rows = np.unravel_index(linear_indices, (img_width, img_height))

    all_img_points = np.column_stack((rows, cols))

    sorted_coords = coords[
        np.lexsort((coords[:, 0], coords[:, 1]))
    ]  # row first, then col
    subsampled_boundary_points = sorted_coords[::3, :]

    # KNN search
    # Use brute-force search with explicit metric for deterministic behavior

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean").fit(
        subsampled_boundary_points
    )

    distances_to_boundary, _ = nbrs.kneighbors(all_img_points)
    distances_to_boundary = distances_to_boundary.flatten()  # shape (N_points,)

    # Apply distance thresholds
    if not min_dist:
        in_img_points_mask = distances_to_boundary <= (dist_thresh + 1e-12)
    else:
        in_img_points_mask = (distances_to_boundary <= dist_thresh) & (
            distances_to_boundary > min_dist
        )

    # Points that satisfy threshold
    in_points = all_img_points[in_img_points_mask]
    num_img_points = len(in_points) * step_size

    # Process all curvs, at this point
    curvs_len = len(obj)
    nearest_boundary_dist = np.full(curvs_len, np.nan)
    nearest_region_dist = np.full(curvs_len, np.nan)
    nearest_boundary_relative_angle = np.full(curvs_len, np.nan)
    extension_point_dist = np.full(curvs_len, np.nan)
    extension_point_relative_angle = np.full(curvs_len, np.nan)
    measurement_boundary = np.full((curvs_len, 2), np.nan)

    in_curvs_flag = np.zeros(curvs_len, dtype=bool)
    out_curvs_flag = np.zeros(curvs_len, dtype=bool)

    if not min_dist:
        for i in range(curvs_len):
            # in region?
            nearest_region_dist[i] = int((reg_dist[i] == 255) | (reg_dist[i] == 1))

            # distance in nearest epithelial boundary
            nearest_boundary_dist[i] = dist[i]

            # relative angle at nearest boundary point
            if dist[i] <= dist_thresh:
                nearest_boundary_relative_angle[i], boundary_point = get_relative_angle(
                    coordinates=coords,
                    idx=idx_dist[i],
                    fiber_angle=obj.iloc[i]["angle"],
                    img_height=img_height,
                    img_width=img_width,
                )
            else:
                # Fiber is too far from boundary
                nearest_boundary_relative_angle[i] = np.nan
                boundary_point = np.full(2, np.nan)

            # Extension point features
            # Convert DataFrame row to dict for get_points_on_line
            fiber_dict = {
                "center": (
                    (
                        obj.iloc[i]["center_1"]
                        if "center_1" in obj.columns
                        else obj.iloc[i]["center_row"]
                    ),
                    (
                        obj.iloc[i]["center_2"]
                        if "center_2" in obj.columns
                        else obj.iloc[i]["center_col"]
                    ),
                ),
                "angle": obj.iloc[i]["angle"],
            }
            line_curvelets, ortho_curvelets = get_points_on_line(
                fiber_dict, dist_thresh
            )
            line_curvelets = np.array(line_curvelets)

            a = line_curvelets
            b = coords
            a_view = a.view([("", a.dtype)] * a.shape[1])
            b_view = b.view([("", b.dtype)] * b.shape[1])
            intersection_line, intersection_line_a, intersection_line_b = (
                np.intersect1d(a_view, b_view, return_indices=True)
            )
            intersection_line = a[intersection_line_a]

            if intersection_line.size != 0:
                # get closest distance from curvelet center to the intersection
                # (get rid of farther ones)
                # Use brute-force search for determinism when selecting intersection points
                nbrs = NearestNeighbors(
                    n_neighbors=1, algorithm="brute", metric="euclidean"
                ).fit(intersection_line)
                fiber_center = (
                    (
                        obj.iloc[i]["center_1"]
                        if "center_1" in obj.columns
                        else obj.iloc[i]["center_row"]
                    ),
                    (
                        obj.iloc[i]["center_2"]
                        if "center_2" in obj.columns
                        else obj.iloc[i]["center_col"]
                    ),
                )
                line_distance, idx_line_distance = nbrs.kneighbors([fiber_center])

                idx_line_distance = idx_line_distance[0][0]
                line_distance = line_distance[0][0]

            else:
                extension_point_dist[i] = np.nan  # no intersection exists
                extension_point_relative_angle[i] = np.nan  # no angle exists

            measurement_boundary[i] = boundary_point

    result_mat = np.column_stack(
        [
            nearest_boundary_dist,
            nearest_region_dist,
            nearest_boundary_relative_angle,
            extension_point_dist,
            extension_point_relative_angle,
            measurement_boundary,
        ]
    )

    result_mat_names = [
        "nearest_boundary_distance",
        "nearest_region_distance",
        "nearest_boundary_angle",
        "extension_point_distance",
        "extension_point_angle",
        "boundary_point_row",
        "boundary_point_col",
    ]

    result_df = pd.DataFrame(result_mat, columns=result_mat_names)

    return result_mat, result_mat_names, num_img_points, result_df


def get_relative_angle(coordinates, idx, fiber_angle, img_height, img_width):
    coordinates = np.asarray(coordinates)
    idx = int(idx)

    boundary_angle = find_outline_slope(coordinates, idx)
    boundary_point = coordinates[idx, :]
    if (
        boundary_point[0] == 0
        or boundary_point[1] == 0
        or boundary_point[0] == img_height - 1
        or boundary_point[1] == img_width - 1
    ):
        # don't count fiber if boundary point is along edge of image
        temp_angle = 0
    else:
        # compute relative angle
        # there is a 90 degree phase shift in fiber_angle and boundary_angle due
        # to image orientation issues in MATLAB. Therefore, no need to invert
        # (ie. 1-X) here
        temp_angle = circ_r(
            [np.radians(2 * fiber_angle), np.radians(2 * boundary_angle)]
        )
        temp_angle = np.degrees(np.arcsin(temp_angle))

    return temp_angle, boundary_point


def get_points_on_line(object, box_size):
    center = object["center"]
    angle = object["angle"]
    slope = -np.tan(np.deg2rad(angle))

    if np.isinf(slope):
        dist_y = 0
        dist_x = box_size
    else:
        dist_y = box_size / np.sqrt(1 + slope * slope)
        dist_x = dist_y * slope

    point_1 = [center[0] - dist_x, center[1] - dist_y]
    point_2 = [center[0] + dist_x, center[1] + dist_y]

    line_curv, _ = get_segment_pixels(point_1, point_2)
    ortho_curv = []

    return line_curv, ortho_curv


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

    print(len(coords))
    img = plt.imread(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "tests", "test_images", "real1.tif"
        ),
        format="TIF",
    )

    print(img)

    dist_thresh = 100
    min_dist = []
    obj = {}

    with open(os.path.join(base_path, "real1_curvelets.csv"), newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            obj[i] = {
                "center": (float(row["center_1"]) - 1, float(row["center_2"]) - 1),
                "angle": float(row["angle"]),
                "weight": float(row["weight"]),
            }

    get_tif_boundary(coords, img, obj, dist_thresh, min_dist)
