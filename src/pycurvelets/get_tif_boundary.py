from sklearn.neighbors import NearestNeighbors
import numpy as np
from pycurvelets.helper_methods import find_outline_slope, circ_r


def get_tif_boundary(coordinates, img, obj, img_name, dist_thresh, min_dist):
    """
    get_tif_boundary - Associates boundary coordinates with curvelets/fibers and
    computes relative angle measures.

    Parameters
    ----------
    coordinates : ndarray
        Array of coordinates for boundary endpoints of line segments.
    img : ndarray
        The image being measured.
    obj : dict
        Dictionary (from new_curv) containing curvelet/fiber properties such as center and angle.
    img_name : str
        Name of the image file.
    dist_thresh : float
        Pixel distance from boundary within which to evaluate curvelets.
    min_dist : float
        Minimum distance to boundary (to exclude fibers on/near boundary).

    Notes
    -----
    This function does not return anything.
    """

    img_height, img_width = img.shape[:2]
    img_size = (img_height, img_width)

    # collect all fiber points
    all_center_points = np.vstack([o["center"] for o in obj])

    # collect all boundary points
    coords = np.vstack(coordinates)

    # collect all region points
    lin_idx = np.ravel_multi_index(
        (all_center_points[:, 0], all_center_points[:, 1]), dims=img_size
    )

    neighbors = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(coords)
    distances, indices = neighbors.kneighbors(all_center_points)
    idx_dist = indices.flatten()
    dist = distances.flatten()

    reg_dist = img.flat[lin_idx]

    step_size = img_width // 20
    linear_indices = np.arange(0, img_height * img_width, step_size)

    # Convert linear indices to (row, col)
    rows, cols = np.unravel_index(linear_indices, (img_height, img_width))
    all_img_points = np.column_stack((rows, cols))

    # Subsample boundary points for speed
    subsampled_boundary_points = coords[::3, :]

    # KNN search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        subsampled_boundary_points
    )
    distances_to_boundary, _ = nbrs.kneighbors(coords)
    distances_to_boundary = distances_to_boundary.flatten()  # shape (N_points,)

    # Apply distance thresholds
    if min_dist is None:
        in_img_points_mask = distances_to_boundary <= dist_thresh
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

    if min_dist is None:
        for i in range(len(curvs_len)):
            # in region?
            nearest_region_dist[i] = (reg_dist[i] == 255) | (reg_dist[i] == 1)

            # distance in nearest epithelial boundary
            nearest_boundary_dist[i] = dist[i]

            # relative angle at nearest boundary point
            if dist[i] <= dist_thresh:
                nearest_boundary_relative_angle[i], boundary_point = get_relative_angle(
                    coordinates=[coords[:, 1], coords[:, 0]],
                    idx=idx_dist[i],
                    fiber_angle=obj[i]["angle"],
                    img_height=img_height,
                    img_width=img_width,
                )
            else:
                nearest_boundary_relative_angle[i] = np.nan
                boundary_point = np.full(2, np.nan)

            # Extension point features

    return None


def get_relative_angle(coordinates, idx, fiber_angle, img_height, img_width):
    boundary_angle = find_outline_slope([coordinates[:, 1], coordinates[:, 0]], idx)
    boundary_point = coordinates[idx, :]
    if (
        boundary_point[0] == 1
        or boundary_point[1] == 1
        or boundary_point[0] == img_height
        or boundary_point[1] == img_width
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
