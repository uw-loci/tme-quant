"""
ROI Alignment Analysis Module

This module provides functionality to calculate fiber alignment measurements relative to
regions of interest (ROIs). It converts the MATLAB getAlignment2ROI.m function to Python
with enhanced data structures and modern Python conventions.

Author: Laboratory for Optical and Computational Instrumentation, UW-Madison
Python conversion: 2025
"""

import logging
from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree

from .models.models import ROIList
from .get_relative_angles import get_relative_angles

# Configure logging
logger = logging.getLogger(__name__)


def get_alignment_to_roi(
    roi_list: ROIList,
    fiber_structure: pd.DataFrame,
    distance_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate fiber alignment measurements relative to regions of interest (ROIs).

    This function analyzes the alignment of fibers with respect to ROI boundaries,
    computing various angle measurements and distance relationships. It supports both
    distance-based fiber selection and pre-selected fiber analysis modes.

    Parameters
    ----------
    roi_list : ROIList
        Single ROI or list of ROIs. Each ROI should be a ROIList dataclass containing:
        - 'coordinates' : tuple
            ROI boundary coordinates as (y, x) pairs
        - 'image_width' : int
            Image width in pixels
        - 'image_height' : int
            Image height in pixels
        For pre-selected fiber mode, additional attributes may be present:
        - 'dist' : array_like, optional
            Pre-calculated distances (used when distance_threshold is None)
        - 'index2object' : array_like, optional
            Pre-selected fiber indices (used when distance_threshold is None)

    fiber_structure : list of list of Fiber or list of Fiber
        Fiber data containing center coordinates and angles. Each Fiber object has:
        - center_row : float
            Fiber center row coordinate (y)
        - center_col : float
            Fiber center column coordinate (x)
        - angle : float
            Fiber orientation angle in degrees

    distance_threshold : float, optional
        Maximum distance from fiber to ROI boundary for inclusion in analysis.
        If None, uses pre-selected fibers from roi_list['index2object'].

    Returns
    -------
    pd.DataFrame
        DataFrame containing fiber alignment measurements with columns:
        - angle_to_boundary_edge : float
            Angles between fiber orientation and boundary edge at closest point
        - angle_to_boundary_center : float
            Angles between fiber orientation and ROI center orientation
        - angle_to_center_line : float
            Angles between fiber orientation and fiber-to-ROI-center line
        - fiber_center_row : float
            Fiber center row coordinate (y)
        - fiber_center_col : float
            Fiber center column coordinate (x)
        - fiber_angle_list : float
            Fiber orientation angles in degrees
        - distance_list : float
            Distances from fibers to closest ROI boundary points
        - boundary_point_row : float
            Row coordinate of the nearest boundary point
        - boundary_point_col : float
            Column coordinate of the nearest boundary point
    fiber_count: int
      Number of fibers
    """

    # Input validation
    if roi_list is None:
        raise ValueError("roi_list cannot be None")

    if fiber_structure is None or len(fiber_structure) == 0:
        raise ValueError("fiber_structure cannot be None or empty")

    # Convert DataFrame to list of fiber dicts
    # Handle both column naming conventions: center_row/center_col or center_1/center_2
    fiber_list = []
    for _, row in fiber_structure.iterrows():
        if "center_row" in fiber_structure.columns:
            fiber_dict = {
                "center_row": row["center_row"],
                "center_col": row["center_col"],
                "angle": row["angle"],
            }
        else:
            # Fallback to center_1/center_2 naming
            fiber_dict = {
                "center_row": row["center_1"],
                "center_col": row["center_2"],
                "angle": row["angle"],
            }
        fiber_list.append(fiber_dict)
    fiber_structure = fiber_list

    # Determine processing mode
    select_fiber_flag = distance_threshold is not None

    if select_fiber_flag:
        logger.info(
            f"Using distance-based selection with threshold: {distance_threshold}"
        )
    else:
        logger.info("Using pre-selected fiber mode")

    # Collect all measurements across all ROIs
    all_measurements = []

    # Process each ROI
    n_rois = len(roi_list.coordinates)

    for roi_idx in range(n_rois):
        roi_coords = np.array(roi_list.coordinates[roi_idx])

        # Get fiber list for this ROI
        if isinstance(fiber_structure[0], list):
            # List of lists - one list per ROI
            current_fiber_list = fiber_structure[roi_idx]
        else:
            # Single list of fibers - use for all ROIs
            current_fiber_list = fiber_structure

        if select_fiber_flag:
            # Distance-based fiber selection
            roi_tree = KDTree(roi_coords)
            fiber_centers = np.array(
                [[f["center_row"], f["center_col"]] for f in current_fiber_list]
            )
            dist, idx_dist = roi_tree.query(fiber_centers)
            fiber_indices = np.where(dist <= distance_threshold)[0]
        else:
            # Pre-selected fiber mode
            fiber_indices = np.arange(len(current_fiber_list))
            # Note: distance would need to be pre-calculated and stored in ROI
            dist = None
            idx_dist = None

        n_fibers = len(fiber_indices)

        if n_fibers == 0:
            if select_fiber_flag:
                logger.info(
                    f"ROI {roi_idx}: NO fiber is within the specified distance "
                    f"({distance_threshold}) to the boundary"
                )
            else:
                logger.info(f"ROI {roi_idx}: NO fiber is selected")
            continue

        if select_fiber_flag:
            if n_fibers == 1:
                logger.info(
                    f"ROI {roi_idx}: Found one fiber within the {distance_threshold} "
                    f"distance to the boundary"
                )
            else:
                logger.info(
                    f"ROI {roi_idx}: Found {n_fibers} fibers within the "
                    f"{distance_threshold} distance to the boundary"
                )
        else:
            logger.info(
                f"Calculate all the relative alignment of pre-selected fibers. "
                f"Total fiber number is {n_fibers}"
            )

        # VECTORIZED: Compute ROI properties ONCE (not per fiber!)
        logger.info(f"Processing {n_fibers} fibers (vectorized)")

        from skimage.draw import polygon2mask
        from skimage.measure import regionprops, label

        mask = polygon2mask((roi_list.image_height, roi_list.image_width), roi_coords)
        labeled = label(mask.astype(np.uint8))
        props = regionprops(labeled)

        if len(props) != 1:
            raise ValueError(f"ROI {roi_idx} does not define a single region")

        prop = props[0]
        boundary_center = np.array(prop.centroid)[::-1]  # to [x, y]
        roi_angle = -90 + np.degrees(prop.orientation)
        if roi_angle < 0:
            roi_angle = 180 + roi_angle

        # Extract fiber data as arrays for vectorization
        fiber_indices_list = fiber_indices.tolist()
        fiber_centers = np.array(
            [
                [
                    current_fiber_list[i]["center_row"],
                    current_fiber_list[i]["center_col"],
                ]
                for i in fiber_indices_list
            ]
        )
        fiber_angles = np.array(
            [current_fiber_list[i]["angle"] for i in fiber_indices_list]
        )

        # VECTORIZED: Angle to boundary center
        angle_diffs = np.abs(fiber_angles - roi_angle)
        angle_diffs = np.where(angle_diffs > 90, 180 - angle_diffs, angle_diffs)

        # VECTORIZED: Angle to centers line
        dx = fiber_centers[:, 1] - boundary_center[1]  # col difference
        dy = fiber_centers[:, 0] - boundary_center[0]  # row difference
        centers_line_angles = np.degrees(np.arctan2(dx, dy))
        centers_line_angles = np.where(
            centers_line_angles < 0,
            np.abs(centers_line_angles),
            180 - centers_line_angles,
        )
        angle_to_centers = np.abs(centers_line_angles - fiber_angles)
        angle_to_centers = np.where(
            angle_to_centers > 90, 180 - angle_to_centers, angle_to_centers
        )

        # Angle to boundary edge (still per-fiber due to different boundary points)
        from pycurvelets.utils.math import find_outline_slope, circ_r

        angles_to_edge = []
        boundary_points = []  # Store boundary point coordinates
        for fiber_i in range(n_fibers):
            i = fiber_indices[fiber_i]
            if select_fiber_flag:
                boundary_idx = int(
                    idx_dist[i].item()
                    if hasattr(idx_dist[i], "item")
                    else idx_dist[i][0]
                )
            else:
                boundary_idx = 0

            boundary_pt = roi_coords[boundary_idx]
            boundary_points.append(boundary_pt)  # Store the boundary point
            boundary_angle = find_outline_slope(roi_coords, boundary_idx)

            if (
                any(boundary_pt == 1)
                or boundary_pt[0] == roi_list.image_height
                or boundary_pt[1] == roi_list.image_width
            ):
                temp_ang = 0
            else:
                if boundary_angle is None:
                    temp_ang = None
                else:
                    temp_ang = circ_r(
                        [
                            np.radians(2 * fiber_angles[fiber_i]),
                            np.radians(2 * boundary_angle),
                        ]
                    )
                    temp_ang = np.degrees(np.arcsin(temp_ang))
            angles_to_edge.append(temp_ang)

        # Store all measurements
        for fiber_i in range(n_fibers):
            i = fiber_indices[fiber_i]
            boundary_pt = boundary_points[fiber_i]
            measurement = {
                "angle_to_boundary_edge": angles_to_edge[fiber_i],
                "angle_to_boundary_center": angle_diffs[fiber_i],
                "angle_to_center_line": angle_to_centers[fiber_i],
                "fiber_center_row": fiber_centers[fiber_i, 0],
                "fiber_center_col": fiber_centers[fiber_i, 1],
                "fiber_angle": fiber_angles[fiber_i],
                "distance": (
                    dist[i][0]
                    if select_fiber_flag and hasattr(dist[i], "__getitem__")
                    else (dist[i] if select_fiber_flag else None)
                ),
                "boundary_point_row": boundary_pt[0],
                "boundary_point_col": boundary_pt[1],
            }
            all_measurements.append(measurement)

    # Convert to DataFrame
    result_df = pd.DataFrame(all_measurements)

    return result_df, result_df.shape[0]
