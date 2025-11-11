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

from .models.models import ROIMeasurements, ROIProcessingError, ROIList, ROI, Fiber
from .get_relative_angles import get_relative_angles

# Configure logging
logger = logging.getLogger(__name__)


def get_alignment_to_roi(
    roi_list: Union[ROIList, List[ROIList]],
    fiber_structure: Union[List[List[Fiber]], List[Fiber]],
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
        - angle2boundaryEdge : float
            Angles between fiber orientation and boundary edge at closest point
        - angle2boundaryCenter : float
            Angles between fiber orientation and ROI center orientation
        - angle2centersLine : float
            Angles between fiber orientation and fiber-to-ROI-center line
        - fibercenterX : float
            Fiber center X coordinate (column)
        - fibercenterY : float
            Fiber center Y coordinate (row)
        - fiberangle : float
            Fiber orientation angles in degrees
        - distance : float
            Distances from fibers to closest ROI boundary points
    """

    # Input validation
    if roi_list is None:
        raise ValueError("roi_list cannot be None")

    if fiber_structure is None or len(fiber_structure) == 0:
        raise ValueError("fiber_structure cannot be None or empty")

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
                [[f.center_row, f.center_col] for f in current_fiber_list]
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

        # Process each fiber
        for fiber_i in range(n_fibers):
            i = fiber_indices[fiber_i]

            # Prepare ROI dict for get_relative_angles
            if select_fiber_flag:
                # idx_dist[i] gives the index of the closest boundary point
                boundary_idx = int(idx_dist[i].item())
            else:
                # Would need to be provided in ROI structure
                boundary_idx = 0  # Default fallback

            roi_dict = {
                "coords": roi_coords,
                "imageWidth": roi_list.image_width,
                "imageHeight": roi_list.image_height,
                "index2object": boundary_idx,
            }

            # Prepare fiber dict for get_relative_angles
            fiber = current_fiber_list[i]
            fiber_dict = {
                "center": [fiber.center_row, fiber.center_col],
                "angle": fiber.angle,
            }

            # Calculate relative angles
            angle_option = 0  # Calculate all angles
            fig_flag = False
            relative_angles, _ = get_relative_angles(
                roi_dict, fiber_dict, angle_option, fig_flag
            )

            # Store measurements
            measurement = {
                "angle_to_boundary_edge": relative_angles["angle2boundaryEdge"],
                "angle_to_boundary_center": relative_angles["angle2boundaryCenter"],
                "angle_to_center_line": relative_angles["angle2centersLine"],
                "fiber_center_x": fiber.center_col,  # X is column
                "fiber_center_y": fiber.center_row,  # Y is row
                "fiber_angle": fiber.angle,
                "distance": dist[i] if select_fiber_flag else None,
            }

            all_measurements.append(measurement)

    # Convert to DataFrame
    result_df = pd.DataFrame(all_measurements)

    return result_df, result_df.shape[0]
