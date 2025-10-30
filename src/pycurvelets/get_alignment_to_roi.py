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
    fiber_structure: Union[pd.DataFrame, List[Dict]],
    distance_threshold: Optional[float] = None,
) -> Union[ROIMeasurements, List[ROIMeasurements]]:
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

    fiber_structure : pd.DataFrame or list of dict
        Fiber data containing center coordinates and angles. If DataFrame, should have:
        - 'center_row' : float
            Fiber center row coordinate (y)
        - 'center_col' : float
            Fiber center column coordinate (x)
        - 'angle' : float
            Fiber orientation angle in degrees
        If list of dicts, each dict should have 'center' and 'angle' keys.

    distance_threshold : float, optional
        Maximum distance from fiber to ROI boundary for inclusion in analysis.
        If None, uses pre-selected fibers from roi_list['index2object'].

    Returns
    -------
    ROIMeasurements or list of ROIMeasurements
        Single ROIMeasurements object if single ROI input, otherwise list.
        Each ROIMeasurements contains:
        - angle_to_boundary_edge : ndarray
            Angles between fiber orientation and boundary edge at closest point
        - angle_to_boundary_center : ndarray
            Angles between fiber orientation and ROI center orientation
        - angle_to_center_line : ndarray
            Angles between fiber orientation and fiber-to-ROI-center line
        - fiber_center_list : ndarray of shape (n_fibers, 2)
            Fiber center coordinates (y, x)
        - fiber_angle_list : ndarray
            Fiber orientation angles in degrees
        - distance_list : ndarray
            Distances from fibers to closest ROI boundary points
        - boundary_points : ndarray of shape (n_fibers, 2)
            Closest boundary point coordinates for each fiber
        - n_fibers : int
            Number of fibers analyzed for this ROI
    """

    # Input validation
    if roi_list is None:
        raise ValueError("roi_list cannot be None")

    if fiber_structure is None or len(fiber_structure) == 0:
        raise ValueError("fiber_structure cannot be None or empty")

    # Determine processing mode
    select_fiber_flag = distance_threshold is not None

    # logger.info(f"Processing {len(roi_list)} ROI(s) with {len(fiber_data)} fibers")
    if select_fiber_flag:
        logger.info(
            f"Using distance-based selection with threshold: {distance_threshold}"
        )
    else:
        logger.info("Using pre-selected fiber mode")

    # Process each ROI
    number_of_roi = len(roi_list)
    number_of_fibers = len(fiber_structure)
    ROI_measurements_all = []

    for roi_i in range(len(roi_list.coordinates)):
        curr_ROI = ROI(
            coordinates=roi_list.coordinates[roi_i],
            image_width=roi_list.image_width,
            image_height=roi_list.image_height,
        )
        distance_precalc = None

        if select_fiber_flag:
            roi_tree = KDTree(roi_list.coordinates[roi_i])
            fiber_centers = np.array([f.center for f in fiber_structure[roi_i]])
            dist, idx_dist = roi_tree.query(fiber_centers)
            fiber_indices = np.where(dist <= distance_threshold)[0]
        else:
            # if distance_threshold is None, then "distance" should be set as a property of the ROI
            fiber_indices = range(len(fiber_structure))

            # -- THIS SHOULDN'T WORK -- NEED TO FIGURE OUT DISTANCE ATTRIBUTE -- #
            distance_precalc = roi_list[roi_i].distance

        curr_ROI_measurements = []

        if fiber_indices is not None:
            number_of_fibers = len(fiber_indices)
            if select_fiber_flag:
                if number_of_fibers == 1:
                    print(
                        f"ROI {roi_i} found one fiber within the {distance_threshold} distance to the boundary"
                    )
                else:
                    print(
                        f"ROI {roi_i} found {number_of_fibers} fiber within the {distance_threshold} distance to the boundary"
                    )
            else:
                print(
                    f"Calculate all the relative alignment of pre-selected fibers. Total fiber number is {number_of_fibers}"
                )
            for fiber_i in number_of_fibers:
                i = fiber_indices[fiber_i]
                if select_fiber_flag:
                    curr_ROI.coordinates = roi_list.coordinates[i]
                    curr_ROI.index_to_object = idx_dist[i]
                else:
                    curr_ROI.index_to_object = roi_list[roi_i].index_to_object[fiber_i]
                curr_fiber = Fiber(
                    center_row=fiber_structure[i].center[0],
                    center_col=fiber_structure[i].center[1],
                    angle=fiber_structure[i].angle,
                )

    return ROI_measurements_all
