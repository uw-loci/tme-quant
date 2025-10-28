import numpy as np
from scipy.spatial import KDTree

from pycurvelets.models import ROIMeasurements, ROIList


def get_alignment_to_roi(roi_list, fiber_structure, distance_threshold):

    number_of_roi = len(roi_list)
    number_of_fibers = len(fiber_structure)
    ROI_measurements_all = []

    select_fiber_flag = distance_threshold is not None

    for fiber, roi in enumerate(roi_list):
        bw_coords = roi.coords
        fiber_indices = np.arange(number_of_fibers)
        distance_precalc = None

        if select_fiber_flag:
            roi_tree = KDTree(bw_coords)
            fiber_centers = np.array([f.center for f in fiber_structure])
            dist, idx_dist = roi_tree.query(fiber_centers)
            fiber_indices = np.where(dist <= distance_threshold)[0]
        else:
            if roi.distance is not None:
                distance_precalc = roi.distance
            idx_dist = None

        fiber_count = len(fiber_indices)
        if fiber_count == 0:
            print(f"ROI {fiber}: No fiber selected or within distance.")
            ROI_measurements_all.append(
                ROIMeasurements(
                    angle_to_boundary_edge=np.nan,
                    angle_to_boundary_center=np.nan,
                    angle_to_center_line=np.nan,
                    fiber_center_list=[],
                    fiber_angle_list=[],
                    distance=np.nan,
                    number_of_fibers=0,
                )
            )
            continue

        angle_to_boundary_edge = np.zeros(fiber_count)

    return None
