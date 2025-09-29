"""
Boundary analysis processor.

This module provides high-level boundary analysis orchestration,
implementing algorithms from getBoundary.m and getTifBoundary.m.
"""

from typing import Sequence, Optional
import numpy as np
from scipy.spatial.distance import cdist

from ...types import Curvelet, Boundary, BoundaryMetrics


def measure_boundary_alignment(
    curvelets: Sequence[Curvelet],
    boundary: Boundary,
    dist_thresh: float,
    min_dist: Optional[float] = None,
    exclude_inside_mask: bool = False,
) -> BoundaryMetrics:
    """
    Measure curvelet alignment relative to a boundary.
    
    This implements the boundary analysis from getBoundary.m and getTifBoundary.m,
    computing distances and relative angles between curvelets and boundary.
    
    Parameters
    ----------
    curvelets : Sequence[Curvelet]
        List of curvelets to analyze
    boundary : Boundary
        Boundary definition (polygon coordinates or binary mask)
    dist_thresh : float
        Distance threshold for inclusion in analysis
    min_dist : float, optional
        Minimum distance from boundary
    exclude_inside_mask : bool, default False
        Whether to exclude curvelets inside boundary mask
        
    Returns
    -------
    BoundaryMetrics
        Boundary analysis results including relative angles and distances
    """
    if not curvelets:
        return BoundaryMetrics(
            relative_angles=np.array([]),
            distances=np.array([]),
            inside_mask=np.array([]),
            alignment_stats={}
        )
    
    n_curvelets = len(curvelets)
    centers = np.array([[c.center_row, c.center_col] for c in curvelets])
    angles = np.array([c.angle_deg for c in curvelets])
    
    if boundary.kind == "mask":
        distances, boundary_angles, inside_mask = _analyze_mask_boundary(
            centers, boundary.data
        )
    elif boundary.kind in ["polygon", "polygons"]:
        distances, boundary_angles, inside_mask = _analyze_polygon_boundary(
            centers, boundary.data
        )
    else:
        raise ValueError(f"Unsupported boundary kind: {boundary.kind}")
    
    # Apply distance thresholding
    distance_mask = distances <= dist_thresh
    if min_dist is not None:
        distance_mask &= distances >= min_dist
    
    # Apply inside/outside exclusion
    if exclude_inside_mask:
        distance_mask &= ~inside_mask
    
    # Filter to relevant curvelets
    valid_indices = np.where(distance_mask)[0]
    
    if len(valid_indices) == 0:
        return BoundaryMetrics(
            relative_angles=np.array([]),
            distances=np.array([]),
            inside_mask=inside_mask,
            alignment_stats={}
        )
    
    # Compute relative angles for valid curvelets
    valid_curvelet_angles = angles[valid_indices]
    valid_boundary_angles = boundary_angles[valid_indices]
    
    relative_angles = _compute_relative_angles(valid_curvelet_angles, valid_boundary_angles)
    
    # Compute alignment statistics
    alignment_stats = _compute_alignment_statistics(relative_angles)
    
    return BoundaryMetrics(
        relative_angles=relative_angles,
        distances=distances[valid_indices],
        inside_mask=inside_mask,
        alignment_stats=alignment_stats
    )


def _analyze_mask_boundary(centers: np.ndarray, mask: np.ndarray) -> tuple:
    """Analyze boundary defined by binary mask."""
    n_points = len(centers)
    distances = np.zeros(n_points)
    boundary_angles = np.zeros(n_points)
    inside_mask = np.zeros(n_points, dtype=bool)
    
    # Find boundary pixels
    from scipy.ndimage import binary_erosion
    boundary_pixels = mask & ~binary_erosion(mask)
    boundary_coords = np.column_stack(np.where(boundary_pixels))
    
    if len(boundary_coords) == 0:
        return distances, boundary_angles, inside_mask
    
    # Compute distances to boundary
    distances_to_boundary = cdist(centers, boundary_coords)
    distances = np.min(distances_to_boundary, axis=1)
    
    # Determine inside/outside
    for i, center in enumerate(centers):
        row, col = int(center[0]), int(center[1])
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            inside_mask[i] = mask[row, col]
    
    # Compute boundary angles (simplified - would need gradient computation)
    boundary_angles = np.zeros(n_points)  # Placeholder
    
    return distances, boundary_angles, inside_mask


def _analyze_polygon_boundary(centers: np.ndarray, polygon_data) -> tuple:
    """Analyze boundary defined by polygon(s)."""
    # This is a simplified implementation
    # In practice, would use shapely or similar for robust polygon operations
    n_points = len(centers)
    distances = np.zeros(n_points)
    boundary_angles = np.zeros(n_points)
    inside_mask = np.zeros(n_points, dtype=bool)
    
    # Placeholder implementation
    return distances, boundary_angles, inside_mask


def _compute_relative_angles(curvelet_angles: np.ndarray, boundary_angles: np.ndarray) -> np.ndarray:
    """
    Compute relative angles between curvelets and boundary.
    
    Parameters
    ----------
    curvelet_angles : np.ndarray
        Curvelet orientation angles in degrees
    boundary_angles : np.ndarray
        Boundary orientation angles in degrees
        
    Returns
    -------
    np.ndarray
        Relative angles in degrees (0 = parallel, 90 = perpendicular)
    """
    # Compute angle differences
    angle_diffs = np.abs(curvelet_angles - boundary_angles)
    
    # Map to [0, 90] range (fiber symmetry and relative angle symmetry)
    relative_angles = np.minimum(angle_diffs, 180 - angle_diffs)
    relative_angles = np.minimum(relative_angles, 90)
    
    return relative_angles


def _compute_alignment_statistics(relative_angles: np.ndarray) -> dict:
    """
    Compute summary statistics for boundary alignment.
    
    Parameters
    ----------
    relative_angles : np.ndarray
        Relative angles in degrees
        
    Returns
    -------
    dict
        Dictionary of alignment statistics
    """
    if len(relative_angles) == 0:
        return {}
    
    # Basic statistics
    mean_relative_angle = np.mean(relative_angles)
    std_relative_angle = np.std(relative_angles)
    
    # Alignment fractions
    parallel_threshold = 30  # degrees
    perpendicular_threshold = 60  # degrees
    
    fraction_parallel = np.sum(relative_angles <= parallel_threshold) / len(relative_angles)
    fraction_perpendicular = np.sum(relative_angles >= perpendicular_threshold) / len(relative_angles)
    
    # Alignment strength (lower relative angle = higher alignment)
    alignment_strength = 1.0 - (mean_relative_angle / 90.0)
    
    return {
        'mean_relative_angle': mean_relative_angle,
        'std_relative_angle': std_relative_angle,
        'fraction_parallel': fraction_parallel,
        'fraction_perpendicular': fraction_perpendicular,
        'alignment_strength': alignment_strength,
        'n_curvelets': len(relative_angles)
    }
