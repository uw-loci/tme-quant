"""
Boundary analysis processor.

This module provides high-level boundary analysis orchestration,
implementing algorithms from getBoundary.m and getTifBoundary.m.
"""

from typing import Sequence, Optional, List
import numpy as np
from scipy.spatial.distance import cdist

from ...types import Curvelet, Boundary, BoundaryMetrics
from ..algorithms.matlab_boundary_angle import (
    calculate_matlab_boundary_angles, 
    calculate_matlab_relative_angles
)


def measure_boundary_alignment(
    curvelets: Sequence[Curvelet],
    boundary: Boundary,
    dist_thresh: float,
    min_dist: Optional[float] = None,
    exclude_inside_mask: bool = False,
    boundary_point_indices: Optional[List[int]] = None,
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
            centers, boundary.data, boundary_point_indices
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
    
    # Use MATLAB-compatible relative angle calculation
    relative_angles = calculate_matlab_relative_angles(valid_curvelet_angles, valid_boundary_angles)
    
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


def _analyze_polygon_boundary(centers: np.ndarray, polygon_data, boundary_point_indices: Optional[List[int]] = None) -> tuple:
    """
    Analyze boundary defined by polygon(s).
    
    Computes distances and boundary tangent angles at nearest points.
    """
    # polygon_data should be Nx2 array of [row, col] coordinates
    if isinstance(polygon_data, list):
        polygon_data = np.array(polygon_data)
    
    n_points = len(centers)
    distances = np.zeros(n_points)
    boundary_angles = np.zeros(n_points)
    inside_mask = np.zeros(n_points, dtype=bool)
    
    if boundary_point_indices is not None:
        # Use specific boundary point indices (MATLAB-style)
        if len(boundary_point_indices) != len(centers):
            raise ValueError("boundary_point_indices must have same length as centers")
        
        # Calculate distances to specific boundary points
        distances = []
        for i, (center, boundary_idx) in enumerate(zip(centers, boundary_point_indices)):
            if boundary_idx >= len(polygon_data):
                raise ValueError(f"Boundary index {boundary_idx} out of range (max: {len(polygon_data)-1})")
            boundary_point = polygon_data[boundary_idx]
            distance = np.sqrt(np.sum((center - boundary_point) ** 2))
            distances.append(distance)
        distances = np.array(distances)
        
        # Calculate boundary angles using specific indices
        boundary_angles = np.zeros(len(centers))
        for i, boundary_idx in enumerate(boundary_point_indices):
            if boundary_idx < len(polygon_data):
                # Call FindOutlineSlope for the specific boundary point
                from ..algorithms.matlab_boundary_angle import find_outline_slope
                boundary_angles[i] = find_outline_slope(polygon_data, boundary_idx)
    else:
        # Use nearest boundary point (fallback behavior)
        distances_to_boundary = cdist(centers, polygon_data)
        
        # Find nearest boundary point for each curvelet
        nearest_indices = np.argmin(distances_to_boundary, axis=1)
        distances = np.min(distances_to_boundary, axis=1)
        
        # Use MATLAB-compatible boundary angle calculation
        boundary_angles = calculate_matlab_boundary_angles(polygon_data, centers)
    
    # For inside_mask, we'd need polygon containment test
    # For now, leave as all False (assumes curvelets are outside)
    # A proper implementation would use shapely's contains() or similar
    
    return distances, boundary_angles, inside_mask


def _compute_relative_angles(curvelet_angles: np.ndarray, boundary_angles: np.ndarray) -> np.ndarray:
    """
    Compute relative angles between curvelets and boundary.
    
    This implements the MATLAB formula from getRelativeangles.m:
        tempAng = circ_r([fibAng*2*pi/180; boundaryAngle*2*pi/180]);
        tempAng = 180*asin(tempAng)/pi;
    
    The circ_r function computes mean resultant vector length:
        r = abs(mean(exp(1i*alpha)))
    
    For two angles [a1, a2], this gives a measure of how aligned they are.
    
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
    # MATLAB implementation: circ_r([fibAng*2*pi/180; boundaryAngle*2*pi/180])
    # Convert to radians (multiply by 2 as in MATLAB to account for 180° periodicity)
    fib_rad = curvelet_angles * 2 * np.pi / 180
    bound_rad = boundary_angles * 2 * np.pi / 180
    
    # Compute circ_r: mean resultant vector length for two angles
    # For two angles [a1, a2]: r = abs(mean(exp(1i*[a1, a2])))
    complex_angles = np.exp(1j * np.vstack([fib_rad, bound_rad]))
    mean_complex = np.mean(complex_angles, axis=0)
    circ_r = np.abs(mean_complex)
    
    # Apply MATLAB formula: 180*asin(tempAng)/pi
    relative_angles = 180 * np.arcsin(np.clip(circ_r, -1, 1)) / np.pi
    
    # Take absolute value to get [0, 90] range
    relative_angles = np.abs(relative_angles)
    
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
