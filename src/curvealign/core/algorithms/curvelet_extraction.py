"""
Curvelet extraction and processing algorithms.

This module provides functions for extracting curvelet objects from
coefficient data and processing them for analysis.
"""

from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree

from ...types import Curvelet


def extract_curvelets_from_coeffs(
    scale_coeffs: List[np.ndarray], 
    X_rows: List[np.ndarray], 
    Y_cols: List[np.ndarray], 
    scale_idx: int
) -> List[Curvelet]:
    """
    Extract Curvelet objects from coefficient data at one scale.
    
    Parameters
    ----------
    scale_coeffs : List[np.ndarray]
        Thresholded coefficients for all wedges at one scale
    X_rows : List[np.ndarray]
        Row center positions for each wedge
    Y_cols : List[np.ndarray] 
        Column center positions for each wedge
    scale_idx : int
        Scale index for angle calculation
        
    Returns
    -------
    List[Curvelet]
        Extracted curvelet objects
    """
    curvelets = []
    
    for wedge_idx, coeff in enumerate(scale_coeffs):
        if coeff.size == 0 or wedge_idx >= len(X_rows) or wedge_idx >= len(Y_cols):
            continue
        
        # Find non-zero coefficient positions
        nonzero_rows, nonzero_cols = np.nonzero(coeff)
        
        if len(nonzero_rows) == 0:
            continue
        
        # Get center positions for this wedge
        x_centers = X_rows[wedge_idx]
        y_centers = Y_cols[wedge_idx]
        
        if x_centers.size == 0 or y_centers.size == 0:
            continue
        
        # Calculate angle for this wedge (based on MATLAB fdct structure)
        n_wedges = len(scale_coeffs)
        if n_wedges > 1:
            angle_deg = (wedge_idx * 180.0) / n_wedges
        else:
            angle_deg = 0.0
        
        # Create curvelet for each non-zero coefficient
        for row, col in zip(nonzero_rows, nonzero_cols):
            if row < x_centers.shape[0] and col < x_centers.shape[1]:
                center_row = int(x_centers[row, col])
                center_col = int(y_centers[row, col])
                weight = float(np.abs(coeff[row, col]))
                
                curvelet = Curvelet(
                    center_row=center_row,
                    center_col=center_col,
                    angle_deg=angle_deg,
                    weight=weight
                )
                curvelets.append(curvelet)
    
    return curvelets


def group_curvelets(curvelets: List[Curvelet], radius: float) -> List[Curvelet]:
    """
    Group nearby curvelets and compute mean angles.
    
    This implements the grouping logic from group6.m.
    
    Parameters
    ----------
    curvelets : List[Curvelet]
        Input curvelets to group
    radius : float
        Grouping radius in pixels
        
    Returns
    -------
    List[Curvelet]
        Grouped curvelets with averaged positions and angles
    """
    if len(curvelets) <= 1:
        return curvelets
    
    # Build position array for spatial search
    positions = np.array([[c.center_row, c.center_col] for c in curvelets])
    
    # Use KD-tree for efficient neighbor search
    tree = cKDTree(positions)
    
    # Track which curvelets have been grouped
    grouped = np.zeros(len(curvelets), dtype=bool)
    result_curvelets = []
    
    for i, curvelet in enumerate(curvelets):
        if grouped[i]:
            continue
        
        # Find neighbors within radius
        neighbor_indices = tree.query_ball_point([curvelet.center_row, curvelet.center_col], radius)
        
        if len(neighbor_indices) == 1:
            # No neighbors, keep original
            result_curvelets.append(curvelet)
            grouped[i] = True
        else:
            # Group neighbors
            neighbor_curvelets = [curvelets[j] for j in neighbor_indices]
            grouped_curvelet = _average_curvelets(neighbor_curvelets)
            result_curvelets.append(grouped_curvelet)
            
            # Mark all neighbors as grouped
            for j in neighbor_indices:
                grouped[j] = True
    
    return result_curvelets


def normalize_angles(curvelets: List[Curvelet]) -> List[Curvelet]:
    """
    Normalize curvelet angles to 0-180 degree range.
    
    This implements the angle normalization from group6.m,
    exploiting fiber symmetry (0° = 180°).
    
    Parameters
    ----------
    curvelets : List[Curvelet]
        Input curvelets
        
    Returns
    -------
    List[Curvelet]
        Curvelets with normalized angles
    """
    normalized = []
    
    for curvelet in curvelets:
        angle = curvelet.angle_deg % 180.0  # Map to [0, 180)
        
        normalized_curvelet = Curvelet(
            center_row=curvelet.center_row,
            center_col=curvelet.center_col,
            angle_deg=angle,
            weight=curvelet.weight
        )
        normalized.append(normalized_curvelet)
    
    return normalized


def filter_edge_curvelets(curvelets: List[Curvelet], image_shape: Tuple[int, int], margin: int = 5) -> List[Curvelet]:
    """
    Filter out curvelets too close to image edges.
    
    Parameters
    ----------
    curvelets : List[Curvelet]
        Input curvelets
    image_shape : Tuple[int, int]
        Shape of the original image (rows, cols)
    margin : int, default 5
        Minimum distance from edge in pixels
        
    Returns
    -------
    List[Curvelet]
        Filtered curvelets
    """
    filtered = []
    rows, cols = image_shape
    
    for curvelet in curvelets:
        # Check if curvelet is far enough from edges
        if (margin <= curvelet.center_row < rows - margin and
            margin <= curvelet.center_col < cols - margin):
            filtered.append(curvelet)
    
    return filtered


def _average_curvelets(curvelets: List[Curvelet]) -> Curvelet:
    """
    Compute average position and angle for a group of curvelets.
    
    Parameters
    ----------
    curvelets : List[Curvelet]
        Curvelets to average
        
    Returns
    -------
    Curvelet
        Averaged curvelet
    """
    if len(curvelets) == 1:
        return curvelets[0]
    
    # Average positions
    avg_row = np.mean([c.center_row for c in curvelets])
    avg_col = np.mean([c.center_col for c in curvelets])
    
    # Average angles (handling circular nature)
    angles_rad = np.array([np.radians(c.angle_deg) for c in curvelets])
    
    # Convert to complex numbers for circular averaging
    complex_angles = np.exp(1j * angles_rad)
    avg_complex = np.mean(complex_angles)
    avg_angle_rad = np.angle(avg_complex)
    avg_angle_deg = np.degrees(avg_angle_rad) % 180.0
    
    # Average weights
    weights = [c.weight for c in curvelets if c.weight is not None]
    avg_weight = np.mean(weights) if weights else None
    
    return Curvelet(
        center_row=int(round(avg_row)),
        center_col=int(round(avg_col)),
        angle_deg=avg_angle_deg,
        weight=avg_weight
    )
