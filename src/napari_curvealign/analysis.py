"""
Analysis module for CurveAlign Napari plugin.

Provides advanced analysis functions including TACS (Tumor-Associated Collagen Signatures)
and integration with fiber features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from scipy.spatial import KDTree

def compute_tacs(
    fiber_features: Union[pd.DataFrame, List[Dict]],
    boundary_coords: np.ndarray,
    max_distance: float = 200.0,
    min_distance: float = 0.0
) -> pd.DataFrame:
    """
    Compute TACS (Tumor-Associated Collagen Signatures) metrics.
    
    Calculates the relative angle of each fiber to the nearest point on the tumor boundary.
    Relative angle of 0 degrees means tangential (TACS-1/2 like).
    Relative angle of 90 degrees means perpendicular (TACS-3 like).
    
    Parameters
    ----------
    fiber_features : pd.DataFrame or List[Dict]
        Fiber properties. Must contain center (x,y) and orientation/angle columns.
    boundary_coords : np.ndarray
        N x 2 array of boundary coordinates (x, y).
    max_distance : float
        Maximum distance from boundary to include fibers.
    min_distance : float
        Minimum distance from boundary (e.g., to exclude fibers inside the tumor).
        
    Returns
    -------
    pd.DataFrame
        Fiber features with 'distance_to_boundary' and 'relative_angle' columns added.
    """
    # Convert to DataFrame if needed
    if isinstance(fiber_features, list):
        df = pd.DataFrame(fiber_features)
    else:
        df = fiber_features.copy()
        
    if df.empty:
        return pd.DataFrame()
        
    if boundary_coords is None or len(boundary_coords) < 3:
        print("Insufficient boundary coordinates for TACS analysis")
        return df
    
    # Standardize column names for processing
    # Search for x, y, angle
    x_col = next((col for col in ['x', 'center_x', 'col', 'xc'] if col in df.columns), None)
    y_col = next((col for col in ['y', 'center_y', 'row', 'yc'] if col in df.columns), None)
    angle_col = next((col for col in ['angle', 'orientation', 'theta', 'orientation_deg'] if col in df.columns), None)
    
    if not all([x_col, y_col, angle_col]):
        print(f"Missing required columns. Found: x={x_col}, y={y_col}, angle={angle_col}")
        return df
        
    # Extract centers
    centers = np.column_stack((df[x_col].values, df[y_col].values))
    
    # Build KDTree for boundary
    tree = KDTree(boundary_coords)
    
    # Query nearest boundary points
    distances, indices = tree.query(centers)
    
    # Calculate relative angles
    relative_angles = []
    valid_indices = []
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # Distance filter
        if dist > max_distance or dist < min_distance:
            continue
            
        fiber_angle = df.iloc[i][angle_col]
        
        # Calculate boundary tangent at nearest point
        # Use simple central difference of neighbors
        p_prev = boundary_coords[(idx - 1) % len(boundary_coords)]
        p_next = boundary_coords[(idx + 1) % len(boundary_coords)]
        
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
        
        # Angle of tangent vector
        boundary_angle = np.degrees(np.arctan2(dy, dx))
        
        # MATLAB-compatible `circ_r` flow:
        # - convert to radians
        # - double angles to account for axial (180-deg periodic) orientation
        # - compute resultant vector length r = |mean(exp(i * theta))|
        # - map via asin back to relative angle in degrees
        fib_rad = np.deg2rad(fiber_angle) * 2
        bound_rad = np.deg2rad(boundary_angle) * 2

        complex_angles = np.exp(1j * np.array([fib_rad, bound_rad], dtype=float))
        r = float(np.abs(np.mean(complex_angles)))

        # Final relative angle
        # Note: MATLAB's asin returns real part, we clip to valid range [-1, 1]
        rel_angle = np.degrees(np.arcsin(np.clip(r, -1.0, 1.0)))
            
        relative_angles.append(rel_angle)
        valid_indices.append(i)
        
    # Create result DataFrame
    if not valid_indices:
        return pd.DataFrame()
        
    result_df = df.iloc[valid_indices].copy()
    result_df['distance_to_boundary'] = distances[valid_indices]
    result_df['relative_angle'] = relative_angles
    
    return result_df

def bin_relative_angles(
    df: pd.DataFrame, 
    bins: int = 9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin relative angles for histogram plotting.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``relative_angle`` column in degrees.
    bins : int, default 9
        Number of histogram bins spanning 0 to 90 degrees.

    Returns
    -------
    counts : numpy.ndarray
        Histogram counts for each angle bin. Returns an empty array when the
        required column is missing.
    bin_edges : numpy.ndarray
        Histogram bin edges. Returns an empty array when the required column is
        missing.
    """
    if 'relative_angle' not in df.columns:
        return np.array([]), np.array([])
        
    hist, edges = np.histogram(df['relative_angle'], bins=bins, range=(0, 90))
    return hist, edges
