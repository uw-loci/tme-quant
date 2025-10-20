"""
MATLAB-compatible boundary angle calculation implementation.

This module implements the exact MATLAB functions:
- FindOutlineSlope.m
- FindConnectedPts.m  
- GetFirstNeighbor.m

These functions are used to calculate boundary tangent angles that match
the original MATLAB implementation exactly.
"""

import numpy as np
from typing import List, Tuple, Optional


def get_first_neighbor(boundary_mask: np.ndarray, idx: int, visited_list: np.ndarray, direction: int) -> int:
    """
    Find the first contiguous neighbor in the boundary mask.
    
    This implements GetFirstNeighbor.m from MATLAB.
    
    Parameters
    ----------
    boundary_mask : np.ndarray
        List of [row, col] coordinates of boundary points
    idx : int
        Index of the point around which to search
    visited_list : np.ndarray
        Array tracking which points have been visited
    direction : int
        1 or 2 for different search directions
        
    Returns
    -------
    int
        Index of the first neighbor found
    """
    pt = boundary_mask[idx]
    
    # Define search order based on direction (MATLAB logic)
    if direction == 1:
        # Search order: E, NE, N, NW, W, SW, S, SE
        npt = np.array([
            [pt[0], pt[1] + 1],     # E
            [pt[0] - 1, pt[1] + 1], # NE
            [pt[0] - 1, pt[1]],     # N
            [pt[0] - 1, pt[1] - 1], # NW
            [pt[0], pt[1] - 1],     # W
            [pt[0] + 1, pt[1] - 1], # SW
            [pt[0] + 1, pt[1]],     # S
            [pt[0] + 1, pt[1] + 1]  # SE
        ])
    else:  # direction == 2
        # Search order: W, SW, S, SE, E, NE, N, NW
        npt = np.array([
            [pt[0], pt[1] - 1],     # W
            [pt[0] + 1, pt[1] - 1], # SW
            [pt[0] + 1, pt[1]],     # S
            [pt[0] + 1, pt[1] + 1], # SE
            [pt[0], pt[1] + 1],     # E
            [pt[0] - 1, pt[1] + 1], # NE
            [pt[0] - 1, pt[1]],     # N
            [pt[0] - 1, pt[1] - 1]  # NW
        ])
    
    outpt = idx
    rows = boundary_mask[:, 0]
    cols = boundary_mask[:, 1]
    
    # Check each neighbor in order
    for i in range(len(npt)):
        # Find position in the list that matches the neighbor
        matches = (rows == npt[i, 0]) & (cols == npt[i, 1])
        if np.any(matches):
            chk_idx = np.where(matches)[0][0]
            if visited_list[chk_idx] == 0:
                outpt = chk_idx
                return outpt
    
    return outpt


def find_connected_pts(boundary_mask: np.ndarray, idx: int, num: int) -> np.ndarray:
    """
    Find list of connected points around a point.
    
    This implements FindConnectedPts.m from MATLAB.
    
    Parameters
    ----------
    boundary_mask : np.ndarray
        List of [row, col] coordinates of boundary points
    idx : int
        Index of the point around which to find connected points
    num : int
        Number of points to return (should be odd)
        
    Returns
    -------
    np.ndarray
        Array of connected points [row, col], with NaN for missing points
    """
    con_pts = np.full((num, 2), np.nan)
    
    # Place point in the middle of output list (odd length)
    hnum = (num - 1) // 2
    mid = hnum
    con_pts[mid] = boundary_mask[idx]
    
    # Fill list to beginning
    sidx = idx  # starting index
    visited_list = np.zeros(len(boundary_mask), dtype=int)  # keep track of visited pixels
    
    for i in range(mid - 1, -1, -1):
        visited_list[idx] = 1
        idx = get_first_neighbor(boundary_mask, idx, visited_list, 1)
        con_pts[i] = boundary_mask[idx]
    
    # Fill list in other direction
    idx = sidx
    for i in range(mid + 1, num):
        visited_list[idx] = 1
        idx = get_first_neighbor(boundary_mask, idx, visited_list, 2)
        con_pts[i] = boundary_mask[idx]
    
    return con_pts


def find_outline_slope(boundary_mask: np.ndarray, idx: int) -> float:
    """
    Find the angle of the boundary edge.
    
    This implements FindOutlineSlope.m from MATLAB.
    
    Parameters
    ----------
    boundary_mask : np.ndarray
        List of [row, col] coordinates of boundary points
    idx : int
        Index to a point on the boundary
        
    Returns
    -------
    float
        The absolute angle of the outline around the point (in degrees, 0 to 180)
    """
    slope = np.nan
    
    # Find the list of connected points on the outline surrounding the point
    num = 21  # number of points to return
    con_pts = find_connected_pts(boundary_mask, idx, num)
    
    if np.isnan(con_pts[0, 0]):
        return slope
    
    # Compute absolute slope of the tangent
    # rise = con_pts(num,1) - con_pts(1,1)
    # run = con_pts(num,2) - con_pts(1,2)
    rise = con_pts[num - 1, 0] - con_pts[0, 0]  # row difference
    run = con_pts[num - 1, 1] - con_pts[0, 1]   # col difference
    
    # theta = atan(rise/run); %range -pi/2 to pi/2
    theta = np.arctan2(rise, run)  # Use atan2 for better numerical stability
    
    # Scale to 0 to 180 degrees
    slope = theta * 180 / np.pi
    if slope < 0:
        slope = slope + 180
    
    # Fit a curve to these points, then compute floating point angle of tangent line
    if slope < 45 or slope > 135:
        # More unique points in vertical direction
        y_f = np.linspace(con_pts[0, 1], con_pts[-1, 1], 50)  # col values
        x_p = np.polyfit(con_pts[:, 1], con_pts[:, 0], 2)    # fit x as function of y
        x_f = np.polyval(x_p, y_f)
        
        # Calculate derivative (tangent) at midpoint
        mid_idx = len(y_f) // 2
        if mid_idx > 0 and mid_idx < len(y_f) - 1:
            dx_dy = (x_f[mid_idx + 1] - x_f[mid_idx - 1]) / (y_f[mid_idx + 1] - y_f[mid_idx - 1])
            slope = np.arctan(dx_dy) * 180 / np.pi
            if slope < 0:
                slope = slope + 180
    else:
        # More unique points in horizontal direction
        x_f = np.linspace(con_pts[0, 0], con_pts[-1, 0], 50)  # row values
        y_p = np.polyfit(con_pts[:, 0], con_pts[:, 1], 2)    # fit y as function of x
        y_f = np.polyval(y_p, x_f)
        
        # Calculate derivative (tangent) at midpoint
        mid_idx = len(x_f) // 2
        if mid_idx > 0 and mid_idx < len(x_f) - 1:
            dy_dx = (y_f[mid_idx + 1] - y_f[mid_idx - 1]) / (x_f[mid_idx + 1] - x_f[mid_idx - 1])
            slope = np.arctan(dy_dx) * 180 / np.pi
            if slope < 0:
                slope = slope + 180
    
    return slope


def calculate_matlab_boundary_angles(boundary_coords: np.ndarray, curvelet_centers: np.ndarray) -> np.ndarray:
    """
    Calculate boundary angles using the exact MATLAB algorithm.
    
    This function implements the MATLAB boundary angle calculation for multiple curvelets.
    
    Parameters
    ----------
    boundary_coords : np.ndarray
        Array of boundary coordinates [row, col]
    curvelet_centers : np.ndarray
        Array of curvelet center positions [row, col]
        
    Returns
    -------
    np.ndarray
        Array of boundary angles in degrees
    """
    n_curvelets = len(curvelet_centers)
    boundary_angles = np.zeros(n_curvelets)
    
    # For each curvelet, find the nearest boundary point and calculate its slope
    for i, center in enumerate(curvelet_centers):
        # Find nearest boundary point
        distances = np.sqrt(np.sum((boundary_coords - center) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        
        # Calculate boundary slope at this point using MATLAB algorithm
        boundary_angles[i] = find_outline_slope(boundary_coords, nearest_idx)
    
    return boundary_angles


def circ_r_matlab(angles_rad: np.ndarray) -> float:
    """
    Compute mean resultant vector length for circular data (MATLAB circ_r function).
    
    Parameters
    ----------
    angles_rad : np.ndarray
        Sample of angles in radians
        
    Returns
    -------
    float
        Mean resultant length
    """
    # Compute weighted sum of cos and sin of angles
    r = np.sum(np.exp(1j * angles_rad))
    
    # Obtain length
    r = np.abs(r) / len(angles_rad)
    
    return r


def calculate_matlab_relative_angles(fiber_angles: np.ndarray, boundary_angles: np.ndarray) -> np.ndarray:
    """
    Calculate relative angles using the exact MATLAB formula.
    
    This implements the MATLAB calculation from getRelativeangles.m:
        tempAng = circ_r([fibAng*2*pi/180; boundaryAngle*2*pi/180]);
        tempAng = 180*asin(tempAng)/pi;
    
    Parameters
    ----------
    fiber_angles : np.ndarray
        Fiber orientation angles in degrees
    boundary_angles : np.ndarray
        Boundary orientation angles in degrees
        
    Returns
    -------
    np.ndarray
        Relative angles in degrees
    """
    relative_angles = np.zeros(len(fiber_angles))
    
    for i in range(len(fiber_angles)):
        fib_ang = fiber_angles[i]
        boundary_ang = boundary_angles[i]
        
        # Convert to radians (multiply by 2 as in MATLAB for 180° periodicity)
        fib_rad = fib_ang * 2 * np.pi / 180
        boundary_rad = boundary_ang * 2 * np.pi / 180
        
        # Apply circ_r function
        angles_rad = np.array([fib_rad, boundary_rad])
        temp_ang = circ_r_matlab(angles_rad)
        
        # Apply MATLAB formula: 180*asin(tempAng)/pi
        relative_angles[i] = 180 * np.arcsin(np.clip(temp_ang, -1, 1)) / np.pi
    
    return relative_angles
