"""
Angle map rendering for curvelet visualization.

This module provides functions for creating angle maps that show
fiber orientation across the image using color coding.
"""

from typing import Sequence, Tuple
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from ...types import Curvelet


def create_angle_maps(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    std_window: int = 24,
    square_window: int = 12,
    gaussian_sigma: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create raw and processed angle maps from curvelets.
    
    This implements the angle map generation from drawMap.m.
    
    Parameters
    ----------
    image : np.ndarray
        Original image for reference
    curvelets : Sequence[Curvelet]
        List of curvelets with angle information
    std_window : int, default 24
        Window size for standard deviation filtering
    square_window : int, default 12
        Window size for square filtering
    gaussian_sigma : float, default 4.0
        Sigma for Gaussian filtering
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw angle map and processed (filtered) angle map
    """
    if not curvelets:
        empty_map = np.zeros(image.shape, dtype=np.float32)
        return empty_map, empty_map
    
    # Create raw angle map
    raw_map = _create_raw_angle_map(image.shape, curvelets)
    
    # Create processed map with filtering
    processed_map = _apply_angle_map_filtering(
        raw_map, 
        std_window=std_window,
        square_window=square_window,
        gaussian_sigma=gaussian_sigma
    )
    
    return raw_map, processed_map


def _create_raw_angle_map(image_shape: tuple, curvelets: Sequence[Curvelet]) -> np.ndarray:
    """
    Create raw angle map by interpolating curvelet angles across the image.
    
    Parameters
    ----------
    image_shape : tuple
        Shape of the target image
    curvelets : Sequence[Curvelet]
        List of curvelets with positions and angles
        
    Returns
    -------
    np.ndarray
        Raw angle map with interpolated angle values
    """
    height, width = image_shape
    
    if not curvelets:
        return np.zeros((height, width), dtype=np.float32)
    
    # Extract curvelet positions and angles
    positions = np.array([[c.center_row, c.center_col] for c in curvelets])
    angles = np.array([c.angle_deg for c in curvelets])
    
    # Create grid for interpolation
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    grid_points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    
    # Interpolate angles to full image grid
    try:
        interpolated_angles = griddata(
            positions, angles, grid_points, 
            method='cubic', fill_value=0.0
        )
        angle_map = interpolated_angles.reshape((height, width))
    except Exception:
        # Fallback to nearest neighbor if cubic fails
        interpolated_angles = griddata(
            positions, angles, grid_points, 
            method='nearest', fill_value=0.0
        )
        angle_map = interpolated_angles.reshape((height, width))
    
    return angle_map.astype(np.float32)


def _apply_angle_map_filtering(
    angle_map: np.ndarray,
    std_window: int = 24,
    square_window: int = 12,
    gaussian_sigma: float = 4.0
) -> np.ndarray:
    """
    Apply filtering operations to create processed angle map.
    
    This implements the filtering logic from drawMap.m including
    standard deviation filtering and Gaussian smoothing.
    
    Parameters
    ----------
    angle_map : np.ndarray
        Raw angle map
    std_window : int
        Window size for standard deviation filtering
    square_window : int
        Window size for square filtering
    gaussian_sigma : float
        Sigma for Gaussian filtering
        
    Returns
    -------
    np.ndarray
        Processed angle map
    """
    processed_map = angle_map.copy()
    
    # Apply standard deviation filtering
    if std_window > 1:
        processed_map = _std_deviation_filter(processed_map, std_window)
    
    # Apply square window averaging
    if square_window > 1:
        processed_map = _square_window_filter(processed_map, square_window)
    
    # Apply Gaussian smoothing
    if gaussian_sigma > 0:
        processed_map = gaussian_filter(processed_map, sigma=gaussian_sigma)
    
    return processed_map


def _std_deviation_filter(angle_map: np.ndarray, window_size: int) -> np.ndarray:
    """Apply standard deviation filtering to angle map."""
    from scipy.ndimage import uniform_filter
    
    # Compute local mean and variance using uniform filters
    local_mean = uniform_filter(angle_map.astype(np.float64), size=window_size)
    local_mean_sq = uniform_filter(angle_map.astype(np.float64)**2, size=window_size)
    local_var = local_mean_sq - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Filter based on standard deviation threshold
    std_threshold = np.percentile(local_std[local_std > 0], 75)
    mask = local_std <= std_threshold
    
    filtered_map = angle_map.copy()
    filtered_map[~mask] = local_mean[~mask]
    
    return filtered_map


def _square_window_filter(angle_map: np.ndarray, window_size: int) -> np.ndarray:
    """Apply square window averaging filter."""
    from scipy.ndimage import uniform_filter
    
    return uniform_filter(angle_map.astype(np.float64), size=window_size).astype(np.float32)


def angle_map_to_rgb(angle_map: np.ndarray, colormap: str = "hsv") -> np.ndarray:
    """
    Convert angle map to RGB image using color coding.
    
    Parameters
    ----------
    angle_map : np.ndarray
        Angle map in degrees
    colormap : str, default "hsv"
        Colormap for angle visualization
        
    Returns
    -------
    np.ndarray
        RGB image with angle-coded colors
    """
    import matplotlib.pyplot as plt
    
    # Normalize angles to [0, 1]
    normalized_angles = (angle_map % 180) / 180.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    rgb_image = cmap(normalized_angles)[:, :, :3]  # Remove alpha channel
    
    return (rgb_image * 255).astype(np.uint8)
