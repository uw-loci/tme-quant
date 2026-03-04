"""
Visualization utilities for image registration.

Location: visualization/checkerboard.py
          visualization/overlay.py
          visualization/difference_map.py
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# ============================================================
# DIFFERENCE MAP
# visualization/difference_map.py
# ============================================================

def compute_difference_map(
    fixed: np.ndarray,
    moving: np.ndarray,
    absolute: bool = True
) -> np.ndarray:
    """
    Compute difference between images.
    
    Highlights registration errors.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        absolute: Use absolute difference
        
    Returns:
        Difference map
    """
    # Ensure same size
    if fixed.shape != moving.shape:
        raise ValueError("Images must have same shape")
    
    # Convert to float
    fixed = fixed.astype(float)
    moving = moving.astype(float)
    
    # Compute difference
    if absolute:
        diff = np.abs(fixed - moving)
    else:
        diff = fixed - moving
    
    return diff


def create_difference_overlay(
    fixed: np.ndarray,
    moving: np.ndarray,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create colored difference map overlay.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        colormap: Matplotlib colormap name
        
    Returns:
        Colored difference overlay
    """
    # Compute difference
    diff = compute_difference_map(fixed, moving, absolute=True)
    
    # Normalize to [0, 1]
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(diff_norm)[:, :, :3]  # Remove alpha channel
    
    return colored


def compute_registration_quality_map(
    fixed: np.ndarray,
    moving: np.ndarray,
    window_size: int = 31
) -> np.ndarray:
    """
    Compute local registration quality using sliding window.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        window_size: Size of local window
        
    Returns:
        Quality map (higher = better registration)
    """
    from scipy import signal
    
    # Convert to grayscale
    if fixed.ndim == 3:
        fixed = np.mean(fixed, axis=2)
    if moving.ndim == 3:
        moving = np.mean(moving, axis=2)
    
    # Normalize
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-8)
    moving = (moving - moving.mean()) / (moving.std() + 1e-8)
    
    # Compute local correlation using convolution
    window = np.ones((window_size, window_size)) / (window_size ** 2)
    
    # Local means
    fixed_mean = signal.correlate2d(fixed, window, mode='same', boundary='symm')
    moving_mean = signal.correlate2d(moving, window, mode='same', boundary='symm')
    
    # Local standard deviations
    fixed_sq_mean = signal.correlate2d(fixed**2, window, mode='same', boundary='symm')
    moving_sq_mean = signal.correlate2d(moving**2, window, mode='same', boundary='symm')
    
    fixed_std = np.sqrt(np.maximum(fixed_sq_mean - fixed_mean**2, 0))
    moving_std = np.sqrt(np.maximum(moving_sq_mean - moving_mean**2, 0))
    
    # Local covariance
    product_mean = signal.correlate2d(fixed * moving, window, mode='same', boundary='symm')
    covariance = product_mean - fixed_mean * moving_mean
    
    # Local correlation coefficient
    correlation = covariance / (fixed_std * moving_std + 1e-8)
    
    # Clip to [0, 1]
    quality_map = np.clip(correlation, 0, 1)
    
    return quality_map
