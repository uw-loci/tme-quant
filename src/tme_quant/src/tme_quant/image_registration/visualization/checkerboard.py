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
# CHECKERBOARD VISUALIZATION
# visualization/checkerboard.py
# ============================================================

def create_checkerboard(
    fixed: np.ndarray,
    moving: np.ndarray,
    num_squares: int = 10
) -> np.ndarray:
    """
    Create checkerboard pattern for visual comparison.
    
    Alternates between fixed and moving images in a grid pattern.
    Useful for assessing registration quality.
    
    Args:
        fixed: Fixed/reference image
        moving: Moving/registered image
        num_squares: Number of squares along each dimension
        
    Returns:
        Checkerboard composite image
    """
    # Ensure same size
    if fixed.shape != moving.shape:
        raise ValueError("Images must have same shape")
    
    # Convert to grayscale if needed
    if fixed.ndim == 3:
        fixed = np.mean(fixed, axis=2)
    if moving.ndim == 3:
        moving = np.mean(moving, axis=2)
    
    # Create checkerboard mask
    h, w = fixed.shape
    square_h = h // num_squares
    square_w = w // num_squares
    
    mask = np.zeros((h, w), dtype=bool)
    
    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                mask[i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w] = True
    
    # Create checkerboard
    checkerboard = np.where(mask, fixed, moving)
    
    return checkerboard


def create_checkerboard_rgb(
    fixed: np.ndarray,
    moving: np.ndarray,
    num_squares: int = 10,
    fixed_color: str = 'red',
    moving_color: str = 'green'
) -> np.ndarray:
    """
    Create color checkerboard for better visualization.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        num_squares: Number of squares
        fixed_color: Color for fixed image ('red', 'green', 'blue')
        moving_color: Color for moving image
        
    Returns:
        RGB checkerboard image
    """
    # Convert to grayscale
    if fixed.ndim == 3:
        fixed = np.mean(fixed, axis=2)
    if moving.ndim == 3:
        moving = np.mean(moving, axis=2)
    
    # Normalize
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    
    # Create RGB channels
    h, w = fixed.shape
    rgb = np.zeros((h, w, 3))
    
    # Color mapping
    color_map = {'red': 0, 'green': 1, 'blue': 2}
    fixed_channel = color_map.get(fixed_color, 0)
    moving_channel = color_map.get(moving_color, 1)
    
    # Create checkerboard mask
    square_h = h // num_squares
    square_w = w // num_squares
    mask = np.zeros((h, w), dtype=bool)
    
    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                mask[i*square_h:(i+1)*square_h, j*square_w:(j+1)*square_w] = True
    
    # Apply colors
    rgb[:, :, fixed_channel] = np.where(mask, fixed, 0)
    rgb[:, :, moving_channel] = np.where(~mask, moving, 0)
    
    return rgb