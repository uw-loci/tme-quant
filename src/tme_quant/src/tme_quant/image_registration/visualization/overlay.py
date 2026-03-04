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
# OVERLAY VISUALIZATION
# visualization/overlay.py
# ============================================================

def create_overlay(
    fixed: np.ndarray,
    moving: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create alpha-blended overlay.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        alpha: Blending factor (0=fixed only, 1=moving only)
        
    Returns:
        Blended overlay
    """
    # Ensure same size
    if fixed.shape != moving.shape:
        raise ValueError("Images must have same shape")
    
    # Normalize to [0, 1]
    fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving_norm = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    
    # Blend
    overlay = (1 - alpha) * fixed_norm + alpha * moving_norm
    
    return overlay


def create_rgb_overlay(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_color: str = 'magenta',
    moving_color: str = 'green'
) -> np.ndarray:
    """
    Create RGB overlay with different colors for each image.
    
    Common combinations:
        - Magenta (fixed) + Green (moving) - good for distinguishing
        - Red (fixed) + Cyan (moving)
    
    Args:
        fixed: Fixed image
        moving: Moving image
        fixed_color: Color for fixed ('magenta', 'red', 'green', 'blue', 'cyan', 'yellow')
        moving_color: Color for moving
        
    Returns:
        RGB overlay image
    """
    # Convert to grayscale if needed
    if fixed.ndim == 3:
        fixed = np.mean(fixed, axis=2)
    if moving.ndim == 3:
        moving = np.mean(moving, axis=2)
    
    # Normalize
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    
    # Create RGB image
    h, w = fixed.shape
    rgb = np.zeros((h, w, 3))
    
    # Color definitions (R, G, B)
    colors = {
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'magenta': (1, 0, 1),
        'cyan': (0, 1, 1),
        'yellow': (1, 1, 0),
    }
    
    fixed_rgb = colors.get(fixed_color, (1, 0, 1))  # Default magenta
    moving_rgb = colors.get(moving_color, (0, 1, 0))  # Default green
    
    # Apply colors
    for c in range(3):
        rgb[:, :, c] = fixed * fixed_rgb[c] + moving * moving_rgb[c]
    
    # Clip to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def create_side_by_side(
    fixed: np.ndarray,
    moving: np.ndarray,
    gap: int = 10
) -> np.ndarray:
    """
    Create side-by-side comparison.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        gap: Gap between images in pixels
        
    Returns:
        Side-by-side composite
    """
    # Ensure same height
    if fixed.shape[0] != moving.shape[0]:
        # Resize to match heights
        from skimage import transform
        target_height = min(fixed.shape[0], moving.shape[0])
        
        if fixed.ndim == 2:
            fixed = transform.resize(fixed, (target_height, fixed.shape[1]))
            moving = transform.resize(moving, (target_height, moving.shape[1]))
        else:
            fixed = transform.resize(fixed, (target_height, fixed.shape[1], fixed.shape[2]))
            moving = transform.resize(moving, (target_height, moving.shape[1], moving.shape[2]))
    
    # Create gap
    if fixed.ndim == 2:
        gap_array = np.ones((fixed.shape[0], gap)) * np.mean(fixed)
        composite = np.concatenate([fixed, gap_array, moving], axis=1)
    else:
        gap_array = np.ones((fixed.shape[0], gap, fixed.shape[2])) * np.mean(fixed)
        composite = np.concatenate([fixed, gap_array, moving], axis=1)
    
    return composite