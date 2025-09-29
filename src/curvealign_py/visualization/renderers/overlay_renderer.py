"""
Overlay rendering for curvelet visualization.

This module provides functions for creating overlay visualizations
that show curvelets superimposed on the original image.
"""

from typing import Sequence, Optional
import numpy as np

from ...types import Curvelet


def create_fiber_overlay(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    mask: Optional[np.ndarray] = None,
    colormap: str = "hsv",
    line_width: float = 2.0,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Create an overlay image showing curvelets on detected fiber regions.
    
    Parameters
    ----------
    image : np.ndarray
        Original grayscale image
    curvelets : Sequence[Curvelet]
        List of curvelets to overlay
    mask : np.ndarray, optional
        Optional mask to apply to the overlay
    colormap : str, default "hsv"
        Colormap for angle visualization
    line_width : float, default 2.0
        Width of curvelet lines
    alpha : float, default 0.7
        Transparency of overlay
        
    Returns
    -------
    np.ndarray
        RGB overlay image with curvelets drawn only where fibers are detected
    """
    height, width = image.shape
    
    # Normalize image
    image_normalized = _normalize_image(image)
    
    # Create RGB background
    background_rgb = np.stack([image_normalized, image_normalized, image_normalized], axis=2)
    
    if not curvelets:
        return (background_rgb * 255).astype(np.uint8)
    
    # Create fiber overlay
    fiber_overlay = np.zeros((height, width, 3), dtype=np.float32)
    
    for curvelet in curvelets:
        draw_thick_curvelet_line(
            fiber_overlay,
            curvelet,
            line_width=int(line_width),
            colormap=colormap
        )
    
    # Apply mask if provided
    if mask is not None:
        mask_3d = np.stack([mask, mask, mask], axis=2).astype(bool)
        fiber_overlay = fiber_overlay * mask_3d
    
    # Blend only where fibers are detected
    fiber_mask = np.any(fiber_overlay > 0, axis=2)
    
    # Create final overlay
    result_rgb = background_rgb.copy()
    result_rgb[fiber_mask] = (
        alpha * fiber_overlay[fiber_mask] + 
        (1 - alpha) * background_rgb[fiber_mask]
    )
    
    return (np.clip(result_rgb, 0, 1) * 255).astype(np.uint8)


def draw_thick_curvelet_line(
    overlay: np.ndarray,
    curvelet: Curvelet,
    line_width: int = 2,
    colormap: str = "hsv"
) -> None:
    """
    Draw a thick colored line representing a curvelet.
    
    Parameters
    ----------
    overlay : np.ndarray
        RGB overlay array to draw on (modified in-place)
    curvelet : Curvelet
        Curvelet to draw
    line_width : int, default 2
        Width of the line in pixels
    colormap : str, default "hsv"
        Colormap for angle-based coloring
    """
    import matplotlib.pyplot as plt
    
    # Get angle-based color
    color = _angle_to_color(curvelet.angle_deg, colormap)
    
    # Calculate line endpoints
    half_length = line_width * 2
    angle_rad = np.radians(curvelet.angle_deg)
    
    dx = half_length * np.cos(angle_rad)
    dy = half_length * np.sin(angle_rad)
    
    x1 = curvelet.center_col - dx
    y1 = curvelet.center_row - dy
    x2 = curvelet.center_col + dx
    y2 = curvelet.center_row + dy
    
    # Use Bresenham's line algorithm for thick lines
    _draw_thick_line(overlay, int(x1), int(y1), int(x2), int(y2), color, line_width)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    if image.dtype not in [np.float32, np.float64]:
        image_normalized = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
    else:
        image_normalized = image.astype(np.float32)
        if image_normalized.max() > 1:
            image_normalized = image_normalized / image_normalized.max()
    
    return image_normalized


def _angle_to_color(angle_deg: float, colormap: str = "hsv") -> np.ndarray:
    """Convert angle to RGB color using specified colormap."""
    import matplotlib.pyplot as plt
    
    # Normalize angle to [0, 1]
    normalized_angle = (angle_deg % 180) / 180.0
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    color_rgba = cmap(normalized_angle)
    
    return np.array(color_rgba[:3])


def _draw_thick_line(overlay: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                    color: np.ndarray, thickness: int) -> None:
    """Draw a thick line using Bresenham's algorithm with thickness."""
    height, width = overlay.shape[:2]
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        # Draw thick pixel at (x, y)
        for offset_x in range(-thickness//2, thickness//2 + 1):
            for offset_y in range(-thickness//2, thickness//2 + 1):
                px = x + offset_x
                py = y + offset_y
                
                if 0 <= px < width and 0 <= py < height:
                    overlay[py, px] = color
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
