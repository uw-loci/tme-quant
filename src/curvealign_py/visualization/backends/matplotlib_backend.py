"""
Matplotlib-based visualization backend.

This module provides the default visualization backend using matplotlib,
suitable for basic visualization needs without additional dependencies.
"""

from typing import Sequence, Optional, Tuple
import numpy as np

from ...types import Curvelet
from ..renderers import create_fiber_overlay, create_angle_maps


def create_overlay(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    mask: Optional[np.ndarray] = None,
    colormap: str = "hsv",
    line_width: float = 2.0,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Create an overlay image showing curvelets on the original image.
    
    This is a wrapper around the overlay renderer that provides the
    public API for the matplotlib backend.
    
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
        RGB overlay image
    """
    return create_fiber_overlay(
        image=image,
        curvelets=curvelets,
        mask=mask,
        colormap=colormap,
        line_width=line_width,
        alpha=alpha
    )


def create_angle_maps_backend(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    std_window: int = 24,
    square_window: int = 12,
    gaussian_sigma: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create raw and processed angle maps from curvelets.
    
    This is a wrapper around the angle map renderer that provides the
    public API for the matplotlib backend.
    
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
    from ..renderers import create_angle_maps
    return create_angle_maps(
        image=image,
        curvelets=curvelets,
        std_window=std_window,
        square_window=square_window,
        gaussian_sigma=gaussian_sigma
    )


def plot_results(
    image: np.ndarray,
    curvelets: Sequence[Curvelet],
    show_overlay: bool = True,
    show_angle_map: bool = True,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Create a matplotlib figure showing analysis results.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    curvelets : Sequence[Curvelet]
        Analysis results
    show_overlay : bool, default True
        Whether to show overlay plot
    show_angle_map : bool, default True
        Whether to show angle map plot
    figsize : Tuple[int, int], default (12, 4)
        Figure size in inches
    """
    import matplotlib.pyplot as plt
    
    n_plots = 1 + int(show_overlay) + int(show_angle_map)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Original image
    axes[plot_idx].imshow(image, cmap='gray')
    axes[plot_idx].set_title('Original Image')
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Overlay
    if show_overlay:
        overlay = create_overlay(image, curvelets)
        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title(f'Fiber Overlay ({len(curvelets)} curvelets)')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Angle map
    if show_angle_map:
        _, processed_map = create_angle_maps(image, curvelets)
        im = axes[plot_idx].imshow(processed_map, cmap='hsv', vmin=0, vmax=180)
        axes[plot_idx].set_title('Angle Map')
        axes[plot_idx].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[plot_idx], shrink=0.6)
        cbar.set_label('Angle (degrees)')
    
    plt.tight_layout()
    plt.show()
