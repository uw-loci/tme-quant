# ============================================================
# INTERACTIVE VISUALIZATION
# visualization/interactive_viewer.py
# ============================================================

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def view_registration_napari(
    fixed: np.ndarray,
    moving: np.ndarray,
    registered: Optional[np.ndarray] = None,
    layer_names: Tuple[str, str, str] = ("Fixed", "Moving", "Registered")
):
    """
    View registration results in Napari.
    
    Args:
        fixed: Fixed image
        moving: Moving image (original)
        registered: Registered image (optional)
        layer_names: Names for layers
    """
    try:
        import napari
    except ImportError:
        raise ImportError("Napari required: pip install napari[all]")
    
    # Create viewer
    viewer = napari.Viewer()
    
    # Add fixed image
    viewer.add_image(fixed, name=layer_names[0], colormap='gray', opacity=0.7)
    
    # Add moving image
    viewer.add_image(moving, name=layer_names[1], colormap='red', opacity=0.5)
    
    # Add registered image if provided
    if registered is not None:
        viewer.add_image(registered, name=layer_names[2], colormap='green', opacity=0.5)
    
    # Run viewer
    napari.run()


def plot_registration_comparison(
    fixed: np.ndarray,
    moving_before: np.ndarray,
    moving_after: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Create matplotlib figure comparing before/after registration.
    
    Args:
        fixed: Fixed image
        moving_before: Moving image before registration
        moving_after: Moving image after registration
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(fixed, cmap='gray')
    axes[0, 0].set_title('Fixed (Reference)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(moving_before, cmap='gray')
    axes[0, 1].set_title('Moving (Before)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(moving_after, cmap='gray')
    axes[0, 2].set_title('Moving (After Registration)')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    overlay_before = create_rgb_overlay(fixed, moving_before)
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title('Overlay Before (Magenta+Green)')
    axes[1, 0].axis('off')
    
    overlay_after = create_rgb_overlay(fixed, moving_after)
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title('Overlay After (Magenta+Green)')
    axes[1, 1].axis('off')
    
    checker = create_checkerboard(fixed, moving_after, num_squares=8)
    axes[1, 2].imshow(checker, cmap='gray')
    axes[1, 2].set_title('Checkerboard')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Export all functions
__all__ = [
    # Checkerboard
    'create_checkerboard',
    'create_checkerboard_rgb',
    
    # Overlay
    'create_overlay',
    'create_rgb_overlay',
    'create_side_by_side',
    
    # Difference
    'compute_difference_map',
    'create_difference_overlay',
    'compute_registration_quality_map',
    
    # Interactive
    'view_registration_napari',
    'plot_registration_comparison',
]


print("✅ Visualization utilities complete")
print("  - Checkerboard (2 functions)")
print("  - Overlay (3 functions)")
print("  - Difference maps (3 functions)")
print("  - Interactive viewers (2 functions)")
