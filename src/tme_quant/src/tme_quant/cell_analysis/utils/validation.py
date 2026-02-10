"""
Validation utilities for cell analysis.
"""

import numpy as np
from ..config.segmentation_params import SegmentationParams


def validate_image(image: np.ndarray) -> bool:
    """
    Validate input image.
    
    Args:
        image: Input image
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If image is invalid
    """
    if image is None:
        raise ValueError("Image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy array")
    
    if image.ndim not in [2, 3, 4]:
        raise ValueError(f"Image must be 2D, 3D, or 4D, got {image.ndim}D")
    
    if image.size == 0:
        raise ValueError("Image is empty")
    
    return True


def validate_segmentation_params(params: SegmentationParams) -> bool:
    """
    Validate segmentation parameters.
    
    Args:
        params: Segmentation parameters
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    if params.pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    
    if params.min_cell_size < 0:
        raise ValueError("min_cell_size must be non-negative")
    
    if params.max_cell_size <= params.min_cell_size:
        raise ValueError("max_cell_size must be greater than min_cell_size")
    
    return True