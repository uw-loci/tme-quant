"""
Landmark-Based Registration and IO Utilities

Files:
- methods/landmark_based/manual_landmarks.py
- methods/landmark_based/thin_plate_spline.py
- io/transform_io.py
- io/landmark_io.py
- utils/image_utils.py
- utils/transform_utils.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf

# ============================================================
# IMAGE UTILITIES
# Location: utils/image_utils.py
# ============================================================

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if image.ndim == 2:
        return image
    elif image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    elif image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image


def pad_to_same_size(
    image1: np.ndarray,
    image2: np.ndarray,
    mode: str = 'constant'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad images to same size.
    
    Args:
        image1: First image
        image2: Second image
        mode: Padding mode
        
    Returns:
        Tuple of padded images
    """
    max_h = max(image1.shape[0], image2.shape[0])
    max_w = max(image1.shape[1], image2.shape[1])
    
    def pad_image(img):
        pad_h = max_h - img.shape[0]
        pad_w = max_w - img.shape[1]
        
        if img.ndim == 2:
            padding = ((0, pad_h), (0, pad_w))
        else:
            padding = ((0, pad_h), (0, pad_w), (0, 0))
        
        return np.pad(img, padding, mode=mode)
    
    return pad_image(image1), pad_image(image2)


def crop_to_common_roi(
    image1: np.ndarray,
    image2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop images to common region."""
    min_h = min(image1.shape[0], image2.shape[0])
    min_w = min(image1.shape[1], image2.shape[1])
    
    return image1[:min_h, :min_w], image2[:min_h, :min_w]