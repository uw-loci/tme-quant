"""
Preprocessing utilities for image registration.

Location: preprocessing/intensity_normalization.py
          preprocessing/modality_conversion.py
          preprocessing/resolution_matching.py
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import exposure, transform
# ============================================================
# INTENSITY NORMALIZATION
# preprocessing/intensity_normalization.py
# ============================================================

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity to [0, 1] range.
    
    Uses robust percentile-based normalization.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    image = image.astype(float)
    
    # Robust normalization using 2nd and 98th percentiles
    p2, p98 = np.percentile(image, (2, 98))
    
    if p98 > p2:
        image = (image - p2) / (p98 - p2)
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def normalize_percentile(
    image: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0
) -> np.ndarray:
    """
    Normalize using custom percentiles.
    
    Args:
        image: Input image
        p_low: Lower percentile
        p_high: Upper percentile
        
    Returns:
        Normalized image
    """
    image = image.astype(float)
    
    p_low_val, p_high_val = np.percentile(image, (p_low, p_high))
    
    if p_high_val > p_low_val:
        image = (image - p_low_val) / (p_high_val - p_low_val)
    
    return np.clip(image, 0, 1)


def match_histograms(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    Match histogram of source image to reference image.
    
    Useful for making images from different modalities more similar.
    
    Args:
        source: Source image to transform
        reference: Reference image with target histogram
        
    Returns:
        Source image with matched histogram
    """
    # Use scikit-image histogram matching
    matched = exposure.match_histograms(source, reference)
    
    return matched


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalization (zero mean, unit variance).
    
    Args:
        image: Input image
        
    Returns:
        Z-score normalized image
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std > 0:
        normalized = (image - mean) / std
    else:
        normalized = image - mean
    
    return normalized


def adaptive_histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 0.01
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image
        clip_limit: Clipping limit for contrast
        
    Returns:
        Equalized image
    """
    # Normalize to [0, 1] if needed
    if image.max() > 1:
        image = image / image.max()
    
    # Apply CLAHE
    equalized = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    
    return equalized
