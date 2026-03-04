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
# MODALITY CONVERSION
# preprocessing/modality_conversion.py
# ============================================================

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Uses standard luminance weights: R*0.299 + G*0.587 + B*0.114
    
    Args:
        image: RGB image
        
    Returns:
        Grayscale image
    """
    if image.ndim == 2:
        return image
    
    if image.ndim == 3:
        if image.shape[2] == 3:
            # Standard RGB to grayscale conversion
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            return gray
        elif image.shape[2] == 1:
            return image[:, :, 0]
    
    return image


def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """
    Extract a specific channel from multi-channel image.
    
    Args:
        image: Multi-channel image
        channel: Channel index (0=Red, 1=Green, 2=Blue for RGB)
        
    Returns:
        Single channel image
    """
    if image.ndim == 2:
        return image
    
    if image.ndim == 3 and channel < image.shape[2]:
        return image[:, :, channel]
    
    raise ValueError(f"Cannot extract channel {channel} from image with shape {image.shape}")


def extract_eosin_channel(he_image: np.ndarray) -> np.ndarray:
    """
    Extract eosin channel from H&E image.
    
    Eosin stains cytoplasm and collagen pink/red.
    Simple method: use red channel.
    
    Args:
        he_image: H&E RGB image
        
    Returns:
        Eosin channel (grayscale)
    """
    if he_image.ndim == 2:
        return he_image
    
    if he_image.ndim == 3:
        # Red channel corresponds to eosin staining
        eosin = he_image[:, :, 0]
        return eosin
    
    raise ValueError(f"Expected 2D or 3D image, got {he_image.ndim}D")


def color_deconvolution_he(
    he_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    H&E color deconvolution using Ruifrok & Johnston method.
    
    Separates hematoxylin (nuclei) and eosin (cytoplasm/collagen) stains.
    
    Args:
        he_image: H&E RGB image
        
    Returns:
        Tuple of (hematoxylin_channel, eosin_channel)
    """
    if he_image.ndim != 3 or he_image.shape[2] != 3:
        raise ValueError("Expected RGB image")
    
    # H&E stain matrix (normalized)
    # From Ruifrok & Johnston (2001)
    he_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin (blue/purple)
        [0.07, 0.99, 0.11],  # Eosin (pink/red)
    ])
    
    # Reshape image
    h, w = he_image.shape[:2]
    rgb = he_image.reshape(-1, 3).astype(float)
    
    # Convert to optical density (Beer-Lambert law)
    # OD = -log10(I / I0)
    rgb = np.maximum(rgb, 1e-6)  # Avoid log(0)
    od = -np.log10(rgb / 255.0)
    
    # Deconvolve: solve for stain concentrations
    stains = np.linalg.lstsq(he_matrix.T, od.T, rcond=None)[0].T
    
    # Extract channels
    hematoxylin = stains[:, 0].reshape(h, w)
    eosin = stains[:, 1].reshape(h, w)
    
    return hematoxylin, eosin


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 format.
    
    Args:
        image: Input image
        
    Returns:
        Image as uint8
    """
    if image.dtype == np.uint8:
        return image
    
    # Normalize to [0, 255]
    if image.max() <= 1.0:
        image = image * 255
    
    # Clip and convert
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image