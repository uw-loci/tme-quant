"""
Preprocessing utilities for image registration.

Covers intensity normalisation, modality conversion, and resolution matching.
Merged from the former preprocessing/ sub-package.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from skimage import exposure, transform




# ============================================================
# INTENSITY NORMALISATION
# ============================================================

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


# ============================================================
# MODALITY CONVERSION
# ============================================================

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

# ============================================================
# RESOLUTION MATCHING
# ============================================================

# ============================================================
# RESOLUTION MATCHING
# preprocessing/resolution_matching.py
# ============================================================

def match_resolution(
    image: np.ndarray,
    target_shape: Tuple[int, int],
    order: int = 1
) -> np.ndarray:
    """
    Resize image to match target shape.
    
    Args:
        image: Input image
        target_shape: Target (height, width)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Resized image
    """
    if image.ndim == 2:
        # Grayscale
        resized = transform.resize(
            image,
            target_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=order > 0
        )
    elif image.ndim == 3:
        # Multi-channel
        resized = transform.resize(
            image,
            target_shape + (image.shape[2],),
            order=order,
            preserve_range=True,
            anti_aliasing=order > 0
        )
    else:
        raise ValueError(f"Unexpected image dimensions: {image.ndim}")
    
    return resized.astype(image.dtype)


def match_pixel_size(
    image: np.ndarray,
    source_pixel_size: float,
    target_pixel_size: float,
    order: int = 1
) -> np.ndarray:
    """
    Resize image to match pixel size.
    
    Args:
        image: Input image
        source_pixel_size: Source pixel size (microns/pixel)
        target_pixel_size: Target pixel size (microns/pixel)
        order: Interpolation order
        
    Returns:
        Resized image
    """
    # Calculate scale factor
    scale = source_pixel_size / target_pixel_size
    
    # New shape
    if image.ndim == 2:
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    else:
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    
    return match_resolution(image, new_shape, order)


def downsample(
    image: np.ndarray,
    factor: int,
    anti_alias: bool = True
) -> np.ndarray:
    """
    Downsample image by integer factor.
    
    Args:
        image: Input image
        factor: Downsampling factor (e.g., 2 = half resolution)
        anti_alias: Apply Gaussian smoothing before downsampling
        
    Returns:
        Downsampled image
    """
    if factor == 1:
        return image
    
    if anti_alias:
        # Gaussian smoothing to prevent aliasing
        sigma = factor / 2.0
        if image.ndim == 2:
            smoothed = ndimage.gaussian_filter(image, sigma)
        else:
            smoothed = np.zeros_like(image)
            for c in range(image.shape[2]):
                smoothed[:, :, c] = ndimage.gaussian_filter(image[:, :, c], sigma)
        image = smoothed
    
    # Downsample
    if image.ndim == 2:
        downsampled = image[::factor, ::factor]
    else:
        downsampled = image[::factor, ::factor, :]
    
    return downsampled


def upsample(
    image: np.ndarray,
    factor: int,
    order: int = 1
) -> np.ndarray:
    """
    Upsample image by integer factor.
    
    Args:
        image: Input image
        factor: Upsampling factor (e.g., 2 = double resolution)
        order: Interpolation order
        
    Returns:
        Upsampled image
    """
    if factor == 1:
        return image
    
    # Calculate new shape
    if image.ndim == 2:
        new_shape = (image.shape[0] * factor, image.shape[1] * factor)
    else:
        new_shape = (image.shape[0] * factor, image.shape[1] * factor)
    
    return match_resolution(image, new_shape, order)


def pad_to_same_size(
    image1: np.ndarray,
    image2: np.ndarray,
    mode: str = 'constant',
    constant_value: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad images to same size.
    
    Args:
        image1: First image
        image2: Second image
        mode: Padding mode ('constant', 'edge', 'reflect')
        constant_value: Value for constant padding
        
    Returns:
        Tuple of padded images
    """
    # Get maximum dimensions
    max_h = max(image1.shape[0], image2.shape[0])
    max_w = max(image1.shape[1], image2.shape[1])
    
    def pad_image(img, target_h, target_w):
        pad_h = target_h - img.shape[0]
        pad_w = target_w - img.shape[1]
        
        if img.ndim == 2:
            padding = ((0, pad_h), (0, pad_w))
        else:
            padding = ((0, pad_h), (0, pad_w), (0, 0))
        
        if mode == 'constant':
            padded = np.pad(img, padding, mode=mode, constant_values=constant_value)
        else:
            padded = np.pad(img, padding, mode=mode)
        
        return padded
    
    image1_padded = pad_image(image1, max_h, max_w)
    image2_padded = pad_image(image2, max_h, max_w)
    
    return image1_padded, image2_padded


def crop_to_common_roi(
    image1: np.ndarray,
    image2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop images to their common region (minimum dimensions).
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Tuple of cropped images
    """
    # Get minimum dimensions
    min_h = min(image1.shape[0], image2.shape[0])
    min_w = min(image1.shape[1], image2.shape[1])
    
    # Crop
    image1_cropped = image1[:min_h, :min_w]
    image2_cropped = image2[:min_h, :min_w]
    
    return image1_cropped, image2_cropped



__all__ = [
    # Intensity normalisation
    'normalize_image',
    'normalize_percentile',
    'match_histograms',
    'normalize_zscore',
    'adaptive_histogram_equalization',
    # Modality conversion
    'rgb_to_grayscale',
    'extract_channel',
    'extract_eosin_channel',
    'color_deconvolution_he',
    'ensure_uint8',
    # Resolution matching
    'match_resolution',
    'match_pixel_size',
    'downsample',
    'upsample',
    'pad_to_same_size',
    'crop_to_common_roi',
]
