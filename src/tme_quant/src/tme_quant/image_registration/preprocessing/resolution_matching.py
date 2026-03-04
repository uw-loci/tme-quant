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


# Export all functions
__all__ = [
    # Intensity normalization
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


print("✅ Preprocessing utilities complete")
print("  - Intensity normalization (5 functions)")
print("  - Modality conversion (5 functions)")
print("  - Resolution matching (6 functions)")