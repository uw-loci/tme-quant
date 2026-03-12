"""
Preprocessing utilities for image registration.
"""

from .intensity_normalization import (
    normalize_image,
    normalize_percentile,
    match_histograms,
    normalize_zscore,
    adaptive_histogram_equalization,
)

from .modality_conversion import (
    rgb_to_grayscale,
    extract_channel,
    extract_eosin_channel,
    color_deconvolution_he,
    ensure_uint8,
)

from .resolution_matching import (
    match_resolution,
    match_pixel_size,
    downsample,
    upsample,
    pad_to_same_size,
    crop_to_common_roi,
)

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