"""
Image preprocessing utilities for cell segmentation.
"""

import numpy as np
from typing import Optional, Tuple
from skimage import exposure, filters


def preprocess_image(
    image: np.ndarray,
    normalize: bool = True,
    denoise: bool = False,
    enhance_contrast: bool = False,
    channel: Optional[int] = None
) -> np.ndarray:
    """
    Preprocess image for cell segmentation.
    
    Args:
        image: Input image
        normalize: Normalize intensity to 0-1
        denoise: Apply denoising
        enhance_contrast: Apply contrast enhancement
        channel: Channel to use (for multi-channel images)
        
    Returns:
        Preprocessed image
    """
    # Select channel if multi-channel
    if image.ndim == 3 and channel is not None:
        image = image[:, :, channel]
    
    # Convert to float
    image = image.astype(np.float32)
    
    # Normalize
    if normalize:
        image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Denoise
    if denoise:
        from skimage.restoration import denoise_nl_means
        image = denoise_nl_means(image, patch_size=5, patch_distance=7, h=0.1)
    
    # Enhance contrast
    if enhance_contrast:
        p2, p98 = np.percentile(image, (2, 98))
        image = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    return image