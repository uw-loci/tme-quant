"""
Normalized Cross-Correlation registration.
"""

import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

from ..base_registration import BaseRegistration
from ...config.registration_params import (
    RegistrationParams,
    RegistrationResult,
    Transform,
    TransformType,
    RegistrationMethod
)


# ============================================================
# FILE 2: methods/intensity_based/cross_correlation.py
# ============================================================

class CrossCorrelationRegistration:
    """
    Normalized Cross-Correlation (NCC) registration.
    
    Fast registration method using phase correlation for sub-pixel accuracy.
    Works well for images with similar intensity distributions.
    
    Example:
        >>> from tme_quant.image_registration.methods.intensity_based import CrossCorrelationRegistration
        >>> 
        >>> registration = CrossCorrelationRegistration(verbose=True)
        >>> result = registration.register(fixed_image, moving_image, params)
        >>> 
        >>> registered_image = result.registered_image
        >>> ncc_score = result.normalized_cross_correlation
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize NCC registration.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.method_name = "cross_correlation"
    
    def register(self, fixed_image, moving_image, params=None):
        """
        Register using normalized cross-correlation.
        
        Uses phase correlation to detect translation with sub-pixel accuracy.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            params: Registration parameters (optional)
            
        Returns:
            RegistrationResult
        """
        if self.verbose:
            print("Starting Cross-Correlation registration")
        
        # Convert to grayscale if needed
        fixed_gray = self._ensure_grayscale(fixed_image)
        moving_gray = self._ensure_grayscale(moving_image)
        
        # Normalize images
        fixed_norm = self._normalize_image(fixed_gray)
        moving_norm = self._normalize_image(moving_gray)
        
        # Compute shift using phase correlation
        # upsample_factor=100 gives sub-pixel accuracy
        shift_vector, error, diffphase = phase_cross_correlation(
            fixed_norm,
            moving_norm,
            upsample_factor=100
        )
        
        if self.verbose:
            print(f"  ✓ Detected shift: [{shift_vector[1]:.2f}, {shift_vector[0]:.2f}] pixels")
            print(f"  Phase correlation error: {error:.6f}")
        
        # Apply shift to image
        if moving_image.ndim == 2:
            # Grayscale
            registered_image = shift(moving_image, shift_vector, order=1)
        else:
            # RGB - shift each channel
            registered_image = np.zeros_like(moving_image)
            for c in range(moving_image.shape[2]):
                registered_image[:, :, c] = shift(
                    moving_image[:, :, c],
                    shift_vector,
                    order=1
                )
        
        # Create transformation matrix (translation only)
        transform_matrix = np.array([
            [1, 0, shift_vector[1]],  # tx
            [0, 1, shift_vector[0]],  # ty
            [0, 0, 1]
        ], dtype=float)
        
        # Compute normalized cross-correlation as quality metric
        ncc = self._compute_ncc(fixed_norm, registered_image)
        
        if self.verbose:
            print(f"  ✓ Registration complete")
            print(f"  Normalized Cross-Correlation: {ncc:.4f}")
        
        # Create result
        class Result:
            def __init__(self):
                self.registered_image = registered_image
                self.transform_matrix = transform_matrix
                self.shift_vector = shift_vector
                self.normalized_cross_correlation = ncc
                self.final_metric_value = ncc
                self.phase_error = error
                self.method = "cross_correlation"
                self.converged = True
        
        return Result()
    
    def _ensure_grayscale(self, image):
        """Convert to grayscale if needed."""
        if image.ndim == 2:
            return image
        elif image.ndim == 3:
            # RGB to grayscale
            return np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return image
    
    def _normalize_image(self, image):
        """Normalize image to [0, 1]."""
        image = image.astype(float)
        
        # Robust normalization using percentiles
        p2, p98 = np.percentile(image, (2, 98))
        
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        
        return np.clip(image, 0, 1)
    
    def _compute_ncc(self, img1, img2):
        """
        Compute normalized cross-correlation.
        
        NCC = mean((img1 - mean(img1)) * (img2 - mean(img2))) / (std(img1) * std(img2))
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            NCC value (range: -1 to 1, higher is better)
        """
        # Ensure same shape
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # Convert to grayscale if needed
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2)
        
        # Normalize
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
        
        # Compute NCC
        ncc = np.mean(img1_norm * img2_norm)
        
        return float(ncc)