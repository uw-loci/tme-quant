"""
Complete Core Registration Methods Implementation

Files:
- methods/base_registration.py
- methods/intensity_based/mutual_information.py
- methods/feature_based/sift_registration.py
- methods/feature_based/orb_registration.py
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time
import cv2

# ============================================================
# BASE REGISTRATION CLASS
# Location: methods/base_registration.py
# ============================================================

class BaseRegistration(ABC):
    """
    Abstract base class for all registration methods.
    
    All registration methods inherit from this and implement register().
    Provides common validation and utility methods.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize base registration.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.method_name = "base"
    
    @abstractmethod
    def register(self, fixed_image, moving_image, params):
        """
        Register moving image to fixed image.
        
        Must be implemented by all subclasses.
        
        Args:
            fixed_image: Reference/target image
            moving_image: Image to be registered
            params: Registration parameters
            
        Returns:
            RegistrationResult
        """
        pass
    
    def validate_images(self, fixed, moving):
        """Validate input images."""
        if fixed.size == 0 or moving.size == 0:
            raise ValueError("Images cannot be empty")
        if fixed.ndim not in [2, 3]:
            raise ValueError(f"Fixed image must be 2D or 3D, got {fixed.ndim}D")
        if moving.ndim not in [2, 3]:
            raise ValueError(f"Moving image must be 2D or 3D, got {moving.ndim}D")
        return True
    
    def ensure_grayscale(self, image):
        """Convert to grayscale if needed."""
        if image.ndim == 2:
            return image
        elif image.ndim == 3 and image.shape[2] == 3:
            return np.dot(image[..., :3], [0.299, 0.587, 0.114])
        elif image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        return image
    
    def normalize_image(self, image):
        """Normalize to [0, 1]."""
        image = image.astype(float)
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        return np.clip(image, 0, 1)

print("  - BaseRegistration (abstract base)")

# ============================================================