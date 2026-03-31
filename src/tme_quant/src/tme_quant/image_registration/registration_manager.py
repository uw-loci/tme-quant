"""
Main coordinator for image registration.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

from .config import (
    RegistrationParams,
    RegistrationMethod,
    RegistrationResult,
    Transform
)


class RegistrationManager:
    """
    Main class for microscopy image registration.
    
    Coordinates registration between cell and fiber images across
    different microscopy modalities.
    
    Example:
        >>> from tme_quant.image_registration import RegistrationManager
        >>> from tme_quant.image_registration.config import RegistrationParams, RegistrationMethod
        >>> 
        >>> manager = RegistrationManager()
        >>> 
        >>> # Register H&E to SHG
        >>> params = RegistrationParams(
        ...     method=RegistrationMethod.HE_SHG,
        ...     transform_type=TransformType.AFFINE
        ... )
        >>> 
        >>> result = manager.register(
        ...     fixed_image=shg_image,
        ...     moving_image=he_image,
        ...     params=params
        ... )
        >>> 
        >>> registered_he = result.registered_image
        >>> transform_matrix = result.transform.matrix
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize registration manager.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.result: Optional[RegistrationResult] = None
    
    def register(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        params: Optional[RegistrationParams] = None,
        fixed_landmarks: Optional[np.ndarray] = None,
        moving_landmarks: Optional[np.ndarray] = None
    ) -> RegistrationResult:
        """
        Register moving image to fixed image.
        
        Args:
            fixed_image: Target/reference image (e.g., SHG)
            moving_image: Image to be registered (e.g., H&E)
            params: Registration parameters
            fixed_landmarks: Landmarks in fixed image (optional)
            moving_landmarks: Landmarks in moving image (optional)
            
        Returns:
            RegistrationResult with transform and registered image
        """
        if params is None:
            params = RegistrationParams()
        
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Image Registration")
            print(f"Method: {params.method.value}")
            print(f"Transform: {params.transform_type.value}")
            print(f"{'='*60}")
        
        # Preprocessing
        fixed_prep, moving_prep = self._preprocess_images(
            fixed_image, moving_image, params
        )
        
        # Route to appropriate method
        if params.method == RegistrationMethod.MUTUAL_INFORMATION:
            result = self._register_mutual_information(
                fixed_prep, moving_prep, params
            )
        
        elif params.method == RegistrationMethod.CROSS_CORRELATION:
            result = self._register_cross_correlation(
                fixed_prep, moving_prep, params
            )
        
        elif params.method == RegistrationMethod.SIFT:
            result = self._register_sift(
                fixed_prep, moving_prep, params
            )
        
        elif params.method == RegistrationMethod.ORB:
            result = self._register_orb(
                fixed_prep, moving_prep, params
            )
        
        elif params.method == RegistrationMethod.MANUAL_LANDMARKS:
            if fixed_landmarks is None or moving_landmarks is None:
                raise ValueError("Landmarks required for manual landmark registration")
            result = self._register_manual_landmarks(
                fixed_prep, moving_prep, fixed_landmarks, moving_landmarks, params
            )
        
        elif params.method == RegistrationMethod.HE_SHG:
            result = self._register_he_shg(
                fixed_prep, moving_prep, params
            )
        
        elif params.method == RegistrationMethod.COMIR:
            result = self._register_comir(
                fixed_prep, moving_prep, params
            )
        
        else:
            raise ValueError(f"Unknown registration method: {params.method}")
        
        # Add timing
        result.optimization_time = time.time() - start_time
        
        # Store result
        self.result = result
        
        if self.verbose:
            print(f"\nRegistration complete in {result.optimization_time:.2f}s")
            print(f"Final metric: {result.final_metric_value:.6f}")
            print(f"{'='*60}\n")
        
        return result
    
    def _preprocess_images(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        params: RegistrationParams
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess images before registration."""
        from .preprocessing import (
            rgb_to_grayscale,
            normalize_image,
            match_histograms,
        )
        
        # Convert to grayscale if needed
        if fixed.ndim == 3:
            fixed = rgb_to_grayscale(fixed)
        if moving.ndim == 3:
            moving = rgb_to_grayscale(moving)
        
        # Normalize intensities
        if params.normalize_intensity:
            fixed = normalize_image(fixed)
            moving = normalize_image(moving)
        
        # Histogram matching
        if params.histogram_matching:
            moving = match_histograms(moving, fixed)
        
        return fixed, moving
    
    # Registration method implementations will be in separate method classes
    # These are placeholder stubs that delegate to method-specific classes
    
    def _register_mutual_information(self, fixed, moving, params):
        """Register using mutual information (delegates to MI class)."""
        from .methods.intensity_based import MutualInformationRegistration
        method = MutualInformationRegistration()
        return method.register(fixed, moving, params)
    
    def _register_cross_correlation(self, fixed, moving, params):
        """Register using normalized cross-correlation."""
        from .methods.intensity_based import CrossCorrelationRegistration
        method = CrossCorrelationRegistration()
        return method.register(fixed, moving, params)
    
    def _register_sift(self, fixed, moving, params):
        """Register using SIFT features."""
        from .methods.feature_based import SIFTRegistration
        method = SIFTRegistration()
        return method.register(fixed, moving, params)
    
    def _register_orb(self, fixed, moving, params):
        """Register using ORB features."""
        from .methods.feature_based import ORBRegistration
        method = ORBRegistration()
        return method.register(fixed, moving, params)
    
    def _register_manual_landmarks(self, fixed, moving, fixed_lm, moving_lm, params):
        """Register using manual landmarks."""
        from .methods.landmark_based import ManualLandmarkRegistration
        method = ManualLandmarkRegistration()
        return method.register(fixed, moving, fixed_lm, moving_lm, params)
    
    def _register_he_shg(self, fixed, moving, params):
        """Register H&E to SHG using Bredfeldt method."""
        from .methods.specialized import HESHGRegistration
        method = HESHGRegistration()
        return method.register(fixed, moving, params)
    
    def _register_comir(self, fixed, moving, params):
        """Register using CoMIR deep learning method."""
        from .methods.deep_learning import CoMIRRegistration
        method = CoMIRRegistration()
        return method.register(fixed, moving, params)
    
    def get_result(self) -> Optional[RegistrationResult]:
        """Get the last registration result."""
        return self.result