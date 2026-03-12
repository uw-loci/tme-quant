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
# MANUAL LANDMARK REGISTRATION
# Location: methods/landmark_based/manual_landmarks.py
# ============================================================

class ManualLandmarkRegistration:
    """
    Registration using manually selected corresponding points.
    
    Uses thin-plate spline (TPS) for non-rigid warping or
    affine transformation for rigid alignment.
    
    Example:
        >>> from tme_quant.image_registration.methods.landmark_based import ManualLandmarkRegistration
        >>> 
        >>> # Fixed landmarks: [[x1, y1], [x2, y2], ...]
        >>> fixed_lm = np.array([[100, 100], [200, 150], [150, 200]])
        >>> moving_lm = np.array([[105, 98], [205, 148], [155, 198]])
        >>> 
        >>> registration = ManualLandmarkRegistration()
        >>> result = registration.register(fixed, moving, fixed_lm, moving_lm, params)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.method_name = "manual_landmarks"
    
    def register(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_landmarks: np.ndarray,
        moving_landmarks: np.ndarray,
        params
    ):
        """
        Register using manual landmarks.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            fixed_landmarks: Landmarks in fixed image (n, 2)
            moving_landmarks: Corresponding landmarks in moving image (n, 2)
            params: Registration parameters
            
        Returns:
            RegistrationResult
        """
        if self.verbose:
            print(f"Starting landmark-based registration with {len(fixed_landmarks)} points")
        
        # Validate landmarks
        if len(fixed_landmarks) != len(moving_landmarks):
            raise ValueError("Number of landmarks must match")
        
        if len(fixed_landmarks) < 3:
            raise ValueError("At least 3 landmark pairs required")
        
        # Compute transformation
        from ...config.registration_params import TransformType
        
        if params.transform_type == TransformType.AFFINE:
            transform_matrix = self._estimate_affine(moving_landmarks, fixed_landmarks)
            registered_image = self._apply_affine(moving_image, transform_matrix)
        
        else:  # Non-rigid (TPS)
            tps = ThinPlateSpline()
            tps.fit(moving_landmarks, fixed_landmarks)
            registered_image = tps.warp_image(moving_image, fixed_image.shape)
            transform_matrix = np.eye(3)  # No simple matrix for TPS
        
        # Compute target registration error (TRE)
        tre = self._compute_tre(
            fixed_landmarks,
            moving_landmarks,
            transform_matrix if params.transform_type == TransformType.AFFINE else None
        )
        
        if self.verbose:
            print(f"  ✓ Registration complete")
            print(f"  Target Registration Error: {tre:.3f} pixels")
        
        # Create result
        from ...config.registration_params import RegistrationResult, Transform, RegistrationMethod
        
        result = RegistrationResult(
            transform=Transform(
                transform_type=params.transform_type,
                matrix=transform_matrix
            ),
            registered_image=registered_image,
            landmarks_fixed=fixed_landmarks,
            landmarks_moving=moving_landmarks,
            target_registration_error=tre,
            converged=True,
            method=RegistrationMethod.MANUAL_LANDMARKS
        )
        
        return result
    
    def _estimate_affine(self, src_pts, dst_pts):
        """Estimate affine transformation from point pairs."""
        import cv2
        
        M = cv2.estimateAffinePartial2D(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32)
        )[0]
        
        # Convert to 3x3
        transform_matrix = np.vstack([M, [0, 0, 1]])
        
        return transform_matrix
    
    def _apply_affine(self, image, transform_matrix):
        """Apply affine transformation to image."""
        import cv2
        
        h, w = image.shape[:2]
        M = transform_matrix[:2, :]
        
        if image.ndim == 2:
            warped = cv2.warpAffine(image, M, (w, h))
        else:
            warped = cv2.warpAffine(image, M, (w, h))
        
        return warped
    
    def _compute_tre(self, fixed_lm, moving_lm, transform_matrix):
        """Compute target registration error."""
        if transform_matrix is not None:
            # Transform moving landmarks
            moving_hom = np.hstack([moving_lm, np.ones((len(moving_lm), 1))])
            transformed = (transform_matrix @ moving_hom.T).T[:, :2]
            
            # Compute error
            errors = np.linalg.norm(fixed_lm - transformed, axis=1)
            tre = np.mean(errors)
        else:
            # For non-rigid, approximate TRE
            tre = np.mean(np.linalg.norm(fixed_lm - moving_lm, axis=1))
        
        return tre