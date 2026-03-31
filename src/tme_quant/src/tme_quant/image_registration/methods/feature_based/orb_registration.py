# ============================================================
# ORB REGISTRATION
# Location: methods/feature_based/orb_registration.py
# ============================================================

import importlib
import time
from typing import Any, cast

import numpy as np

from ..base_registration import BaseRegistration
from ...config import (
    RegistrationMethod,
    RegistrationResult,
    Transform,
    TransformType,
)

cv2: Any

try:
    cv2 = importlib.import_module("cv2")
except ImportError:
    cv2 = cast(Any, None)

class ORBRegistration(BaseRegistration):
    """
    ORB (Oriented FAST and Rotated BRIEF) registration.
    
    Faster than SIFT, uses binary descriptors.
    Good for real-time applications.
    
    Example:
        >>> from tme_quant.image_registration.methods.feature_based import ORBRegistration
        >>> 
        >>> registration = ORBRegistration(verbose=True)
        >>> result = registration.register(fixed, moving, params)
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.method_name = "orb"
    
    def register(self, fixed_image, moving_image, params):
        """
        Register using ORB features.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            params: Registration parameters
            
        Returns:
            RegistrationResult
        """
        self.validate_images(fixed_image, moving_image)
        
        if self.verbose:
            print("Starting ORB registration")
        
        start_time = time.time()
        
        # Convert to grayscale and uint8
        fixed_gray = self._prepare_image(fixed_image)
        moving_gray = self._prepare_image(moving_image)
        
        # Detect ORB features
        kp1, des1 = self._detect_orb(fixed_gray)
        kp2, des2 = self._detect_orb(moving_gray)
        
        if self.verbose:
            print(f"  ✓ Detected {len(kp1)} and {len(kp2)} keypoints")
        
        # Match features
        matches = self._match_orb_features(des1, des2)
        
        if self.verbose:
            print(f"  ✓ Found {len(matches)} matches")
        
        if len(matches) < 4:
            raise ValueError("Not enough matches found for registration")
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate transformation
        from ...config import TransformType
        
        if params.transform_type == TransformType.AFFINE:
            M, mask = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            transform_matrix = np.vstack([M, [0, 0, 1]])
        else:
            M, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC
            )
            transform_matrix = M
        
        # Apply transformation
        h, w = fixed_gray.shape
        if moving_image.ndim == 3:
            registered_image = cv2.warpAffine(moving_image, M, (w, h))
        else:
            registered_image = cv2.warpAffine(moving_gray, M, (w, h))
        
        # Quality metric
        inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0.0
        
        if self.verbose:
            print(f"  ✓ ORB complete")
            print(f"  Inlier ratio: {inlier_ratio:.2%}")
            print(f"  Time: {time.time() - start_time:.2f}s")
        
        # Create result
        from ...config import RegistrationResult, Transform, RegistrationMethod
        
        result = RegistrationResult(
            transform=Transform(
                transform_type=params.transform_type,
                matrix=transform_matrix
            ),
            registered_image=registered_image,
            final_metric_value=inlier_ratio,
            num_iterations=1,
            converged=True,
            optimization_time=time.time() - start_time,
            method=RegistrationMethod.ORB,
            transform_type=params.transform_type
        )
        
        return result
    
    def _prepare_image(self, image):
        """Convert to grayscale uint8."""
        gray = self.ensure_grayscale(image)
        
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
        
        return gray
    
    def _detect_orb(self, image, n_features=5000):
        """Detect ORB keypoints and descriptors."""
        orb = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def _match_orb_features(self, des1, des2):
        """Match ORB features using BFMatcher."""
        # Brute-force matcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches