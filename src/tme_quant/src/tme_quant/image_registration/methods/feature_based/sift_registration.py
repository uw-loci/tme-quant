# ============================================================
# SIFT REGISTRATION
# Location: methods/feature_based/sift_registration.py
# ============================================================

import importlib
import time
from typing import Any, cast

import numpy as np

from ..base_registration import BaseRegistration
from ...config.registration_params import (
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

class SIFTRegistration(BaseRegistration):
    """
    SIFT (Scale-Invariant Feature Transform) registration.
    
    Detects keypoints, computes descriptors, matches features,
    estimates transformation using RANSAC.
    
    Good for images with distinct features and texture.
    
    Example:
        >>> from tme_quant.image_registration.methods.feature_based import SIFTRegistration
        >>> 
        >>> registration = SIFTRegistration(verbose=True)
        >>> result = registration.register(fixed, moving, params)
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.method_name = "sift"
    
    def register(self, fixed_image, moving_image, params):
        """
        Register using SIFT features.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            params: Registration parameters
            
        Returns:
            RegistrationResult
        """
        self.validate_images(fixed_image, moving_image)
        
        if self.verbose:
            print("Starting SIFT registration")
        
        start_time = time.time()
        
        # Convert to grayscale and uint8
        fixed_gray = self._prepare_image(fixed_image)
        moving_gray = self._prepare_image(moving_image)
        
        # Detect SIFT features
        kp1, des1 = self._detect_sift(fixed_gray)
        kp2, des2 = self._detect_sift(moving_gray)
        
        if self.verbose:
            print(f"  ✓ Detected {len(kp1)} and {len(kp2)} keypoints")
        
        # Match features
        matches = self._match_features(des1, des2)
        
        if self.verbose:
            print(f"  ✓ Found {len(matches)} matches")
        
        if len(matches) < 4:
            raise ValueError("Not enough matches found for registration")
        
        # Extract matched points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate transformation
        from ...config.registration_params import TransformType
        
        if params.transform_type == TransformType.AFFINE:
            M, mask = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            transform_matrix = np.vstack([M, [0, 0, 1]])
        else:
            # Homography for more general transform
            M, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            transform_matrix = M
        
        # Apply transformation
        h, w = fixed_gray.shape
        if moving_image.ndim == 3:
            registered_image = cv2.warpAffine(moving_image, M, (w, h))
        else:
            registered_image = cv2.warpAffine(moving_gray, M, (w, h))
        
        # Quality metric (inlier ratio)
        inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0.0
        
        if self.verbose:
            print(f"  ✓ SIFT complete")
            print(f"  Inlier ratio: {inlier_ratio:.2%}")
            print(f"  Time: {time.time() - start_time:.2f}s")
        
        # Create result
        from ...config.registration_params import RegistrationResult, Transform, RegistrationMethod
        
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
            method=RegistrationMethod.SIFT,
            transform_type=params.transform_type
        )
        
        return result
    
    def _prepare_image(self, image):
        """Convert to grayscale uint8."""
        gray = self.ensure_grayscale(image)
        
        # Convert to uint8 if needed
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
        
        return gray
    
    def _detect_sift(self, image):
        """Detect SIFT keypoints and descriptors."""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def _match_features(self, des1, des2):
        """Match features using FLANN."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches