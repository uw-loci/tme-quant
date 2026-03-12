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
# THIN-PLATE SPLINE WARPING
# Location: methods/landmark_based/thin_plate_spline.py
# ============================================================

class ThinPlateSpline:
    """
    Thin-plate spline (TPS) warping for non-rigid registration.
    
    Provides smooth interpolation between landmark pairs.
    
    Example:
        >>> tps = ThinPlateSpline()
        >>> tps.fit(source_landmarks, target_landmarks)
        >>> warped_image = tps.warp_image(moving_image, target_shape)
    """
    
    def __init__(self, smoothing: float = 0.0):
        """
        Initialize TPS.
        
        Args:
            smoothing: Regularization parameter (0 = exact interpolation)
        """
        self.smoothing = smoothing
        self.source_landmarks = None
        self.target_landmarks = None
        self.rbf_x = None
        self.rbf_y = None
    
    def fit(self, source_landmarks: np.ndarray, target_landmarks: np.ndarray):
        """
        Fit TPS transformation.
        
        Args:
            source_landmarks: Source points (n, 2)
            target_landmarks: Target points (n, 2)
        """
        self.source_landmarks = source_landmarks
        self.target_landmarks = target_landmarks
        
        # Create RBF interpolators for x and y coordinates
        self.rbf_x = Rbf(
            source_landmarks[:, 0],
            source_landmarks[:, 1],
            target_landmarks[:, 0],
            function='thin_plate',
            smooth=self.smoothing
        )
        
        self.rbf_y = Rbf(
            source_landmarks[:, 0],
            source_landmarks[:, 1],
            target_landmarks[:, 1],
            function='thin_plate',
            smooth=self.smoothing
        )
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using TPS.
        
        Args:
            points: Points to transform (n, 2)
            
        Returns:
            Transformed points (n, 2)
        """
        if self.rbf_x is None:
            raise ValueError("TPS not fitted. Call fit() first.")
        
        transformed_x = self.rbf_x(points[:, 0], points[:, 1])
        transformed_y = self.rbf_y(points[:, 0], points[:, 1])
        
        return np.column_stack([transformed_x, transformed_y])
    
    def warp_image(self, image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warp image using TPS.
        
        Args:
            image: Input image
            output_shape: Output image shape (height, width)
            
        Returns:
            Warped image
        """
        from scipy import ndimage
        
        h, w = output_shape
        
        # Create output grid
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        
        # Transform grid points (inverse mapping)
        # We need to find where each output pixel came from in the input
        # For simplicity, use forward mapping approximation
        
        if image.ndim == 2:
            # Grayscale
            output = np.zeros((h, w), dtype=image.dtype)
            
            # Simple nearest-neighbor interpolation
            for i in range(len(self.source_landmarks)):
                src = self.source_landmarks[i]
                dst = self.target_landmarks[i]
                
                # Map neighborhood
                if (0 <= dst[1] < h) and (0 <= dst[0] < w):
                    if (0 <= src[1] < image.shape[0]) and (0 <= src[0] < image.shape[1]):
                        output[int(dst[1]), int(dst[0])] = image[int(src[1]), int(src[0])]
            
            # Fill holes with interpolation
            output = ndimage.gaussian_filter(output, sigma=1.0)
        
        else:
            # RGB
            output = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            for c in range(image.shape[2]):
                for i in range(len(self.source_landmarks)):
                    src = self.source_landmarks[i]
                    dst = self.target_landmarks[i]
                    
                    if (0 <= dst[1] < h) and (0 <= dst[0] < w):
                        if (0 <= src[1] < image.shape[0]) and (0 <= src[0] < image.shape[1]):
                            output[int(dst[1]), int(dst[0]), c] = image[int(src[1]), int(src[0]), c]
                
                output[:, :, c] = ndimage.gaussian_filter(output[:, :, c], sigma=1.0)
        
        return output