
"""
Additional Registration Components
TransformHandler and CrossCorrelationRegistration

Location: 
  - tme_quant/image_registration/core/transform_handler.py
  - tme_quant/image_registration/methods/intensity_based/cross_correlation.py
"""

import numpy as np
from typing import List, Tuple
from scipy.ndimage import shift, affine_transform
from skimage.registration import phase_cross_correlation

# ============================================================
# FILE 1: core/transform_handler.py
# ============================================================

class TransformHandler:
    """
    Handle transformation composition, inversion, and application.
    
    Provides utilities for working with geometric transformations.
    
    Example:
        >>> from tme_quant.image_registration.core import TransformHandler
        >>> 
        >>> handler = TransformHandler()
        >>> 
        >>> # Compose multiple transforms
        >>> composed = handler.compose([transform1, transform2, transform3])
        >>> 
        >>> # Invert transform
        >>> inverted = handler.invert(transform)
        >>> 
        >>> # Apply to points
        >>> transformed_points = handler.apply_to_points(points, transform)
        >>> 
        >>> # Apply to image
        >>> transformed_image = handler.apply_to_image(image, transform)
    """
    
    def __init__(self):
        """Initialize transform handler."""
        self.transforms = []
    
    def compose(self, transforms: List) -> 'Transform':
        """
        Compose multiple transformations.
        
        Combines transforms by matrix multiplication: T_final = T1 @ T2 @ T3
        
        Args:
            transforms: List of Transform objects to compose
            
        Returns:
            Composed Transform object
        """
        if len(transforms) == 0:
            # Return identity transform
            return self._create_identity_transform()
        
        # Start with first transform's matrix
        result_matrix = transforms[0].matrix.copy()
        
        # Compose with remaining transforms
        for transform in transforms[1:]:
            result_matrix = result_matrix @ transform.matrix
        
        # Create composed transform
        class Transform:
            def __init__(self, matrix, transform_type=None):
                self.matrix = matrix
                self.transform_type = transform_type or transforms[0].transform_type
        
        return Transform(result_matrix)
    
    def invert(self, transform) -> 'Transform':
        """
        Invert transformation.
        
        Computes the inverse transformation matrix.
        
        Args:
            transform: Transform to invert
            
        Returns:
            Inverted Transform
        """
        inv_matrix = np.linalg.inv(transform.matrix)
        
        class Transform:
            def __init__(self, matrix, transform_type=None):
                self.matrix = matrix
                self.transform_type = transform_type or transform.transform_type
        
        return Transform(inv_matrix, transform.transform_type)
    
    def apply_to_points(
        self,
        points: np.ndarray,
        transform
    ) -> np.ndarray:
        """
        Apply transformation to points.
        
        Args:
            points: Points to transform, shape (n, 2)
            transform: Transform to apply
            
        Returns:
            Transformed points, shape (n, 2)
        """
        # Convert to homogeneous coordinates
        n_points = len(points)
        points_hom = np.hstack([points, np.ones((n_points, 1))])
        
        # Apply transformation
        transformed_hom = (transform.matrix @ points_hom.T).T
        
        # Convert back to Cartesian coordinates
        transformed = transformed_hom[:, :2] / transformed_hom[:, 2:3]
        
        return transformed
    
    def apply_to_image(
        self,
        image: np.ndarray,
        transform,
        output_shape: Tuple[int, int] = None,
        order: int = 1
    ) -> np.ndarray:
        """
        Apply transformation to image.
        
        Args:
            image: Input image
            transform: Transform to apply
            output_shape: Output image shape (height, width), default same as input
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            
        Returns:
            Transformed image
        """
        if output_shape is None:
            output_shape = image.shape[:2]
        
        # Get inverse transformation for image warping
        inv_matrix = np.linalg.inv(transform.matrix)
        
        # Extract affine parameters
        affine_matrix = inv_matrix[:2, :2]
        offset = inv_matrix[:2, 2]
        
        if image.ndim == 2:
            # Grayscale image
            transformed = affine_transform(
                image,
                affine_matrix,
                offset=offset,
                output_shape=output_shape,
                order=order,
                mode='constant',
                cval=0
            )
        else:
            # RGB or multi-channel image
            n_channels = image.shape[2]
            transformed = np.zeros(output_shape + (n_channels,), dtype=image.dtype)
            
            for c in range(n_channels):
                transformed[:, :, c] = affine_transform(
                    image[:, :, c],
                    affine_matrix,
                    offset=offset,
                    output_shape=output_shape,
                    order=order,
                    mode='constant',
                    cval=0
                )
        
        return transformed
    
    def _create_identity_transform(self):
        """Create identity transformation."""
        class Transform:
            def __init__(self):
                self.matrix = np.eye(3)
                self.transform_type = "identity"
        
        return Transform()
    
    def get_translation(self, transform) -> np.ndarray:
        """
        Extract translation vector from transform.
        
        Args:
            transform: Transform object
            
        Returns:
            Translation vector [tx, ty]
        """
        return transform.matrix[:2, 2]
    
    def get_rotation_angle(self, transform) -> float:
        """
        Extract rotation angle from transform.
        
        Args:
            transform: Transform object
            
        Returns:
            Rotation angle in radians
        """
        # Extract rotation from matrix
        angle = np.arctan2(transform.matrix[1, 0], transform.matrix[0, 0])
        return angle
    
    def get_scale(self, transform) -> Tuple[float, float]:
        """
        Extract scale factors from transform.
        
        Args:
            transform: Transform object
            
        Returns:
            Tuple of (scale_x, scale_y)
        """
        # Extract scale from matrix
        scale_x = np.sqrt(transform.matrix[0, 0]**2 + transform.matrix[1, 0]**2)
        scale_y = np.sqrt(transform.matrix[0, 1]**2 + transform.matrix[1, 1]**2)
        
        return scale_x, scale_y