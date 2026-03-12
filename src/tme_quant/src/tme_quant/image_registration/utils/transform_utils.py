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
# TRANSFORM UTILITIES
# Location: utils/transform_utils.py
# ============================================================

def compose_transforms(transform1, transform2):
    """
    Compose two transformations.
    
    Args:
        transform1: First transform
        transform2: Second transform
        
    Returns:
        Composed transform
    """
    from ..config.registration_params import Transform
    
    # Matrix multiplication
    composed_matrix = transform1.matrix @ transform2.matrix
    
    return Transform(
        transform_type=transform1.transform_type,
        matrix=composed_matrix
    )


def invert_transform(transform):
    """
    Invert transformation.
    
    Args:
        transform: Transform to invert
        
    Returns:
        Inverted transform
    """
    from ..config.registration_params import Transform
    
    return Transform(
        transform_type=transform.transform_type,
        matrix=np.linalg.inv(transform.matrix),
        parameters=-transform.parameters if transform.parameters is not None else None
    )


def apply_transform_to_points(points: np.ndarray, transform) -> np.ndarray:
    """
    Apply transformation to points.
    
    Args:
        points: Points to transform (n, 2)
        transform: Transform object
        
    Returns:
        Transformed points (n, 2)
    """
    # Convert to homogeneous coordinates
    points_hom = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply transform
    transformed_hom = (transform.matrix @ points_hom.T).T
    
    # Convert back to Cartesian
    transformed = transformed_hom[:, :2] / transformed_hom[:, 2:3]
    
    return transformed

'''
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
...
...
# Export all
__all__ = [
    # Landmark registration
    'ManualLandmarkRegistration',
    'ThinPlateSpline',
    
    # Transform IO
    'save_transform',
    'load_transform',
    'export_transform_matrix',
    
    # Landmark IO
    'save_landmarks',
    'load_landmarks',
    'export_landmark_pairs',
    
    # Image utils
    'ensure_grayscale',
    'pad_to_same_size',
    'crop_to_common_roi',
    
    # Transform utils
    'compose_transforms',
    'invert_transform',
    'apply_transform_to_points',
]


print("✅ Landmark-based registration and IO utilities complete")
print("  - ManualLandmarkRegistration")
print("  - ThinPlateSpline warping")
print("  - Transform IO (save/load)")
print("  - Landmark IO (save/load)")
'''