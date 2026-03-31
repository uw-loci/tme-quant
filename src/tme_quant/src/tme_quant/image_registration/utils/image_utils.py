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
# IMAGE UTILITIES
# Location: utils/image_utils.py
# ============================================================

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if image.ndim == 2:
        return image
    elif image.ndim == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    elif image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image


# pad_to_same_size and crop_to_common_roi are defined (with richer options)
# in image_registration.preprocessing — import from there
# to keep a single canonical implementation.
from ..preprocessing import (  # noqa: E402
    pad_to_same_size,
    crop_to_common_roi,
)