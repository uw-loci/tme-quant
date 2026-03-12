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
# LANDMARK IO
# Location: io/landmark_io.py
# ============================================================

def save_landmarks(landmarks: np.ndarray, filepath: str, image_name: str = ""):
    """
    Save landmarks to file.
    
    Args:
        landmarks: Landmark coordinates (n, 2)
        filepath: Output file path
        image_name: Optional image name
    """
    filepath = Path(filepath)
    
    data = {
        'image_name': image_name,
        'num_landmarks': len(landmarks),
        'landmarks': landmarks.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_landmarks(filepath: str) -> np.ndarray:
    """
    Load landmarks from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Landmark coordinates (n, 2)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return np.array(data['landmarks'])


def export_landmark_pairs(
    fixed_landmarks: np.ndarray,
    moving_landmarks: np.ndarray,
    filepath: str
):
    """
    Export landmark pairs to file.
    
    Args:
        fixed_landmarks: Fixed image landmarks (n, 2)
        moving_landmarks: Moving image landmarks (n, 2)
        filepath: Output file path
    """
    data = {
        'num_pairs': len(fixed_landmarks),
        'fixed_landmarks': fixed_landmarks.tolist(),
        'moving_landmarks': moving_landmarks.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)