"""
IO utilities for registration transforms and landmarks.
"""

from __future__ import annotations


# ========================================================
# LANDMARK IO
# ========================================================

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
# ========================================================
# TRANSFORM IO
# ========================================================

import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy.spatial import distance_matrix
from scipy.interpolate import Rbf

# ============================================================
# TRANSFORM IO
# Location: io/transform_io.py
# ============================================================

def save_transform(transform, filepath: str):
    """
    Save transformation to file.
    
    Args:
        transform: Transform object
        filepath: Output file path (.json or .txt)
    """
    filepath = Path(filepath)
    
    data = {
        'transform_type': transform.transform_type.value,
        'matrix': transform.matrix.tolist(),
        'parameters': transform.parameters.tolist() if transform.parameters is not None else None,
        'source_image_shape': transform.source_image_shape,
        'target_image_shape': transform.target_image_shape,
        'pixel_size_source': transform.pixel_size_source,
        'pixel_size_target': transform.pixel_size_target,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_transform(filepath: str):
    """
    Load transformation from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Transform object
    """
    from .config import Transform, TransformType
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    transform = Transform(
        transform_type=TransformType(data['transform_type']),
        matrix=np.array(data['matrix']),
        parameters=np.array(data['parameters']) if data['parameters'] else None,
        source_image_shape=tuple(data['source_image_shape']) if data['source_image_shape'] else None,
        target_image_shape=tuple(data['target_image_shape']) if data['target_image_shape'] else None,
        pixel_size_source=data['pixel_size_source'],
        pixel_size_target=data['pixel_size_target']
    )
    
    return transform


def export_transform_matrix(transform, filepath: str):
    """
    Export transform matrix as text file.
    
    Args:
        transform: Transform object
        filepath: Output file path
    """
    np.savetxt(filepath, transform.matrix, fmt='%.6f')