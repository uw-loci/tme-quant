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
    from ..config.registration_params import Transform, TransformType
    
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