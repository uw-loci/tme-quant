"""
TACS (Tumor-Associated Collagen Signatures) Classification

Classifies fibers based on their orientation relative to tumor boundaries.

References:
    - Provenzano et al. (2006) - TACS identification
    - Conklin et al. (2011) - TACS-3 and prognosis
"""

import numpy as np
from typing import Optional, List, Tuple
from shapely.geometry import Point, Polygon

# Import geometry utilities from fiber_analysis module
from ...fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
    compute_angle_to_boundary_normal_simplified
)


def classify_fiber_tacs(
    angle_to_boundary: float,
    straightness: float,
    distance_to_boundary: float,
    tacs_zone_width: float = 100.0,
    straightness_threshold: float = 0.7
) -> Optional[str]:
    """
    Classify fiber as TACS-1/2/3 based on orientation and morphology.
    
    CORRECTED angle ranges relative to boundary normal:
    - TACS-3: 60-90° (perpendicular, INVASIVE) - HIGH RISK
    - TACS-2: 0-30° (parallel) - Medium risk
    - TACS-1: 30-60° (intermediate) OR curly fibers - Low risk
    
    Args:
        angle_to_boundary: Angle between fiber orientation and boundary normal (0-90°)
        straightness: Fiber straightness coefficient (0-1)
        distance_to_boundary: Distance from fiber to tumor boundary (microns)
        tacs_zone_width: Width of TACS zone around boundary (default 100 microns)
        straightness_threshold: Minimum straightness for TACS-2/3 (default 0.7)
        
    Returns:
        TACS classification ('TACS-1', 'TACS-2', 'TACS-3') or None if outside zone
        
    Notes:
        - TACS classification only applies within tacs_zone_width of boundary
        - TACS-2 and TACS-3 require straight fibers (straightness >= threshold)
        - Curly fibers are always classified as TACS-1 regardless of orientation
        
    Example:
        >>> # Perpendicular, straight fiber within TACS zone
        >>> tacs = classify_fiber_tacs(75, 0.85, 50)
        >>> print(tacs)  # 'TACS-3' (INVASIVE)
        
        >>> # Parallel, straight fiber
        >>> tacs = classify_fiber_tacs(15, 0.85, 50)
        >>> print(tacs)  # 'TACS-2'
        
        >>> # Perpendicular but curly fiber
        >>> tacs = classify_fiber_tacs(75, 0.5, 50)
        >>> print(tacs)  # 'TACS-1' (not straight enough for TACS-3)
    """
    # Check if within TACS zone
    if distance_to_boundary > tacs_zone_width:
        return None
    
    if angle_to_boundary is None or np.isnan(angle_to_boundary):
        return None
    
    # Normalize angle to [0, 90]
    angle = np.abs(angle_to_boundary) % 90
    
    # CORRECTED classification logic
    if 60 <= angle <= 90:
        # Perpendicular to boundary (INVASIVE - high risk)
        if straightness >= straightness_threshold:
            return 'TACS-3'
        else:
            return 'TACS-1'  # Curly perpendicular -> random pattern
    
    elif 0 <= angle < 30:
        # Parallel to boundary
        if straightness >= straightness_threshold:
            return 'TACS-2'
        else:
            return 'TACS-1'  # Curly parallel -> random pattern
    
    elif 30 <= angle < 60:
        # Intermediate orientation
        return 'TACS-1'
    
    else:
        return None


def classify_fiber_segment_tacs_like(
    angle_to_boundary: float,
    distance_to_boundary: float,
    tacs_zone_width: float = 100.0
) -> Optional[str]:
    """
    Classify fiber segment (from CurveAlign) as TACS-like.
    
    For orientation-only analysis without straightness information.
    Returns TACS-X-like classifications.
    
    Args:
        angle_to_boundary: Angle to boundary normal (0-90°)
        distance_to_boundary: Distance to boundary (microns)
        tacs_zone_width: TACS zone width (microns)
        
    Returns:
        'TACS-1-like', 'TACS-2-like', 'TACS-3-like', or None
    """
    if distance_to_boundary > tacs_zone_width:
        return None
    
    if angle_to_boundary is None or np.isnan(angle_to_boundary):
        return None
    
    angle = np.abs(angle_to_boundary) % 90
    
    if 60 <= angle <= 90:
        return 'TACS-3-like'  # Perpendicular-like (invasive-like)
    elif 0 <= angle < 30:
        return 'TACS-2-like'  # Parallel-like
    elif 30 <= angle < 60:
        return 'TACS-1-like'  # Intermediate-like
    else:
        return None


def get_tacs_color(tacs_type: str) -> Tuple[int, int, int]:
    """
    Get RGB color for TACS type visualization.
    
    Args:
        tacs_type: TACS classification string
        
    Returns:
        RGB color tuple (0-255)
        
    Colors:
        - TACS-3: RED (255, 0, 0) - Perpendicular, INVASIVE, high risk
        - TACS-2: GREEN (0, 255, 0) - Parallel, medium risk
        - TACS-1: BLUE (0, 0, 255) - Random/intermediate, low risk
    """
    colors = {
        'TACS-3': (255, 0, 0),          # RED - INVASIVE
        'TACS-2': (0, 255, 0),          # GREEN
        'TACS-1': (0, 0, 255),          # BLUE
        'TACS-3-like': (255, 0, 0),     # RED
        'TACS-2-like': (0, 255, 0),     # GREEN
        'TACS-1-like': (0, 0, 255),     # BLUE
    }
    return colors.get(tacs_type, (128, 128, 128))  # Gray for unknown


__all__ = [
    'classify_fiber_tacs',
    'classify_fiber_segment_tacs_like',
    'get_tacs_color',
]