"""
Curvelet data structure.

This module defines the core Curvelet class representing an individual
curvelet with position, orientation, and strength.
"""

from typing import NamedTuple, Optional


class Curvelet(NamedTuple):
    """
    Represents a single curvelet with position, orientation, and strength.
    
    Attributes
    ----------
    center_row : int
        Row coordinate of curvelet center
    center_col : int  
        Column coordinate of curvelet center
    angle_deg : float
        Orientation angle in degrees (0-180)
    weight : float, optional
        Curvelet coefficient magnitude/strength
    """
    center_row: int
    center_col: int
    angle_deg: float
    weight: Optional[float] = None
