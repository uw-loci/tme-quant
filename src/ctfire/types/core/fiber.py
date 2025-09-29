"""
Individual fiber data structure.

This module defines the Fiber class representing an individual fiber
extracted by the FIRE algorithm with geometric properties.
"""

from typing import NamedTuple, List, Optional
import numpy as np


class Fiber(NamedTuple):
    """
    Represents an individual fiber extracted by FIRE algorithm.
    
    Attributes
    ----------
    points : List[tuple]
        List of (row, col) coordinates defining the fiber path
    length : float
        Total fiber length in pixels
    width : float
        Mean fiber width in pixels
    angle_deg : float
        Mean fiber orientation angle in degrees (0-180)
    straightness : float
        Straightness measure (end-to-end distance / total length)
    endpoints : tuple
        Start and end coordinates ((row1, col1), (row2, col2))
    curvature : float, optional
        Mean curvature along the fiber
    """
    points: List[tuple]
    length: float
    width: float
    angle_deg: float
    straightness: float
    endpoints: tuple
    curvature: Optional[float] = None
