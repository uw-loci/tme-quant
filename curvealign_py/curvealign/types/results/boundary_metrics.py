"""
Boundary analysis results.

This module defines the data structure for storing results from
boundary analysis operations.
"""

from typing import NamedTuple, Dict
import numpy as np


class BoundaryMetrics(NamedTuple):
    """
    Results from boundary analysis.
    
    Attributes
    ----------
    relative_angles : np.ndarray
        Relative angles between curvelets and boundary
    distances : np.ndarray
        Distances from curvelets to boundary
    inside_mask : np.ndarray
        Boolean array indicating curvelets inside boundary
    alignment_stats : Dict[str, float]
        Summary statistics for boundary alignment
    """
    relative_angles: np.ndarray
    distances: np.ndarray
    inside_mask: np.ndarray
    alignment_stats: Dict[str, float]
