"""
CT-FIRE analysis results.

This module defines the result structure returned by
CT-FIRE fiber extraction operations.
"""

from typing import NamedTuple, List, Dict, Optional

from ..core.fiber import Fiber
from ..core.fiber_network import FiberNetwork


class CTFireResult(NamedTuple):
    """
    Complete results from CT-FIRE analysis.
    
    Attributes
    ----------
    fibers : List[Fiber]
        Extracted individual fibers
    network : FiberNetwork
        Network analysis of fiber connectivity
    stats : Dict[str, float]
        Summary statistics (mean length, width, angle, etc.)
    enhanced_image : np.ndarray, optional
        Curvelet-enhanced image used for extraction
    """
    fibers: List[Fiber]
    network: FiberNetwork
    stats: Dict[str, float]
    enhanced_image: Optional['np.ndarray'] = None


class FiberMetrics(NamedTuple):
    """
    Detailed metrics for individual fibers.
    
    Attributes
    ----------
    lengths : List[float]
        Length of each fiber
    widths : List[float]
        Mean width of each fiber  
    angles : List[float]
        Mean angle of each fiber
    straightness : List[float]
        Straightness measure for each fiber
    curvatures : List[float]
        Mean curvature for each fiber
    """
    lengths: List[float]
    widths: List[float]
    angles: List[float]
    straightness: List[float]
    curvatures: List[float]
