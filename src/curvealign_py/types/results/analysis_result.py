"""
Single image analysis results.

This module defines the primary result structure returned by
single image analysis operations.
"""

from typing import NamedTuple, List, Dict, Optional

from ..core.curvelet import Curvelet
from .feature_table import FeatureTable
from .boundary_metrics import BoundaryMetrics


class AnalysisResult(NamedTuple):
    """
    Complete results from single image analysis.
    
    Attributes
    ----------
    curvelets : List[Curvelet]
        Extracted curvelets
    features : FeatureTable
        Computed features for each curvelet
    boundary_metrics : BoundaryMetrics, optional
        Boundary analysis results if boundary provided
    stats : Dict[str, float]
        Summary statistics
    """
    curvelets: List[Curvelet]
    features: FeatureTable
    boundary_metrics: Optional[BoundaryMetrics]
    stats: Dict[str, float]
