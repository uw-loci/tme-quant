"""
Main configuration options for CurveAlign analysis.

This module defines the primary configuration class for controlling
CurveAlign analysis behavior.
"""

from typing import Optional
from dataclasses import dataclass

from .feature_options import FeatureOptions


@dataclass
class CurveAlignOptions:
    """
    Configuration options for CurveAlign analysis.
    
    Parameters
    ----------
    keep : float, default 0.001
        Fraction of curvelet coefficients to keep
    scale : int, optional
        Specific curvelet scale to analyze
    group_radius : float, optional
        Radius for grouping nearby curvelets
    dist_thresh : float, default 100.0
        Distance threshold for boundary analysis (pixels)
    min_dist : float, optional
        Minimum distance from boundary (pixels)
    exclude_inside_mask : bool, default False
        Exclude curvelets inside boundary mask
    minimum_nearest_fibers : int, default 4
        Minimum number of nearest neighbors for density/alignment features
    minimum_box_size : int, default 16
        Minimum box size for local feature computation
    """
    keep: float = 0.001
    scale: Optional[int] = None
    group_radius: Optional[float] = None
    dist_thresh: float = 100.0
    min_dist: Optional[float] = None
    exclude_inside_mask: bool = False
    minimum_nearest_fibers: int = 4
    minimum_box_size: int = 16
    
    def to_feature_options(self) -> FeatureOptions:
        """Convert to FeatureOptions."""
        return FeatureOptions(
            minimum_nearest_fibers=self.minimum_nearest_fibers,
            minimum_box_size=self.minimum_box_size
        )
