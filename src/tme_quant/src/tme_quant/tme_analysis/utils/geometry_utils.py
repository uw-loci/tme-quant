"""
Geometry utilities for TME analysis.

Note: Many functions can be imported from fiber_analysis.utils.geometry_utils
to avoid duplication. This file contains TME-specific additions.
"""

import numpy as np
from typing import Tuple, Optional
from shapely.geometry import Point, LineString, Polygon


# Import shared utilities from fiber_analysis
try:
    from ...fiber_analysis.utils.geometry_utils import (
        find_nearest_boundary_point,
        compute_boundary_normal,
        compute_relative_angles
    )
except ImportError:
    # Fallback if fiber_analysis not available
    pass


def compute_region_centroid(
    points: np.ndarray
) -> np.ndarray:
    """
    Compute centroid of a region defined by points.
    
    Args:
        points: Array of points (n, 2)
        
    Returns:
        Centroid coordinates
    """
    return np.mean(points, axis=0)


def compute_region_area(
    polygon: Polygon
) -> float:
    """
    Compute area of a polygon region.
    
    Args:
        polygon: Shapely Polygon
        
    Returns:
        Area in square units
    """
    return polygon.area


def point_in_polygon(
    point: Tuple[float, float],
    polygon: Polygon
) -> bool:
    """
    Test if a point is inside a polygon.
    
    Args:
        point: (x, y) coordinates
        polygon: Shapely Polygon
        
    Returns:
        True if point is inside polygon
    """
    return polygon.contains(Point(point))


def compute_convex_hull(
    points: np.ndarray
) -> Polygon:
    """
    Compute convex hull of points.
    
    Args:
        points: Array of points (n, 2)
        
    Returns:
        Shapely Polygon representing convex hull
    """
    from shapely.geometry import MultiPoint
    
    multi_point = MultiPoint(points)
    return multi_point.convex_hull


def buffer_polygon(
    polygon: Polygon,
    distance: float
) -> Polygon:
    """
    Create buffer around polygon.
    
    Args:
        polygon: Input polygon
        distance: Buffer distance
        
    Returns:
        Buffered polygon
    """
    return polygon.buffer(distance)