"""
Geometry helpers shared across all TMEQuant modules.

Provides:
  - BoundingBox / ROI dataclasses (used by tme_models)
  - Shapely-based polygon utility functions (merged from tme_analysis/utils/geometry_utils.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in 2D."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class ROI:
    """Simple ROI wrapper around a Shapely polygon."""

    polygon: Polygon

    @property
    def boundary(self):
        return self.polygon.boundary

    @staticmethod
    def from_shapely(geometry: Any) -> "ROI":
        if isinstance(geometry, Polygon):
            return ROI(polygon=geometry)
        if hasattr(geometry, "geom_type"):
            return ROI(polygon=Polygon(getattr(geometry, "exterior", geometry).coords))
        raise ValueError(f"Unsupported geometry for ROI: {type(geometry)}")


# ── Polygon utility functions ─────────────────────────────────────────────────
# Merged from tme_analysis/utils/geometry_utils.py


def compute_region_centroid(points: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of a region defined by a point array.

    Args:
        points: Array of shape (n, 2).

    Returns:
        Centroid as a 1-D array of shape (2,).
    """
    return np.mean(points, axis=0)


def compute_region_area(polygon: Polygon) -> float:
    """
    Return the area of a Shapely polygon.

    Args:
        polygon: Shapely Polygon.

    Returns:
        Area in square units.
    """
    return polygon.area


def point_in_polygon(point: Tuple[float, float], polygon: Polygon) -> bool:
    """
    Test whether a point lies inside a polygon.

    Args:
        point: (x, y) coordinates.
        polygon: Shapely Polygon.

    Returns:
        True if the point is inside the polygon.
    """
    return polygon.contains(Point(point))


def compute_convex_hull(points: np.ndarray) -> Polygon:
    """
    Compute the convex hull of a set of points.

    Args:
        points: Array of shape (n, 2).

    Returns:
        Shapely Polygon representing the convex hull.
    """
    return MultiPoint(points).convex_hull


def buffer_polygon(polygon: Polygon, distance: float) -> Polygon:
    """
    Create a buffer around a polygon.

    Args:
        polygon: Input polygon.
        distance: Buffer distance in the same units as the polygon coordinates.

    Returns:
        Buffered Shapely Polygon.
    """
    return polygon.buffer(distance)
