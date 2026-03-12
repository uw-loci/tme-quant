"""Geometry helper models used by TME object models."""

from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in 2D."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class ROI:
    """Simple ROI wrapper around a shapely polygon."""

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
