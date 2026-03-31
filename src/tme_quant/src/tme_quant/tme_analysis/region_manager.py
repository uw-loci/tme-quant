"""
Manage regions of interest and automated tumor detection.
"""

from typing import Any, Dict, List, Optional
import importlib

import numpy as np

shapely_geometry = importlib.import_module("shapely.geometry")
MultiPoint = shapely_geometry.MultiPoint
Point = shapely_geometry.Point
Polygon = shapely_geometry.Polygon
DBSCAN = importlib.import_module("sklearn.cluster").DBSCAN

from .config import TumorDetectionMethod, TumorDetectionParams
from ..core.base_models import Geometry, GeometryType
from ..core.tme_models.cell_model import CellObject
from ..core.tme_models.tumor_model import TumorRegion


class RegionManager:
    """Manage ROIs and automated tumor region detection."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def detect_tumor_regions(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams,
    ) -> List[TumorRegion]:
        if params.method == TumorDetectionMethod.CLUSTERING:
            return self._detect_by_clustering(cells, params)
        if params.method == TumorDetectionMethod.DENSITY:
            return self._detect_by_density(cells, params)
        if params.method == TumorDetectionMethod.CELL_TYPE:
            return self._detect_by_cell_type(cells, params)
        if params.method == TumorDetectionMethod.DEEP_LEARNING:
            return self._detect_by_deep_learning(cells, params)
        if params.method == TumorDetectionMethod.MANUAL:
            return []
        raise ValueError(f"Unknown detection method: {params.method}")

    def _detect_by_clustering(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams,
    ) -> List[TumorRegion]:
        if len(cells) < params.dbscan_min_samples:
            return []

        centroids = np.array([c.centroid for c in cells])
        clustering = DBSCAN(eps=params.dbscan_eps, min_samples=params.dbscan_min_samples).fit(centroids)
        labels = clustering.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if self.verbose:
            print(f"DBSCAN found {n_clusters} clusters")

        tumor_regions: List[TumorRegion] = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            cluster_centroids = centroids[cluster_mask]

            points = MultiPoint(cluster_centroids)
            boundary_polygon = points.convex_hull.buffer(50)
            area = boundary_polygon.area
            if area < params.min_tumor_area:
                continue

            if params.smooth_boundary:
                boundary_polygon = boundary_polygon.simplify(tolerance=params.smoothing_sigma)

            geometry = Geometry(
                type=GeometryType.POLYGON,
                coordinates=np.array(
                    boundary_polygon.exterior.coords
                    if hasattr(boundary_polygon, "exterior")
                    else boundary_polygon.coords
                ),
            )

            tumor_region = TumorRegion(
                object_id=f"tumor_cluster_{cluster_id}",
                name=f"tumor_cluster_{cluster_id}",
                geometry=geometry,
                metadata={
                    "detection_method": "clustering",
                    "n_cells": int(np.sum(cluster_mask)),
                    "area": float(area),
                },
            )
            tumor_regions.append(tumor_region)

        return tumor_regions

    def _detect_by_cell_type(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams,
    ) -> List[TumorRegion]:
        tumor_cells = [
            c
            for c in cells
            if c.cell_type and c.cell_type.value in params.tumor_cell_types
        ]

        if not tumor_cells:
            if self.verbose:
                print("No tumor cells found for region detection")
            return []

        return self._detect_by_clustering(tumor_cells, params)

    def _detect_by_density(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams,
    ) -> List[TumorRegion]:
        gaussian_kde = importlib.import_module("scipy.stats").gaussian_kde

        centroids = np.array([c.centroid for c in cells])
        kde = gaussian_kde(centroids.T, bw_method=params.density_bandwidth)

        x_min, y_min = centroids.min(axis=0) - 100
        x_max, y_max = centroids.max(axis=0) + 100

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        positions = np.vstack([xx.ravel(), yy.ravel()])
        _density = np.reshape(kde(positions), xx.shape)

        return []

    def _detect_by_deep_learning(
        self,
        cells: List[CellObject],
        params: TumorDetectionParams,
    ) -> List[TumorRegion]:
        del cells
        if params.dl_model_path is None:
            raise ValueError("Deep learning model path required")
        return []

    def generate_tumor_zones(
        self,
        tumor_regions: List[TumorRegion],
        invasive_margin_width: float = 50.0,
        stroma_width: float = 200.0,
    ) -> Dict[str, List[Any]]:
        zones: Dict[str, List[Any]] = {
            "tumor_core": [],
            "invasive_margin": [],
            "stroma": [],
        }

        for tumor in tumor_regions:
            polygon = self._to_polygon(tumor)
            zones["tumor_core"].append(polygon)

            boundary = polygon.boundary
            invasive_zone = boundary.buffer(invasive_margin_width)
            zones["invasive_margin"].append(invasive_zone)

            stroma_zone = boundary.buffer(invasive_margin_width + stroma_width)
            stroma_ring = stroma_zone.difference(invasive_zone)
            zones["stroma"].append(stroma_ring)

        return zones

    def filter_cells_by_roi(self, cells: Optional[List[CellObject]], roi: Any) -> List[CellObject]:
        if not cells:
            return []

        roi_polygon = self._get_roi_polygon(roi)
        filtered: List[CellObject] = []
        for cell in cells:
            if roi_polygon.contains(Point(cell.centroid)):
                filtered.append(cell)
        return filtered

    def filter_fibers_by_roi(self, fibers: Optional[List], roi: Any) -> List:
        if not fibers:
            return []

        roi_polygon = self._get_roi_polygon(roi)
        filtered: List[Any] = []
        for fiber in fibers:
            if roi_polygon.intersects(fiber.geometry):
                filtered.append(fiber)
        return filtered

    def _to_polygon(self, tumor: TumorRegion) -> Polygon:
        return Polygon(np.asarray(tumor.geometry.coordinates))

    def _get_roi_polygon(self, roi: Any) -> Polygon:
        if isinstance(roi, Polygon):
            return roi
        if hasattr(roi, "polygon"):
            return roi.polygon
        if hasattr(roi, "geometry") and hasattr(roi.geometry, "coordinates"):
            return Polygon(np.asarray(roi.geometry.coordinates))
        raise ValueError(f"Unsupported ROI type: {type(roi)}")
