# Specialized features for interactions:
# - Mechanical coupling scores
# - Migration guidance scores
# - Invasive potential scores
# - Contact pattern analysis
# - Alignment heterogeneity

"""
Detect interactions between TME components based on distance and spatial criteria.
"""

from typing import List, Optional, Tuple
import importlib

import numpy as np

cKDTree = importlib.import_module("scipy.spatial").cKDTree
shapely_geometry = importlib.import_module("shapely.geometry")
LineString = shapely_geometry.LineString
Point = shapely_geometry.Point
Polygon = shapely_geometry.Polygon

from ..config.analysis_params import InteractionPair, InteractionStrategy
from ...core.tme_models.cell_model import CellObject
from ...core.tme_models.fiber_model import FiberObject
from ...core.tme_models.tumor_model import TumorRegion
from .tacs_classifier import classify_fiber_tacs

class InteractionDetector:
    """Detect spatial interactions between TME components."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def detect_cell_fiber_interactions(
        self,
        cells: List[CellObject],
        fibers: List[FiberObject],
        max_distance: float = 50.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS,
        k: int = 5,
    ) -> List[InteractionPair]:
        if not cells or not fibers:
            return []

        pairs: List[InteractionPair] = []
        fiber_points = np.array([self._get_fiber_midpoint(f) for f in fibers])
        fiber_tree = cKDTree(fiber_points)

        for cell in cells:
            cell_centroid = np.array(cell.centroid)

            if strategy == InteractionStrategy.NEAREST:
                distance, idx = fiber_tree.query(cell_centroid)
                idx = int(idx)
                if distance <= max_distance:
                    pairs.append(self._create_cell_fiber_pair(cell, fibers[idx], float(distance)))

            elif strategy == InteractionStrategy.RADIUS:
                indices = fiber_tree.query_ball_point(cell_centroid, max_distance)
                for idx in indices:
                    idx = int(idx)
                    distance = np.linalg.norm(cell_centroid - fiber_points[idx])
                    pairs.append(self._create_cell_fiber_pair(cell, fibers[idx], float(distance)))

            elif strategy == InteractionStrategy.K_NEAREST:
                distances, indices = fiber_tree.query(cell_centroid, k=k)
                for dist, idx in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
                    if dist <= max_distance:
                        pairs.append(self._create_cell_fiber_pair(cell, fibers[int(idx)], float(dist)))

            elif strategy == InteractionStrategy.CONTACT:
                cell_geom = cell.geometry
                for fiber in fibers:
                    fiber_geom = fiber.geometry
                    if cell_geom.intersects(fiber_geom):
                        distance = cell_geom.distance(fiber_geom)
                        pairs.append(self._create_cell_fiber_pair(cell, fiber, distance, contact=True))

        if self.verbose:
            print(f"Found {len(pairs)} cell-fiber interactions")

        return pairs

    def _create_cell_fiber_pair(
        self,
        cell: CellObject,
        fiber: FiberObject,
        distance: float,
        contact: bool = False,
    ) -> InteractionPair:
        relative_angle = fiber.angle if hasattr(fiber, "angle") else None
        return InteractionPair(
            source_id=cell.object_id,
            target_id=self._fiber_id(fiber),
            source_type="cell",
            target_type="fiber",
            distance=distance,
            contact=contact,
            relative_angle=relative_angle,
            interaction_point=cell.centroid,
        )

    def detect_cell_cell_interactions(
        self,
        cells: List[CellObject],
        max_distance: float = 30.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS,
    ) -> List[InteractionPair]:
        if len(cells) < 2:
            return []

        pairs: List[InteractionPair] = []
        centroids = np.array([c.centroid for c in cells])
        tree = cKDTree(centroids)

        for i, cell in enumerate(cells):
            if strategy == InteractionStrategy.RADIUS:
                indices = tree.query_ball_point(centroids[i], max_distance)
                for idx in indices:
                    idx = int(idx)
                    if idx != i:
                        distance = np.linalg.norm(centroids[i] - centroids[idx])
                        pairs.append(self._create_cell_cell_pair(cell, cells[idx], float(distance)))

        return pairs

    def _create_cell_cell_pair(self, cell1: CellObject, cell2: CellObject, distance: float) -> InteractionPair:
        return InteractionPair(
            source_id=cell1.object_id,
            target_id=cell2.object_id,
            source_type="cell",
            target_type="cell",
            distance=distance,
        )

    def detect_fiber_fiber_interactions(
        self,
        fibers: List[FiberObject],
        max_distance: float = 20.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS,
    ) -> List[InteractionPair]:
        del strategy
        if len(fibers) < 2:
            return []

        pairs: List[InteractionPair] = []
        midpoints = np.array([self._get_fiber_midpoint(f) for f in fibers])
        tree = cKDTree(midpoints)

        for i, fiber in enumerate(fibers):
            indices = tree.query_ball_point(midpoints[i], max_distance)
            for idx in indices:
                idx = int(idx)
                if idx != i:
                    distance = np.linalg.norm(midpoints[i] - midpoints[idx])
                    relative_angle = self._compute_fiber_fiber_angle(fiber, fibers[idx])
                    pairs.append(
                        InteractionPair(
                            source_id=self._fiber_id(fiber),
                            target_id=self._fiber_id(fibers[idx]),
                            source_type="fiber",
                            target_type="fiber",
                            distance=float(distance),
                            relative_angle=relative_angle,
                        )
                    )

        return pairs

    def _compute_fiber_fiber_angle(self, fiber1: FiberObject, fiber2: FiberObject) -> Optional[float]:
        if not (hasattr(fiber1, "angle") and hasattr(fiber2, "angle")):
            return None
        if fiber1.angle is None or fiber2.angle is None:
            return None
        angle_diff = abs(fiber1.angle - fiber2.angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        return angle_diff

    def detect_fiber_tumor_interactions(
        self,
        fibers: List[FiberObject],
        tumor_regions: List[TumorRegion],
        boundary_distance: float = 100.0,
    ) -> List[InteractionPair]:
        pairs: List[InteractionPair] = []

        for tumor in tumor_regions:
            tumor_polygon = self._tumor_polygon(tumor)
            tumor_boundary = tumor_polygon.boundary
            tumor_id = self._tumor_id(tumor)

            for fiber in fibers:
                fiber_midpoint = Point(self._get_fiber_midpoint(fiber))
                distance = fiber_midpoint.distance(tumor_boundary)
                if distance > boundary_distance:
                    continue

                nearest_point = tumor_boundary.interpolate(tumor_boundary.project(fiber_midpoint))
                second_point = tumor_boundary.interpolate(
                    min(tumor_boundary.length, tumor_boundary.project(fiber_midpoint) + 1.0)
                )

                angle_to_normal = None
                angle_to_tangent = None
                if hasattr(fiber, "angle") and fiber.angle is not None:
                    angle_to_normal = self._compute_angle_to_boundary_normal(
                        fiber_orientation=float(fiber.angle),
                        boundary_point1=(nearest_point.x, nearest_point.y),
                        boundary_point2=(second_point.x, second_point.y),
                    )
                    angle_to_tangent = 90 - angle_to_normal if angle_to_normal is not None else None

                tacs_type = self._classify_tacs_type(angle_to_normal, angle_to_tangent, fiber, distance)

                pairs.append(
                    InteractionPair(
                        source_id=self._fiber_id(fiber),
                        target_id=tumor_id,
                        source_type="fiber",
                        target_type="tumor_boundary",
                        distance=distance,
                        angle_to_boundary_normal=angle_to_normal,
                        angle_to_boundary_tangent=angle_to_tangent,
                        nearest_boundary_point=(nearest_point.x, nearest_point.y),
                        interaction_type=tacs_type,
                    )
                )

        if self.verbose:
            print(f"Found {len(pairs)} fiber-tumor boundary interactions")

        return pairs

    def _classify_tacs_type(
        self,
        angle_to_normal: Optional[float],
        angle_to_tangent: Optional[float],
        fiber: FiberObject,
        distance: float = 0.0,
    ) -> Optional[str]:
        """
        Classify a fiber-tumor interaction as TACS-1/2/3.

        Delegates to the canonical classify_fiber_tacs() using the boundary
        TANGENT angle:
          angle_to_tangent = 0-30deg  → TACS-2 (parallel)
          angle_to_tangent = 60-90deg → TACS-3 (perpendicular, INVASIVE)
          angle_to_tangent = 30-60deg or curly → TACS-1
        """
        if angle_to_tangent is None:
            return None
        straightness = getattr(fiber, 'straightness', None)
        if straightness is None:
            straightness = 1.0
        return classify_fiber_tacs(
            angle_to_tangent=angle_to_tangent,
            straightness=straightness,
            distance_to_boundary=distance,
        )

    def detect_cell_tumor_interactions(
        self,
        cells: List[CellObject],
        tumor_regions: List[TumorRegion],
        boundary_distance: float = 100.0,
    ) -> List[InteractionPair]:
        pairs: List[InteractionPair] = []

        for tumor in tumor_regions:
            tumor_boundary = self._tumor_polygon(tumor).boundary
            tumor_id = self._tumor_id(tumor)

            for cell in cells:
                cell_point = Point(cell.centroid)
                distance = cell_point.distance(tumor_boundary)
                if distance <= boundary_distance:
                    pairs.append(
                        InteractionPair(
                            source_id=cell.object_id,
                            target_id=tumor_id,
                            source_type="cell",
                            target_type="tumor_boundary",
                            distance=float(distance),
                        )
                    )

        return pairs

    def detect_fiber_cell_interactions(
        self,
        fibers: List[FiberObject],
        cells: List[CellObject],
        max_distance: float = 50.0,
        strategy: InteractionStrategy = InteractionStrategy.NEAREST,
    ) -> List[InteractionPair]:
        if not fibers or not cells:
            return []

        pairs: List[InteractionPair] = []
        cell_centroids = np.array([c.centroid for c in cells])
        cell_tree = cKDTree(cell_centroids)

        for fiber in fibers:
            fiber_midpoint = self._get_fiber_midpoint(fiber)

            if strategy == InteractionStrategy.NEAREST:
                distance, idx = cell_tree.query(fiber_midpoint)
                idx = int(idx)
                if distance <= max_distance:
                    pairs.append(
                        InteractionPair(
                            source_id=self._fiber_id(fiber),
                            target_id=cells[idx].object_id,
                            source_type="fiber",
                            target_type="cell",
                            distance=float(distance),
                        )
                    )

            elif strategy == InteractionStrategy.RADIUS:
                indices = cell_tree.query_ball_point(fiber_midpoint, max_distance)
                for idx in indices:
                    idx = int(idx)
                    distance = np.linalg.norm(fiber_midpoint - cell_centroids[idx])
                    pairs.append(
                        InteractionPair(
                            source_id=self._fiber_id(fiber),
                            target_id=cells[idx].object_id,
                            source_type="fiber",
                            target_type="cell",
                            distance=float(distance),
                        )
                    )

        return pairs

    def _get_fiber_midpoint(self, fiber: FiberObject) -> np.ndarray:
        if len(fiber.centerline) == 0:
            return np.array([0.0, 0.0])
        mid_idx = len(fiber.centerline) // 2
        return np.array(fiber.centerline[mid_idx])

    def _compute_angle_to_boundary_normal(
        self,
        fiber_orientation: float,
        boundary_point1: Tuple[float, float],
        boundary_point2: Tuple[float, float],
    ) -> float:
        """
        Compute angle between fiber and boundary normal.

        Delegates to the canonical implementation in
        fiber_analysis.utils.geometry_utils.compute_angle_to_boundary_normal
        so that the geometry logic is defined in one place.
        """
        from ...fiber_analysis.utils.geometry_utils import (
            compute_angle_to_boundary_normal,
        )
        return compute_angle_to_boundary_normal(
            fiber_orientation=fiber_orientation,
            boundary_point1=boundary_point1,
            boundary_point2=boundary_point2,
        )

    def _tumor_polygon(self, tumor: TumorRegion) -> Polygon:
        roi = getattr(tumor, "roi", None)
        if roi is not None and hasattr(roi, "polygon"):
            return roi.polygon
        return Polygon(np.asarray(tumor.geometry.coordinates))

    def _tumor_id(self, tumor: TumorRegion) -> str:
        object_id = getattr(tumor, "object_id", None)
        if object_id is not None:
            return str(object_id)
        return str(getattr(tumor, "id", "tumor"))

    def _fiber_id(self, fiber: FiberObject) -> str:
        object_id = getattr(fiber, "object_id", None)
        if object_id is not None:
            return str(object_id)
        return str(getattr(fiber, "id", "fiber"))