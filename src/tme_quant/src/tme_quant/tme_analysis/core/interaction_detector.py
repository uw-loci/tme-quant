"""
Detect interactions between TME components based on distance and spatial criteria.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

from ..config.analysis_params import InteractionPair, InteractionStrategy
from ...core.tme_models.cell_model import CellObject
from ...core.tme_models.fiber_model import FiberObject
from ...core.tme_models.tumor_model import TumorRegion


class InteractionDetector:
    """
    Detect spatial interactions between TME components.
    
    Supports multiple strategies:
        - NEAREST: Find nearest neighbor only
        - RADIUS: Find all within radius
        - K_NEAREST: Find K nearest neighbors
        - CONTACT: Physical contact/overlap detection
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize interaction detector."""
        self.verbose = verbose
    
    # ============================================================
    # CELL-FIBER INTERACTIONS
    # ============================================================
    
    def detect_cell_fiber_interactions(
        self,
        cells: List[CellObject],
        fibers: List[FiberObject],
        max_distance: float = 50.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS,
        k: int = 5
    ) -> List[InteractionPair]:
        """
        Detect cell-fiber interactions.
        
        Args:
            cells: List of cells
            fibers: List of fibers
            max_distance: Maximum interaction distance (microns)
            strategy: Interaction detection strategy
            k: Number of nearest neighbors (for K_NEAREST)
            
        Returns:
            List of InteractionPair objects
        """
        if not cells or not fibers:
            return []
        
        pairs = []
        
        # Build KD-tree for fiber midpoints
        fiber_points = np.array([self._get_fiber_midpoint(f) for f in fibers])
        fiber_tree = cKDTree(fiber_points)
        
        # For each cell, find interacting fibers
        for cell in cells:
            cell_centroid = np.array(cell.centroid)
            
            if strategy == InteractionStrategy.NEAREST:
                # Find single nearest fiber
                distance, idx = fiber_tree.query(cell_centroid)
                
                if distance <= max_distance:
                    pair = self._create_cell_fiber_pair(
                        cell, fibers[idx], distance
                    )
                    pairs.append(pair)
            
            elif strategy == InteractionStrategy.RADIUS:
                # Find all fibers within radius
                indices = fiber_tree.query_ball_point(cell_centroid, max_distance)
                
                for idx in indices:
                    distance = np.linalg.norm(cell_centroid - fiber_points[idx])
                    pair = self._create_cell_fiber_pair(
                        cell, fibers[idx], distance
                    )
                    pairs.append(pair)
            
            elif strategy == InteractionStrategy.K_NEAREST:
                # Find K nearest fibers
                distances, indices = fiber_tree.query(cell_centroid, k=k)
                
                for dist, idx in zip(distances, indices):
                    if dist <= max_distance:
                        pair = self._create_cell_fiber_pair(
                            cell, fibers[idx], dist
                        )
                        pairs.append(pair)
            
            elif strategy == InteractionStrategy.CONTACT:
                # Check for physical contact/overlap
                cell_geom = cell.geometry
                
                for fiber in fibers:
                    fiber_geom = fiber.geometry
                    
                    if cell_geom.intersects(fiber_geom):
                        distance = cell_geom.distance(fiber_geom)
                        pair = self._create_cell_fiber_pair(
                            cell, fiber, distance, contact=True
                        )
                        pairs.append(pair)
        
        if self.verbose:
            print(f"Found {len(pairs)} cell-fiber interactions")
        
        return pairs
    
    def _create_cell_fiber_pair(
        self,
        cell: CellObject,
        fiber: FiberObject,
        distance: float,
        contact: bool = False
    ) -> InteractionPair:
        """Create InteractionPair for cell-fiber interaction."""
        # Compute relative angle if fiber has orientation
        relative_angle = None
        if hasattr(fiber, 'angle') and fiber.angle is not None:
            relative_angle = fiber.angle
        
        pair = InteractionPair(
            source_id=cell.object_id,
            target_id=fiber.object_id,
            source_type="cell",
            target_type="fiber",
            distance=distance,
            contact=contact,
            relative_angle=relative_angle,
            interaction_point=cell.centroid
        )
        
        return pair
    
    # ============================================================
    # CELL-CELL INTERACTIONS
    # ============================================================
    
    def detect_cell_cell_interactions(
        self,
        cells: List[CellObject],
        max_distance: float = 30.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS
    ) -> List[InteractionPair]:
        """
        Detect cell-cell interactions (clustering, neighbors).
        
        Args:
            cells: List of cells
            max_distance: Maximum interaction distance
            strategy: Interaction detection strategy
            
        Returns:
            List of InteractionPair objects
        """
        if len(cells) < 2:
            return []
        
        pairs = []
        
        # Build KD-tree
        centroids = np.array([c.centroid for c in cells])
        tree = cKDTree(centroids)
        
        # Find neighbors for each cell
        for i, cell in enumerate(cells):
            if strategy == InteractionStrategy.RADIUS:
                # All cells within radius
                indices = tree.query_ball_point(centroids[i], max_distance)
                
                for idx in indices:
                    if idx != i:  # Skip self
                        distance = np.linalg.norm(centroids[i] - centroids[idx])
                        pair = self._create_cell_cell_pair(
                            cell, cells[idx], distance
                        )
                        pairs.append(pair)
        
        return pairs
    
    def _create_cell_cell_pair(
        self,
        cell1: CellObject,
        cell2: CellObject,
        distance: float
    ) -> InteractionPair:
        """Create InteractionPair for cell-cell interaction."""
        pair = InteractionPair(
            source_id=cell1.object_id,
            target_id=cell2.object_id,
            source_type="cell",
            target_type="cell",
            distance=distance
        )
        
        return pair
    
    # ============================================================
    # FIBER-FIBER INTERACTIONS
    # ============================================================
    
    def detect_fiber_fiber_interactions(
        self,
        fibers: List[FiberObject],
        max_distance: float = 20.0,
        strategy: InteractionStrategy = InteractionStrategy.RADIUS
    ) -> List[InteractionPair]:
        """
        Detect fiber-fiber interactions (alignment, crossings).
        
        Args:
            fibers: List of fibers
            max_distance: Maximum interaction distance
            strategy: Interaction detection strategy
            
        Returns:
            List of InteractionPair objects
        """
        if len(fibers) < 2:
            return []
        
        pairs = []
        
        # Build KD-tree for fiber midpoints
        midpoints = np.array([self._get_fiber_midpoint(f) for f in fibers])
        tree = cKDTree(midpoints)
        
        # Find neighboring fibers
        for i, fiber in enumerate(fibers):
            indices = tree.query_ball_point(midpoints[i], max_distance)
            
            for idx in indices:
                if idx != i:  # Skip self
                    distance = np.linalg.norm(midpoints[i] - midpoints[idx])
                    
                    # Compute relative angle
                    relative_angle = self._compute_fiber_fiber_angle(
                        fiber, fibers[idx]
                    )
                    
                    pair = InteractionPair(
                        source_id=fiber.object_id,
                        target_id=fibers[idx].object_id,
                        source_type="fiber",
                        target_type="fiber",
                        distance=distance,
                        relative_angle=relative_angle
                    )
                    
                    pairs.append(pair)
        
        return pairs
    
    def _compute_fiber_fiber_angle(
        self,
        fiber1: FiberObject,
        fiber2: FiberObject
    ) -> Optional[float]:
        """Compute angle between two fibers."""
        if not (hasattr(fiber1, 'angle') and hasattr(fiber2, 'angle')):
            return None
        
        if fiber1.angle is None or fiber2.angle is None:
            return None
        
        # Compute relative angle
        angle_diff = abs(fiber1.angle - fiber2.angle)
        
        # Normalize to [0, 90]
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        return angle_diff
    
    # ============================================================
    # FIBER-TUMOR INTERACTIONS (TACS)
    # ============================================================
    
    def detect_fiber_tumor_interactions(
        self,
        fibers: List[FiberObject],
        tumor_regions: List[TumorRegion],
        boundary_distance: float = 100.0
    ) -> List[InteractionPair]:
        """
        Detect fiber-tumor boundary interactions for TACS analysis.
        
        Args:
            fibers: List of fibers
            tumor_regions: List of tumor regions
            boundary_distance: Boundary zone width (microns)
            
        Returns:
            List of InteractionPair objects with TACS metrics
        """
        pairs = []
        
        for tumor in tumor_regions:
            tumor_boundary = tumor.roi.boundary  # Shapely LineString
            
            for fiber in fibers:
                fiber_midpoint = Point(self._get_fiber_midpoint(fiber))
                
                # Distance from fiber to tumor boundary
                distance = fiber_midpoint.distance(tumor_boundary)
                
                # Only include fibers in boundary zone
                if distance <= boundary_distance:
                    # Find nearest point on boundary
                    nearest_point = tumor_boundary.interpolate(
                        tumor_boundary.project(fiber_midpoint)
                    )
                    
                    # Compute boundary normal at nearest point
                    boundary_normal = self._compute_boundary_normal(
                        tumor_boundary, nearest_point
                    )
                    
                    # Compute fiber angle relative to boundary
                    angle_to_normal, angle_to_tangent = self._compute_fiber_boundary_angles(
                        fiber, boundary_normal
                    )
                    
                    # Classify TACS type
                    tacs_type = self._classify_tacs_type(
                        angle_to_normal, angle_to_tangent, fiber
                    )
                    
                    pair = InteractionPair(
                        source_id=fiber.object_id,
                        target_id=tumor.object_id,
                        source_type="fiber",
                        target_type="tumor_boundary",
                        distance=distance,
                        angle_to_boundary_normal=angle_to_normal,
                        angle_to_boundary_tangent=angle_to_tangent,
                        nearest_boundary_point=(nearest_point.x, nearest_point.y),
                        interaction_type=tacs_type
                    )
                    
                    pairs.append(pair)
        
        if self.verbose:
            print(f"Found {len(pairs)} fiber-tumor boundary interactions")
        
        return pairs
    
    def _compute_boundary_normal(
        self,
        boundary: LineString,
        point: Point
    ) -> np.ndarray:
        """Compute outward normal at point on boundary."""
        from ...fiber_analysis.utils.geometry_utils import compute_boundary_normal
        
        return compute_boundary_normal(
            boundary_coords=np.array(boundary.coords),
            query_point=np.array([point.x, point.y])
        )
    
    def _compute_fiber_boundary_angles(
        self,
        fiber: FiberObject,
        boundary_normal: np.ndarray
    ) -> Tuple[float, float]:
        """Compute fiber angle relative to boundary normal and tangent."""
        if not hasattr(fiber, 'angle') or fiber.angle is None:
            return None, None
        
        # Fiber direction vector
        fiber_angle_rad = np.radians(fiber.angle)
        fiber_vector = np.array([np.cos(fiber_angle_rad), np.sin(fiber_angle_rad)])
        
        # Angle to normal
        angle_to_normal = np.degrees(np.arccos(
            np.clip(np.dot(fiber_vector, boundary_normal), -1.0, 1.0)
        ))
        
        # Normalize to [0, 90]
        if angle_to_normal > 90:
            angle_to_normal = 180 - angle_to_normal
        
        # Angle to tangent (perpendicular to normal)
        angle_to_tangent = 90 - angle_to_normal
        
        return angle_to_normal, angle_to_tangent
    
    def _classify_tacs_type(
        self,
        angle_to_normal: Optional[float],
        angle_to_tangent: Optional[float],
        fiber: FiberObject
    ) -> Optional[str]:
        """Classify fiber as TACS-1, TACS-2, or TACS-3."""
        if angle_to_normal is None:
            return None
        
        # Check straightness
        straightness = fiber.straightness if hasattr(fiber, 'straightness') else 1.0
        
        # TACS-3: Perpendicular to boundary (angle to normal < 30°)
        if angle_to_normal < 30 and straightness > 0.7:
            return "TACS-3"
        
        # TACS-2: Parallel to boundary (angle to tangent < 30°)
        elif angle_to_tangent < 30 and straightness > 0.7:
            return "TACS-2"
        
        # TACS-1: Random, curly
        else:
            return "TACS-1"
    
    # ============================================================
    # CELL-TUMOR INTERACTIONS
    # ============================================================
    
    def detect_cell_tumor_interactions(
        self,
        cells: List[CellObject],
        tumor_regions: List[TumorRegion],
        boundary_distance: float = 100.0
    ) -> List[InteractionPair]:
        """
        Detect cell-tumor boundary interactions.
        
        Args:
            cells: List of cells
            tumor_regions: List of tumor regions
            boundary_distance: Boundary zone width
            
        Returns:
            List of InteractionPair objects
        """
        pairs = []
        
        for tumor in tumor_regions:
            tumor_boundary = tumor.roi.boundary
            
            for cell in cells:
                cell_point = Point(cell.centroid)
                distance = cell_point.distance(tumor_boundary)
                
                if distance <= boundary_distance:
                    pair = InteractionPair(
                        source_id=cell.object_id,
                        target_id=tumor.object_id,
                        source_type="cell",
                        target_type="tumor_boundary",
                        distance=distance
                    )
                    
                    pairs.append(pair)
        
        return pairs
    
    # ============================================================
    # FIBER-CELL INTERACTIONS (reverse of cell-fiber)
    # ============================================================
    
    def detect_fiber_cell_interactions(
        self,
        fibers: List[FiberObject],
        cells: List[CellObject],
        max_distance: float = 50.0,
        strategy: InteractionStrategy = InteractionStrategy.NEAREST
    ) -> List[InteractionPair]:
        """
        Detect fiber-cell interactions (fiber-centric view).
        
        For each fiber, find interacting cells.
        """
        if not fibers or not cells:
            return []
        
        pairs = []
        
        # Build KD-tree for cell centroids
        cell_centroids = np.array([c.centroid for c in cells])
        cell_tree = cKDTree(cell_centroids)
        
        # For each fiber, find cells
        for fiber in fibers:
            fiber_midpoint = self._get_fiber_midpoint(fiber)
            
            if strategy == InteractionStrategy.NEAREST:
                distance, idx = cell_tree.query(fiber_midpoint)
                
                if distance <= max_distance:
                    pair = InteractionPair(
                        source_id=fiber.object_id,
                        target_id=cells[idx].object_id,
                        source_type="fiber",
                        target_type="cell",
                        distance=distance
                    )
                    pairs.append(pair)
            
            elif strategy == InteractionStrategy.RADIUS:
                indices = cell_tree.query_ball_point(fiber_midpoint, max_distance)
                
                for idx in indices:
                    distance = np.linalg.norm(
                        fiber_midpoint - cell_centroids[idx]
                    )
                    pair = InteractionPair(
                        source_id=fiber.object_id,
                        target_id=cells[idx].object_id,
                        source_type="fiber",
                        target_type="cell",
                        distance=distance
                    )
                    pairs.append(pair)
        
        return pairs
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _get_fiber_midpoint(self, fiber: FiberObject) -> np.ndarray:
        """Get fiber midpoint for spatial queries."""
        if len(fiber.centerline) == 0:
            return np.array([0, 0])
        
        mid_idx = len(fiber.centerline) // 2
        return np.array(fiber.centerline[mid_idx])