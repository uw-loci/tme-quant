"""Spatial relationship feature extraction for prognostic TME analysis."""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...tme_analysis.config import InteractionPair

np = importlib.import_module("numpy")


class SpatialRelationshipExtractor:
    """
    Extract spatial relationship features for prognostic analysis.
    """
    
    def extract_features(
        self,
        cells: List,
        fibers: List,
        interaction_pairs: List[InteractionPair]
    ) -> Dict[str, float]:
        """Extract spatial relationship features."""
        features = {}
        
        # 1. Cell-Fiber Spatial Coupling
        features['cell_fiber_coupling'] = self._compute_coupling_score(
            cells, fibers, interaction_pairs
        )
        
        # 2. Fiber Network Connectivity
        features['fiber_network_connectivity'] = self._compute_network_connectivity(
            fibers
        )
        
        # 3. Cell Cluster Compactness
        features['cell_cluster_compactness'] = self._compute_cluster_compactness(
            cells
        )
        
        # 4. Invasion Directionality
        features['invasion_directionality'] = self._compute_invasion_directionality(
            interaction_pairs
        )
        
        return features
    
    def _compute_coupling_score(
        self,
        cells: List,
        fibers: List,
        pairs: List[InteractionPair]
    ) -> float:
        """Compute cell-fiber spatial coupling."""
        if not pairs:
            return 0.0
        
        # Ratio of cells with fiber interactions
        cell_ids_with_fibers = set([
            p.source_id for p in pairs
            if p.source_type == "cell" and p.target_type == "fiber"
        ])
        
        coupling = len(cell_ids_with_fibers) / len(cells) if cells else 0.0
        
        return float(coupling)
    
    def _compute_network_connectivity(self, fibers: List) -> float:
        """Compute fiber network connectivity."""
        # Simplified: based on fiber proximity
        # In full implementation, use graph-based metrics
        
        if len(fibers) < 2:
            return 0.0
        
        from scipy.spatial import cKDTree
        
        # Get fiber midpoints
        midpoints = []
        for fiber in fibers:
            if len(fiber.centerline) > 0:
                mid_idx = len(fiber.centerline) // 2
                midpoints.append(fiber.centerline[mid_idx])
        
        if len(midpoints) < 2:
            return 0.0
        
        # Build KD-tree
        tree = cKDTree(midpoints)
        
        # Count fibers with neighbors within 50 microns
        neighbors = tree.query_ball_tree(tree, r=50.0)
        
        # Connectivity = mean number of neighbors
        connectivity = np.mean([len(n) - 1 for n in neighbors])  # -1 for self
        
        # Normalize (assume max 10 neighbors)
        connectivity_norm = min(connectivity / 10.0, 1.0)
        
        return float(connectivity_norm)
    
    def _compute_cluster_compactness(self, cells: List) -> float:
        """Compute cell cluster compactness."""
        if len(cells) < 2:
            return 0.0
        
        from scipy.spatial import ConvexHull
        
        # Get centroids
        centroids = np.array([c.centroid for c in cells])
        
        # Compute convex hull
        try:
            hull = ConvexHull(centroids)
            hull_area = hull.volume  # 2D area
            
            # Compactness = total cell area / hull area
            total_cell_area = sum(c.area for c in cells)
            compactness = total_cell_area / hull_area if hull_area > 0 else 0.0
            
            return float(compactness)
        except:
            return 0.0
    
    def _compute_invasion_directionality(
        self,
        pairs: List[InteractionPair]
    ) -> float:
        """Compute invasion directionality from TACS-3 fibers."""
        tacs3_pairs = [p for p in pairs if p.interaction_type == 'TACS-3']
        
        if not tacs3_pairs:
            return 0.0
        
        # Compute mean angle to boundary normal
        angles = [
            p.angle_to_boundary_normal for p in tacs3_pairs
            if p.angle_to_boundary_normal is not None
        ]
        
        if not angles:
            return 0.0
        
        # Directionality: lower angle variance = more directed
        angle_std = np.std(angles)
        
        # Normalize (assume max std = 30 degrees)
        directionality = max(0, 1.0 - angle_std / 30.0)
        
        return float(directionality)