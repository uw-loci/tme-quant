"""
Measurement engine for computing TME features from interaction pairs.

Computes:
    - TACS features (TACS-1/2/3 scores, distribution)
    - Morphological features (in interaction regions)
    - Spatial features (distance maps, distributions)
    - Orientation features (relative angles, alignment)
    - Density features (component densities)
    - Prognostic features (clinical outcome predictors)
"""

import numpy as np
from typing import List, Dict, Optional, Any
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from ..config.analysis_params import InteractionPair
from ...core.tme_models.cell_model import CellObject
from ...core.tme_models.fiber_model import FiberObject


class MeasurementEngine:
    """
    Compute measurements from interaction pairs and TME components.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize measurement engine."""
        self.verbose = verbose
    
    # ============================================================
    # TACS FEATURES
    # ============================================================
    
    def compute_tacs_features(
        self,
        interaction_pairs: List[InteractionPair],
        angle_threshold_perp: float = 30.0,
        angle_threshold_para: float = 60.0,
        straightness_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compute TACS features from fiber-tumor boundary interactions.
        
        TACS Classification:
            - TACS-1: Random, curly fibers (low straightness)
            - TACS-2: Straight fibers parallel to boundary (angle > 60°)
            - TACS-3: Straight fibers perpendicular to boundary (angle < 30°)
        
        Args:
            interaction_pairs: List of fiber-tumor interactions
            angle_threshold_perp: Threshold for perpendicular (TACS-3)
            angle_threshold_para: Threshold for parallel (TACS-2)
            straightness_threshold: Minimum straightness for TACS-2/3
            
        Returns:
            Dictionary of TACS features
        """
        # Filter for fiber-tumor boundary interactions
        fiber_tumor_pairs = [
            p for p in interaction_pairs
            if p.source_type == "fiber" and p.target_type == "tumor_boundary"
        ]
        
        if not fiber_tumor_pairs:
            return {
                'tacs1_count': 0,
                'tacs2_count': 0,
                'tacs3_count': 0,
                'tacs1_ratio': 0.0,
                'tacs2_ratio': 0.0,
                'tacs3_ratio': 0.0,
                'dominant_tacs_type': None,
                'mean_tacs_score': 0.0,
                'tacs_heterogeneity': 0.0,
            }
        
        # Count TACS types
        tacs_counts = Counter([p.interaction_type for p in fiber_tumor_pairs])
        
        total = len(fiber_tumor_pairs)
        tacs1_count = tacs_counts.get('TACS-1', 0)
        tacs2_count = tacs_counts.get('TACS-2', 0)
        tacs3_count = tacs_counts.get('TACS-3', 0)
        
        # Compute ratios
        tacs1_ratio = tacs1_count / total if total > 0 else 0.0
        tacs2_ratio = tacs2_count / total if total > 0 else 0.0
        tacs3_ratio = tacs3_count / total if total > 0 else 0.0
        
        # Dominant TACS type
        dominant_tacs = max(
            [('TACS-1', tacs1_count), ('TACS-2', tacs2_count), ('TACS-3', tacs3_count)],
            key=lambda x: x[1]
        )[0]
        
        # TACS score (weighted: TACS-3 > TACS-2 > TACS-1)
        # Higher score = more invasive
        tacs_score = (tacs1_ratio * 1.0 + tacs2_ratio * 2.0 + tacs3_ratio * 3.0)
        
        # TACS heterogeneity (entropy of distribution)
        tacs_probs = [tacs1_ratio, tacs2_ratio, tacs3_ratio]
        tacs_probs = [p for p in tacs_probs if p > 0]  # Remove zeros
        tacs_heterogeneity = entropy(tacs_probs) if tacs_probs else 0.0
        
        # Distance statistics
        distances = [p.distance for p in fiber_tumor_pairs]
        mean_distance = float(np.mean(distances)) if distances else 0.0
        std_distance = float(np.std(distances)) if distances else 0.0
        
        # Angle statistics (for TACS-2 and TACS-3)
        angles_to_normal = [
            p.angle_to_boundary_normal for p in fiber_tumor_pairs
            if p.angle_to_boundary_normal is not None
        ]
        mean_angle_to_normal = float(np.mean(angles_to_normal)) if angles_to_normal else 0.0
        
        features = {
            # TACS counts
            'tacs1_count': tacs1_count,
            'tacs2_count': tacs2_count,
            'tacs3_count': tacs3_count,
            'total_boundary_fibers': total,
            
            # TACS ratios
            'tacs1_ratio': tacs1_ratio,
            'tacs2_ratio': tacs2_ratio,
            'tacs3_ratio': tacs3_ratio,
            
            # Classification
            'dominant_tacs_type': dominant_tacs,
            
            # Scores
            'mean_tacs_score': tacs_score,
            'tacs_heterogeneity': tacs_heterogeneity,
            
            # Distance metrics
            'mean_distance_to_boundary': mean_distance,
            'std_distance_to_boundary': std_distance,
            
            # Angle metrics
            'mean_angle_to_normal': mean_angle_to_normal,
            
            # Individual scores
            'tacs1_score': tacs1_ratio * 1.0,  # Low risk
            'tacs2_score': tacs2_ratio * 2.0,  # Medium risk
            'tacs3_score': tacs3_ratio * 3.0,  # High risk (invasive)
        }
        
        if self.verbose:
            print(f"TACS Features:")
            print(f"  TACS-1: {tacs1_ratio:.1%} ({tacs1_count})")
            print(f"  TACS-2: {tacs2_ratio:.1%} ({tacs2_count})")
            print(f"  TACS-3: {tacs3_ratio:.1%} ({tacs3_count})")
            print(f"  Dominant: {dominant_tacs}")
        
        return features
    
    # ============================================================
    # MORPHOLOGICAL FEATURES
    # ============================================================
    
    def compute_morphological_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        interaction_pairs: List[InteractionPair]
    ) -> Dict[str, Any]:
        """
        Compute morphological features in interaction regions.
        
        Measures:
            - Cell morphology: area, circularity, eccentricity
            - Fiber morphology: length, width, straightness, curvature
            - Interaction zone metrics
        
        Args:
            cells: List of cells
            fibers: List of fibers
            interaction_pairs: Interaction pairs
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        # Get cells/fibers involved in interactions
        interacting_cell_ids = set([
            p.source_id for p in interaction_pairs if p.source_type == "cell"
        ] + [
            p.target_id for p in interaction_pairs if p.target_type == "cell"
        ])
        
        interacting_fiber_ids = set([
            p.source_id for p in interaction_pairs if p.source_type == "fiber"
        ] + [
            p.target_id for p in interaction_pairs if p.target_type == "fiber"
        ])
        
        # Cell morphology in interaction zones
        if cells:
            interacting_cells = [c for c in cells if c.object_id in interacting_cell_ids]
            
            if interacting_cells:
                areas = [c.area for c in interacting_cells]
                circularities = [c.circularity for c in interacting_cells]
                eccentricities = [c.eccentricity for c in interacting_cells]
                
                features.update({
                    'interaction_cell_count': len(interacting_cells),
                    'mean_cell_area': float(np.mean(areas)),
                    'std_cell_area': float(np.std(areas)),
                    'mean_cell_circularity': float(np.mean(circularities)),
                    'mean_cell_eccentricity': float(np.mean(eccentricities)),
                })
        
        # Fiber morphology in interaction zones
        if fibers:
            interacting_fibers = [f for f in fibers if f.object_id in interacting_fiber_ids]
            
            if interacting_fibers:
                lengths = [f.length for f in interacting_fibers if hasattr(f, 'length')]
                straightnesses = [
                    f.straightness for f in interacting_fibers
                    if hasattr(f, 'straightness')
                ]
                
                features.update({
                    'interaction_fiber_count': len(interacting_fibers),
                    'mean_fiber_length': float(np.mean(lengths)) if lengths else 0.0,
                    'std_fiber_length': float(np.std(lengths)) if lengths else 0.0,
                    'mean_fiber_straightness': float(np.mean(straightnesses)) if straightnesses else 0.0,
                })
        
        return features
    
    # ============================================================
    # SPATIAL FEATURES
    # ============================================================
    
    def compute_spatial_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        interaction_pairs: List[InteractionPair]
    ) -> Dict[str, Any]:
        """
        Compute spatial distribution features.
        
        Measures:
            - Distance distributions
            - Spatial clustering
            - Component organization
        
        Args:
            cells: List of cells
            fibers: List of fibers
            interaction_pairs: Interaction pairs
            
        Returns:
            Dictionary of spatial features
        """
        features = {}
        
        # Distance statistics from interaction pairs
        if interaction_pairs:
            distances = [p.distance for p in interaction_pairs]
            
            features.update({
                'mean_interaction_distance': float(np.mean(distances)),
                'median_interaction_distance': float(np.median(distances)),
                'std_interaction_distance': float(np.std(distances)),
                'min_interaction_distance': float(np.min(distances)),
                'max_interaction_distance': float(np.max(distances)),
            })
            
            # Interaction type distribution
            interaction_types = [
                f"{p.source_type}-{p.target_type}" for p in interaction_pairs
            ]
            type_counts = Counter(interaction_types)
            
            features['interaction_type_distribution'] = dict(type_counts)
        
        # Cell spatial distribution
        if cells and len(cells) > 1:
            centroids = np.array([c.centroid for c in cells])
            
            # Nearest neighbor distances
            distances_matrix = cdist(centroids, centroids)
            np.fill_diagonal(distances_matrix, np.inf)  # Ignore self
            nn_distances = distances_matrix.min(axis=1)
            
            features.update({
                'cell_mean_nn_distance': float(np.mean(nn_distances)),
                'cell_std_nn_distance': float(np.std(nn_distances)),
            })
            
            # Clustering coefficient (ratio of observed to expected NN distance)
            # Expected distance for random distribution
            if len(cells) > 0:
                # Approximate region area (bounding box)
                min_coords = centroids.min(axis=0)
                max_coords = centroids.max(axis=0)
                region_area = np.prod(max_coords - min_coords)
                
                if region_area > 0:
                    expected_nn_dist = 0.5 / np.sqrt(len(cells) / region_area)
                    clustering_coeff = np.mean(nn_distances) / expected_nn_dist
                    
                    features['cell_clustering_coefficient'] = float(clustering_coeff)
        
        # Fiber spatial organization
        if fibers and len(fibers) > 1:
            # Fiber alignment (if angle information available)
            angles = [f.angle for f in fibers if hasattr(f, 'angle') and f.angle is not None]
            
            if angles:
                # Circular variance (measure of alignment)
                angles_rad = np.radians(angles)
                mean_cos = np.mean(np.cos(2 * angles_rad))
                mean_sin = np.mean(np.sin(2 * angles_rad))
                
                alignment_score = np.sqrt(mean_cos**2 + mean_sin**2)
                
                features['fiber_alignment_score'] = float(alignment_score)
        
        return features
    
    # ============================================================
    # ORIENTATION FEATURES
    # ============================================================
    
    def compute_orientation_features(
        self,
        interaction_pairs: List[InteractionPair]
    ) -> Dict[str, Any]:
        """
        Compute orientation features from interactions.
        
        Measures:
            - Relative angles between components
            - Alignment distributions
            - Orientation coherence
        
        Args:
            interaction_pairs: Interaction pairs
            
        Returns:
            Dictionary of orientation features
        """
        features = {}
        
        # Filter pairs with angle information
        pairs_with_angles = [
            p for p in interaction_pairs
            if p.relative_angle is not None
        ]
        
        if not pairs_with_angles:
            return features
        
        angles = [p.relative_angle for p in pairs_with_angles]
        
        # Angle statistics
        features.update({
            'mean_relative_angle': float(np.mean(angles)),
            'median_relative_angle': float(np.median(angles)),
            'std_relative_angle': float(np.std(angles)),
        })
        
        # Alignment categories
        perpendicular_count = sum(1 for a in angles if a < 30)
        parallel_count = sum(1 for a in angles if a > 60)
        oblique_count = len(angles) - perpendicular_count - parallel_count
        
        total = len(angles)
        features.update({
            'perpendicular_ratio': perpendicular_count / total if total > 0 else 0.0,
            'parallel_ratio': parallel_count / total if total > 0 else 0.0,
            'oblique_ratio': oblique_count / total if total > 0 else 0.0,
        })
        
        # Orientation coherence (circular variance)
        angles_rad = np.radians(angles)
        mean_cos = np.mean(np.cos(2 * angles_rad))
        mean_sin = np.mean(np.sin(2 * angles_rad))
        coherence = np.sqrt(mean_cos**2 + mean_sin**2)
        
        features['orientation_coherence'] = float(coherence)
        
        return features
    
    # ============================================================
    # DENSITY FEATURES
    # ============================================================
    
    def compute_density_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        region_area: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute component density features.
        
        Args:
            cells: List of cells
            fibers: List of fibers
            region_area: Region area in square microns (optional)
            
        Returns:
            Dictionary of density features
        """
        features = {}
        
        # Estimate region area if not provided
        if region_area is None and (cells or fibers):
            # Use bounding box of all components
            all_points = []
            
            if cells:
                all_points.extend([c.centroid for c in cells])
            
            if fibers:
                for fiber in fibers:
                    if len(fiber.centerline) > 0:
                        all_points.extend(fiber.centerline)
            
            if all_points:
                points = np.array(all_points)
                min_coords = points.min(axis=0)
                max_coords = points.max(axis=0)
                region_area = np.prod(max_coords - min_coords)
        
        if region_area and region_area > 0:
            # Cell density (cells per mm²)
            if cells:
                cell_density = len(cells) / (region_area / 1e6)
                features['cell_density'] = float(cell_density)
            
            # Fiber density (fibers per mm²)
            if fibers:
                fiber_density = len(fibers) / (region_area / 1e6)
                features['fiber_density'] = float(fiber_density)
            
            # Combined density
            if cells and fibers:
                features['cell_fiber_ratio'] = len(cells) / len(fibers) if fibers else 0.0
        
        return features
    
    # ============================================================
    # PROGNOSTIC FEATURES
    # ============================================================
    
    def compute_prognostic_scores(
        self,
        tacs_features: Optional[Dict[str, Any]],
        spatial_features: Optional[Dict[str, Any]],
        interaction_pairs: List[InteractionPair]
    ) -> Dict[str, float]:
        """
        Compute prognostic features for clinical outcome prediction.
        
        Combines multiple feature types into composite scores:
            - Collagen Prognostic Score (CPS)
            - TME Interaction Score
            - Invasive Potential Score
        
        Args:
            tacs_features: TACS features
            spatial_features: Spatial features
            interaction_pairs: Interaction pairs
            
        Returns:
            Dictionary of prognostic scores
        """
        scores = {}
        
        # Collagen Prognostic Score (CPS)
        # Higher TACS-3 = worse prognosis
        if tacs_features:
            tacs3_score = tacs_features.get('tacs3_score', 0.0)
            tacs2_score = tacs_features.get('tacs2_score', 0.0)
            tacs_heterogeneity = tacs_features.get('tacs_heterogeneity', 0.0)
            
            # CPS: weighted combination
            # TACS-3 (high risk) + TACS-2 (medium) + heterogeneity
            cps = (
                tacs3_score * 0.5 +        # TACS-3 weight: 0.5
                tacs2_score * 0.3 +        # TACS-2 weight: 0.3
                tacs_heterogeneity * 0.2   # Heterogeneity weight: 0.2
            )
            
            scores['collagen_prognostic_score'] = float(cps)
            scores['tacs3_prognostic'] = float(tacs3_score)
        
        # TME Interaction Score
        # Higher interaction count and closer distances = more aggressive
        if interaction_pairs:
            n_interactions = len(interaction_pairs)
            
            distances = [p.distance for p in interaction_pairs]
            mean_distance = np.mean(distances) if distances else 0.0
            
            # Normalize interaction count (assume 100 interactions as baseline)
            interaction_norm = min(n_interactions / 100.0, 1.0)
            
            # Distance score (closer = higher score)
            # Assume 50 microns as baseline
            distance_score = max(0, 1.0 - mean_distance / 50.0)
            
            tme_score = (interaction_norm * 0.6 + distance_score * 0.4)
            
            scores['tme_interaction_score'] = float(tme_score)
        
        # Invasive Potential Score
        # Combines TACS-3, fiber alignment, and spatial clustering
        if tacs_features and spatial_features:
            tacs3_ratio = tacs_features.get('tacs3_ratio', 0.0)
            
            # Fiber alignment (higher = more organized = more invasive)
            fiber_alignment = spatial_features.get('fiber_alignment_score', 0.0)
            
            # Cell clustering (higher = more aggressive)
            clustering_coeff = spatial_features.get('cell_clustering_coefficient', 1.0)
            clustering_score = min(clustering_coeff, 2.0) / 2.0  # Normalize
            
            invasive_potential = (
                tacs3_ratio * 0.5 +
                fiber_alignment * 0.3 +
                clustering_score * 0.2
            )
            
            scores['invasive_potential_score'] = float(invasive_potential)
        
        # Mechanical Coupling Score
        # High density + high alignment = strong mechanical coupling
        if spatial_features:
            fiber_alignment = spatial_features.get('fiber_alignment_score', 0.0)
            
            # Interaction density (from interaction pairs)
            interaction_density = len(interaction_pairs) / 100.0  # Normalize
            interaction_density = min(interaction_density, 1.0)
            
            mechanical_coupling = (fiber_alignment * 0.6 + interaction_density * 0.4)
            
            scores['mechanical_coupling_score'] = float(mechanical_coupling)
        
        # Overall TME Risk Score (composite)
        if scores:
            # Average of all available scores
            risk_score = np.mean(list(scores.values()))
            scores['overall_tme_risk_score'] = float(risk_score)
        
        if self.verbose:
            print(f"Prognostic Scores:")
            for key, value in scores.items():
                print(f"  {key}: {value:.3f}")
        
        return scores
    
    # ============================================================
    # DISTANCE MAP COMPUTATION
    # ============================================================
    
    def compute_distance_maps(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        tumor_regions: Optional[List] = None,
        grid_size: Tuple[int, int] = (100, 100)
    ) -> Dict[str, np.ndarray]:
        """
        Compute distance maps for visualization.
        
        Creates heatmaps showing distance to:
            - Nearest cell
            - Nearest fiber
            - Tumor boundary
        
        Args:
            cells: List of cells
            fibers: List of fibers
            tumor_regions: List of tumor regions
            grid_size: Grid resolution (height, width)
            
        Returns:
            Dictionary of distance maps
        """
        distance_maps = {}
        
        # Determine spatial extent
        all_points = []
        
        if cells:
            all_points.extend([c.centroid for c in cells])
        
        if fibers:
            for fiber in fibers:
                if len(fiber.centerline) > 0:
                    all_points.extend(fiber.centerline)
        
        if not all_points:
            return distance_maps
        
        points = np.array(all_points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Create grid
        x = np.linspace(x_min, x_max, grid_size[1])
        y = np.linspace(y_min, y_max, grid_size[0])
        xx, yy = np.meshgrid(x, y)
        
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Distance to nearest cell
        if cells:
            cell_centroids = np.array([c.centroid for c in cells])
            cell_distances = cdist(grid_points, cell_centroids).min(axis=1)
            distance_maps['cell_distance'] = cell_distances.reshape(grid_size)
        
        # Distance to nearest fiber
        if fibers:
            fiber_midpoints = np.array([
                fiber.centerline[len(fiber.centerline)//2]
                for fiber in fibers if len(fiber.centerline) > 0
            ])
            
            if len(fiber_midpoints) > 0:
                fiber_distances = cdist(grid_points, fiber_midpoints).min(axis=1)
                distance_maps['fiber_distance'] = fiber_distances.reshape(grid_size)
        
        # Distance to tumor boundary
        if tumor_regions:
            # Compute distance to nearest tumor boundary
            min_tumor_distances = []
            
            for point in grid_points:
                min_dist = float('inf')
                
                for tumor in tumor_regions:
                    boundary = tumor.roi.boundary
                    from shapely.geometry import Point
                    dist = Point(point).distance(boundary)
                    min_dist = min(min_dist, dist)
                
                min_tumor_distances.append(min_dist)
            
            distance_maps['tumor_boundary_distance'] = np.array(
                min_tumor_distances
            ).reshape(grid_size)
        
        return distance_maps