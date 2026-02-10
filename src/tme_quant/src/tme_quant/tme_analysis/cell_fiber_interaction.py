"""
Comprehensive cell-fiber interaction analysis engine.
"""
# Main engine for:
# - Finding candidate cell-fiber pairs
# - Calculating interaction metrics (distance, contact, angles)
# - Classifying interaction types
# - Calculating tumor-associated collagen features (TACS scores)
# - Network analysis of interactions

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from scipy.spatial import KDTree, distance_matrix
from scipy.ndimage import distance_transform_edt
import warnings

from ..core.tme_models.interaction_models import (
    CellFiberInteraction, InteractionNetwork, InteractionCategory
)
from ..core.tme_models.cell_model import Cell
from ..core.tme_models.fiber_model import CollagenFiber
from ..core.tme_models.tumor_model import TumorRegion
from ..utils.geometry_utils import calculate_distance_matrix
from ..utils.parallel_processing import parallel_apply


@dataclass
class InteractionAnalyzer:
    """
    Engine for analyzing cell-fiber interactions in TME.
    """
    
    def __init__(
        self,
        max_interaction_distance: float = 50.0,  # microns
        contact_threshold: float = 5.0,  # microns for physical contact
        parallel_processing: bool = True,
        verbose: bool = False
    ):
        self.max_distance = max_interaction_distance
        self.contact_threshold = contact_threshold
        self.parallel = parallel_processing
        self.verbose = verbose
        
        # Cache for performance
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self._interaction_cache: Dict[str, CellFiberInteraction] = {}
    
    def analyze_region(
        self,
        cells: List[Cell],
        fibers: List[CollagenFiber],
        region_context: Optional[TumorRegion] = None,
        pixel_size: Tuple[float, float] = (1.0, 1.0)
    ) -> InteractionNetwork:
        """
        Analyze all cell-fiber interactions in a region.
        
        Args:
            cells: List of cells in region
            fibers: List of fibers in region
            region_context: Tumor region context (optional)
            pixel_size: Pixel size in microns (x, y)
            
        Returns:
            InteractionNetwork containing all detected interactions
        """
        if self.verbose:
            print(f"Analyzing {len(cells)} cells and {len(fibers)} fibers")
        
        # Filter out cells/fibers without ROIs
        valid_cells = [c for c in cells if c.rois]
        valid_fibers = [f for f in fibers if f.rois]
        
        if self.verbose:
            print(f"Valid: {len(valid_cells)} cells, {len(valid_fibers)} fibers")
        
        # Step 1: Find candidate pairs using spatial indexing
        candidate_pairs = self._find_candidate_pairs(
            valid_cells, valid_fibers, pixel_size
        )
        
        if self.verbose:
            print(f"Found {len(candidate_pairs)} candidate pairs")
        
        # Step 2: Analyze each candidate pair
        interactions = []
        if self.parallel and len(candidate_pairs) > 100:
            # Parallel processing for large datasets
            interactions = self._analyze_pairs_parallel(
                candidate_pairs, region_context, pixel_size
            )
        else:
            # Serial processing
            for cell, fiber in candidate_pairs:
                interaction = self._analyze_pair(
                    cell, fiber, region_context, pixel_size
                )
                if interaction is not None:
                    interactions.append(interaction)
        
        if self.verbose:
            print(f"Found {len(interactions)} interactions")
        
        # Step 3: Create interaction network
        network = InteractionNetwork(
            name=f"Interaction_Network_{len(interactions)}",
            interactions=interactions
        )
        
        # Step 4: Build network graph and calculate metrics
        network.build_network_graph()
        
        # Step 5: Calculate region-level statistics
        self._calculate_region_statistics(network, region_context)
        
        return network
    
    def _find_candidate_pairs(
        self,
        cells: List[Cell],
        fibers: List[CollagenFiber],
        pixel_size: Tuple[float, float]
    ) -> List[Tuple[Cell, CollagenFiber]]:
        """Find candidate cell-fiber pairs using spatial indexing."""
        if not cells or not fibers:
            return []
        
        # Extract centroids
        cell_centroids = []
        for cell in cells:
            if cell.rois:
                centroid = cell.rois[0].centroid()
                cell_centroids.append(centroid)
        
        fiber_centroids = []
        for fiber in fibers:
            if fiber.rois:
                centroid = fiber.rois[0].centroid()
                fiber_centroids.append(centroid)
        
        if not cell_centroids or not fiber_centroids:
            return []
        
        # Convert to arrays
        cell_points = np.array(cell_centroids)
        fiber_points = np.array(fiber_centroids)
        
        # Scale by pixel size
        cell_points_scaled = cell_points * np.array(pixel_size)
        fiber_points_scaled = fiber_points * np.array(pixel_size)
        
        # Build KD-trees for efficient nearest neighbor search
        fiber_tree = KDTree(fiber_points_scaled)
        
        # Find cells near fibers within max distance
        candidate_pairs = []
        for i, cell_point in enumerate(cell_points_scaled):
            # Find fibers within max_distance
            indices = fiber_tree.query_ball_point(
                cell_point, 
                self.max_distance
            )
            
            for j in indices:
                # Calculate exact distance
                distance = np.linalg.norm(cell_point - fiber_points_scaled[j])
                if distance <= self.max_distance:
                    candidate_pairs.append((cells[i], fibers[j]))
        
        return candidate_pairs
    
    def _analyze_pair(
        self,
        cell: Cell,
        fiber: CollagenFiber,
        region_context: Optional[TumorRegion],
        pixel_size: Tuple[float, float]
    ) -> Optional[CellFiberInteraction]:
        """Analyze a single cell-fiber pair."""
        # Create interaction object
        interaction = CellFiberInteraction(cell=cell, fiber=fiber)
        
        # Set spatial context
        if region_context:
            interaction.within_tumor_region = True
            if hasattr(region_context, 'invasive_front'):
                interaction.at_invasive_front = region_context.invasive_front
        
        # Calculate interaction metrics
        metrics = interaction.calculate_interaction_metrics()
        
        # Filter by distance threshold
        distance = metrics.get('distance', float('inf'))
        if distance > self.max_distance:
            return None
        
        # Cache interaction
        cache_key = f"{cell.id}_{fiber.id}"
        self._interaction_cache[cache_key] = interaction
        self._distance_cache[(cell.id, fiber.id)] = distance
        
        return interaction
    
    def _analyze_pairs_parallel(
        self,
        candidate_pairs: List[Tuple[Cell, CollagenFiber]],
        region_context: Optional[TumorRegion],
        pixel_size: Tuple[float, float]
    ) -> List[CellFiberInteraction]:
        """Analyze pairs in parallel."""
        from ..utils.parallel_processing import ParallelProcessor
        
        processor = ParallelProcessor(n_jobs=-1)
        
        # Prepare arguments
        args = [
            (cell, fiber, region_context, pixel_size)
            for cell, fiber in candidate_pairs
        ]
        
        # Process in parallel
        results = processor.process(
            self._analyze_pair_wrapper,
            args,
            desc="Analyzing cell-fiber interactions"
        )
        
        # Filter out None results
        interactions = [r for r in results if r is not None]
        
        return interactions
    
    def _analyze_pair_wrapper(self, args):
        """Wrapper for parallel processing."""
        cell, fiber, region_context, pixel_size = args
        return self._analyze_pair(cell, fiber, region_context, pixel_size)
    
    def _calculate_region_statistics(
        self,
        network: InteractionNetwork,
        region_context: Optional[TumorRegion]
    ) -> None:
        """Calculate region-level interaction statistics."""
        if not network.interactions:
            return
        
        stats = {
            'total_interactions': len(network.interactions),
            'avg_interactions_per_cell': 0,
            'avg_interactions_per_fiber': 0,
            'physical_contact_ratio': 0,
            'tumor_associated_ratio': 0,
            'invasive_front_ratio': 0,
        }
        
        # Count unique cells and fibers
        unique_cells = set()
        unique_fibers = set()
        
        physical_contacts = 0
        tumor_associated = 0
        invasive_front = 0
        
        for interaction in network.interactions:
            unique_cells.add(interaction.cell.id)
            unique_fibers.add(interaction.fiber.id)
            
            if interaction.interaction_type == InteractionCategory.PHYSICAL_CONTACT:
                physical_contacts += 1
            
            if interaction.is_tumor_associated():
                tumor_associated += 1
            
            if interaction.at_invasive_front:
                invasive_front += 1
        
        # Calculate ratios
        if unique_cells:
            stats['avg_interactions_per_cell'] = (
                len(network.interactions) / len(unique_cells)
            )
        
        if unique_fibers:
            stats['avg_interactions_per_fiber'] = (
                len(network.interactions) / len(unique_fibers)
            )
        
        stats['physical_contact_ratio'] = (
            physical_contacts / len(network.interactions)
            if network.interactions else 0
        )
        
        stats['tumor_associated_ratio'] = (
            tumor_associated / len(network.interactions)
            if network.interactions else 0
        )
        
        stats['invasive_front_ratio'] = (
            invasive_front / len(network.interactions)
            if network.interactions else 0
        )
        
        # Distance statistics
        distances = [i.distance for i in network.interactions if i.distance]
        if distances:
            stats['avg_distance'] = np.mean(distances)
            stats['min_distance'] = np.min(distances)
            stats['max_distance'] = np.max(distances)
            stats['distance_std'] = np.std(distances)
        
        # Angle statistics
        angles = [i.angle for i in network.interactions if i.angle]
        if angles:
            stats['avg_angle'] = np.mean(angles)
            stats['angle_std'] = np.std(angles)
            
            # Alignment categories
            aligned = [a for a in angles if a < 30]
            perpendicular = [a for a in angles if a > 60]
            stats['aligned_ratio'] = len(aligned) / len(angles) if angles else 0
            stats['perpendicular_ratio'] = len(perpendicular) / len(angles) if angles else 0
        
        # Store in network
        network.add_measurement('region_statistics', stats)
        
        # Also store in region context if available
        if region_context:
            region_context.add_measurement('interaction_statistics', stats)
    
    def calculate_tumor_associated_collagen_features(
        self,
        network: InteractionNetwork,
        tumor_region: TumorRegion
    ) -> Dict[str, float]:
        """
        Calculate tumor-associated collagen features.
        
        These are prognostic features related to collagen organization
        around tumor cells.
        """
        features = {}
        
        # Get tumor-associated interactions
        tumor_interactions = [
            i for i in network.interactions
            if i.is_tumor_associated()
        ]
        
        if not tumor_interactions:
            return features
        
        # Tumor Associated Collagen Signatures (TACS)
        # TACS-1: Dense, randomly organized collagen
        # TACS-2: Thick, straightened collagen bundles
        # TACS-3: Aligned collagen fibers at tumor boundary
        
        # Calculate TACS scores
        tacs_scores = self._calculate_tacs_scores(tumor_interactions)
        features.update(tacs_scores)
        
        # Fiber alignment heterogeneity
        alignment_features = self._calculate_alignment_heterogeneity(
            tumor_interactions
        )
        features.update(alignment_features)
        
        # Contact pattern analysis
        contact_features = self._analyze_contact_patterns(tumor_interactions)
        features.update(contact_features)
        
        # Mechanical microenvironment features
        mechanical_features = self._calculate_mechanical_features(
            tumor_interactions
        )
        features.update(mechanical_features)
        
        # Prognostic scores
        prognostic_scores = self._calculate_prognostic_scores(features)
        features.update(prognostic_scores)
        
        return features
    
    def _calculate_tacs_scores(
        self,
        interactions: List[CellFiberInteraction]
    ) -> Dict[str, float]:
        """Calculate Tumor Associated Collagen Signature scores."""
        scores = {}
        
        if not interactions:
            return scores
        
        # Extract fiber properties
        fiber_orientations = []
        fiber_curvatures = []
        fiber_widths = []
        
        for interaction in interactions:
            fiber = interaction.fiber
            if fiber.orientation is not None:
                fiber_orientations.append(fiber.orientation)
            if fiber.curvature is not None:
                fiber_curvatures.append(fiber.curvature)
            if fiber.width is not None:
                fiber_widths.append(fiber.width)
        
        # TACS-1: Random organization (high orientation variance)
        if fiber_orientations:
            # Convert to circular variance
            from scipy.stats import circstd
            orientations_rad = np.radians(fiber_orientations)
            circular_std = circstd(orientations_rad)
            scores['tacs1_score'] = float(circular_std / (np.pi / np.sqrt(3)))  # Normalized 0-1
        
        # TACS-2: Bundle formation (low curvature, consistent width)
        if fiber_curvatures and fiber_widths:
            # Low curvature indicates straightened fibers
            avg_curvature = np.mean(fiber_curvatures)
            curvature_score = max(0, 1 - avg_curvature / 10.0)  # Normalized
            
            # Consistent width indicates bundle formation
            width_cv = np.std(fiber_widths) / np.mean(fiber_widths) if np.mean(fiber_widths) > 0 else 0
            consistency_score = max(0, 1 - width_cv)
            
            scores['tacs2_score'] = 0.6 * curvature_score + 0.4 * consistency_score
        
        # TACS-3: Boundary alignment (cells aligned with fibers)
        alignment_angles = [i.angle for i in interactions if i.angle is not None]
        if alignment_angles:
            # Low angles indicate alignment
            avg_alignment = np.mean(alignment_angles)
            alignment_score = max(0, 1 - avg_alignment / 90.0)
            
            # Also consider if at invasive front
            invasive_front_interactions = [
                i for i in interactions if i.at_invasive_front
            ]
            if invasive_front_interactions:
                invasive_angles = [i.angle for i in invasive_front_interactions if i.angle]
                if invasive_angles:
                    invasive_alignment = np.mean(invasive_angles)
                    invasive_score = max(0, 1 - invasive_alignment / 90.0)
                    alignment_score = 0.7 * alignment_score + 0.3 * invasive_score
            
            scores['tacs3_score'] = alignment_score
        
        return scores
    
    def _calculate_alignment_heterogeneity(
        self,
        interactions: List[CellFiberInteraction]
    ) -> Dict[str, float]:
        """Calculate heterogeneity in fiber alignment."""
        features = {}
        
        if not interactions:
            return features
        
        # Get interaction angles
        angles = [i.angle for i in interactions if i.angle is not None]
        if not angles:
            return features
        
        # Basic statistics
        features['alignment_mean'] = np.mean(angles)
        features['alignment_std'] = np.std(angles)
        features['alignment_cv'] = features['alignment_std'] / features['alignment_mean'] if features['alignment_mean'] > 0 else 0
        
        # Bimodality test (Hartigan's Dip Test)
        try:
            from scipy.stats import dip
            dip_stat, _ = dip(angles)
            features['alignment_bimodality'] = dip_stat
        except ImportError:
            pass
        
        # Entropy of angle distribution (discretized)
        hist, bins = np.histogram(angles, bins=18, range=(0, 180), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        features['alignment_entropy'] = entropy
        
        # Local vs global alignment
        # Compare fiber orientation variance within vs between interactions
        fiber_orientations = []
        for interaction in interactions:
            if interaction.fiber.orientation is not None:
                fiber_orientations.append(interaction.fiber.orientation)
        
        if len(fiber_orientations) > 1:
            global_variance = np.var(fiber_orientations)
            
            # Calculate local variance (within 20 micron radius groups)
            # Simplified implementation
            features['alignment_heterogeneity_index'] = global_variance
        
        return features
    
    def _analyze_contact_patterns(
        self,
        interactions: List[CellFiberInteraction]
    ) -> Dict[str, float]:
        """Analyze spatial patterns of cell-fiber contacts."""
        features = {}
        
        if not interactions:
            return features
        
        # Contact statistics
        contact_lengths = []
        contact_areas = []
        contact_percentages = []
        
        for interaction in interactions:
            if interaction.contact_length:
                contact_lengths.append(interaction.contact_length)
            if interaction.contact_area:
                contact_areas.append(interaction.contact_area)
            # Calculate contact percentage from measurements
            contact_perc = interaction.get_measurement('contact_percentage')
            if contact_perc:
                contact_percentages.append(contact_perc)
        
        if contact_lengths:
            features['avg_contact_length'] = np.mean(contact_lengths)
            features['contact_length_cv'] = (
                np.std(contact_lengths) / np.mean(contact_lengths)
                if np.mean(contact_lengths) > 0 else 0
            )
        
        if contact_areas:
            features['avg_contact_area'] = np.mean(contact_areas)
        
        if contact_percentages:
            features['avg_contact_percentage'] = np.mean(contact_percentages)
        
        # Contact type distribution
        contact_types = {}
        for interaction in interactions:
            int_type = interaction.interaction_type.value
            contact_types[int_type] = contact_types.get(int_type, 0) + 1
        
        total = len(interactions)
        for int_type, count in contact_types.items():
            features[f'ratio_{int_type}'] = count / total if total > 0 else 0
        
        # Spatial clustering of contacts
        # Calculate if contacts are clustered or uniformly distributed
        cell_positions = []
        for interaction in interactions:
            if interaction.cell.rois:
                centroid = interaction.cell.rois[0].centroid()
                cell_positions.append(centroid)
        
        if len(cell_positions) > 2:
            # Calculate nearest neighbor distances
            from scipy.spatial import cKDTree
            positions_array = np.array(cell_positions)
            tree = cKDTree(positions_array)
            distances, _ = tree.query(positions_array, k=2)  # k=2 for self + nearest
            nn_distances = distances[:, 1]  # Exclude self
            
            # Clark-Evans test for spatial randomness
            area = 10000  # Approximate area (simplified)
            density = len(cell_positions) / area
            expected_nn_distance = 1 / (2 * np.sqrt(density)) if density > 0 else 0
            observed_nn_distance = np.mean(nn_distances)
            
            features['spatial_clustering_index'] = (
                observed_nn_distance / expected_nn_distance
                if expected_nn_distance > 0 else 0
            )
        
        return features
    
    def _calculate_mechanical_features(
        self,
        interactions: List[CellFiberInteraction]
    ) -> Dict[str, float]:
        """Calculate mechanical microenvironment features."""
        features = {}
        
        if not interactions:
            return features
        
        # Extract mechanical scores
        invasive_scores = []
        guidance_scores = []
        coupling_scores = []
        
        for interaction in interactions:
            if interaction.invasive_potential_score:
                invasive_scores.append(interaction.invasive_potential_score)
            if interaction.migration_guidance_score:
                guidance_scores.append(interaction.migration_guidance_score)
            if interaction.mechanical_coupling_score:
                coupling_scores.append(interaction.mechanical_coupling_score)
        
        if invasive_scores:
            features['avg_invasive_potential'] = np.mean(invasive_scores)
            features['max_invasive_potential'] = np.max(invasive_scores)
        
        if guidance_scores:
            features['avg_migration_guidance'] = np.mean(guidance_scores)
        
        if coupling_scores:
            features['avg_mechanical_coupling'] = np.mean(coupling_scores)
        
        # Fiber stiffness proxy (width * straightness)
        stiffness_proxies = []
        for interaction in interactions:
            fiber = interaction.fiber
            if fiber.width is not None and fiber.straightness is not None:
                stiffness = fiber.width * fiber.straightness
                stiffness_proxies.append(stiffness)
        
        if stiffness_proxies:
            features['avg_fiber_stiffness_proxy'] = np.mean(stiffness_proxies)
        
        # Mechanical heterogeneity
        if invasive_scores and len(invasive_scores) > 1:
            features['mechanical_heterogeneity'] = np.std(invasive_scores)
        
        return features
    
    def _calculate_prognostic_scores(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate prognostic scores from interaction features."""
        scores = {}
        
        # Collagen Prognostic Score (CPS)
        # Combines TACS scores and alignment features
        tacs1 = features.get('tacs1_score', 0.5)
        tacs2 = features.get('tacs2_score', 0.5)
        tacs3 = features.get('tacs3_score', 0.5)
        
        # Weighted combination based on literature
        # TACS-3 (alignment at boundary) is most prognostic
        cps = 0.2 * tacs1 + 0.3 * tacs2 + 0.5 * tacs3
        scores['collagen_prognostic_score'] = cps
        
        # Interaction Complexity Score
        # Higher for diverse interaction types
        type_diversity = 0
        for key in features:
            if key.startswith('ratio_'):
                ratio = features[key]
                if ratio > 0:
                    type_diversity -= ratio * np.log2(ratio)
        scores['interaction_complexity_score'] = type_diversity
        
        # Mechanical Risk Score
        invasive_potential = features.get('avg_invasive_potential', 0.5)
        coupling = features.get('avg_mechanical_coupling', 0.5)
        scores['mechanical_risk_score'] = 0.6 * invasive_potential + 0.4 * coupling
        
        # Overall TME Interaction Score
        # Composite of all factors
        overall_score = (
            0.3 * cps +
            0.2 * scores['interaction_complexity_score'] +
            0.3 * scores['mechanical_risk_score'] +
            0.2 * (1 - features.get('alignment_heterogeneity_index', 0.5))
        )
        scores['tme_interaction_score'] = overall_score
        
        return scores

    """Enhanced with boundary-relative fiber analysis."""
    
    def calculate_tumor_associated_collagen_features(
        self,
        network: InteractionNetwork,
        tumor_region: TumorRegion,
        use_nearest_point: bool = True,
        pixel_size: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate TACS features using boundary-relative metrics.
        
        Args:
            network: Interaction network
            tumor_region: Tumor region with boundary
            use_nearest_point: Use nearest-point method for angles
            pixel_size: Pixel size in microns
        """
        features = {}
        
        # Get tumor-associated interactions
        tumor_interactions = [
            i for i in network.interactions
            if i.is_tumor_associated()
        ]
        
        if not tumor_interactions:
            return features
        
        # Compute boundary-relative metrics for all fibers if not already done
        if use_nearest_point:
            for interaction in tumor_interactions:
                fiber = interaction.fiber
                if fiber.nearest_boundary_point is None:
                    fiber.compute_boundary_relative_metrics(
                        tumor_region.roi, pixel_size
                    )
        
        # Calculate TACS scores with boundary-relative metrics
        tacs_scores = self._calculate_tacs_scores_boundary_relative(
            tumor_interactions, use_nearest_point
        )
        features.update(tacs_scores)
        
        # Rest of the analysis remains the same...
        alignment_features = self._calculate_alignment_heterogeneity(
            tumor_interactions
        )
        features.update(alignment_features)
        
        contact_features = self._analyze_contact_patterns(tumor_interactions)
        features.update(contact_features)
        
        mechanical_features = self._calculate_mechanical_features(
            tumor_interactions
        )
        features.update(mechanical_features)
        
        prognostic_scores = self._calculate_prognostic_scores(features)
        features.update(prognostic_scores)
        
        return features
    
    def _calculate_tacs_scores_boundary_relative(
        self,
        interactions: List[CellFiberInteraction],
        use_nearest_point: bool = True
    ) -> Dict[str, float]:
        """
        Calculate TACS scores using boundary-relative angles.
        
        NEW: Uses nearest-point method for more accurate classification.
        """
        scores = {}
        
        if not interactions:
            return scores
        
        # Extract fiber properties
        fiber_curvatures = []
        fiber_widths = []
        fiber_straightnesses = []
        
        # Angle metrics depend on method
        if use_nearest_point:
            angles_to_normal = []
            angles_to_tangent = []
            
            for interaction in interactions:
                fiber = interaction.fiber
                if fiber.relative_angle_to_boundary_normal is not None:
                    angles_to_normal.append(fiber.relative_angle_to_boundary_normal)
                if fiber.relative_angle_to_boundary_tangent is not None:
                    angles_to_tangent.append(fiber.relative_angle_to_boundary_tangent)
                
                if fiber.curvature is not None:
                    fiber_curvatures.append(fiber.curvature)
                if fiber.width is not None:
                    fiber_widths.append(fiber.width)
                if fiber.straightness is not None:
                    fiber_straightnesses.append(fiber.straightness)
        else:
            # Use original method
            fiber_orientations = []
            for interaction in interactions:
                fiber = interaction.fiber
                if fiber.orientation is not None:
                    fiber_orientations.append(fiber.orientation)
                if fiber.curvature is not None:
                    fiber_curvatures.append(fiber.curvature)
                if fiber.width is not None:
                    fiber_widths.append(fiber.width)
        
        # TACS-1: Random, curly collagen
        if fiber_straightnesses:
            # Low straightness indicates curly fibers
            mean_straightness = np.mean(fiber_straightnesses)
            curvature_score = 1 - mean_straightness
            
            if use_nearest_point and angles_to_normal:
                # Add orientation randomness
                # Random orientation = angles distributed around 45°
                angle_randomness = np.mean([abs(a - 45) for a in angles_to_normal])
                randomness_score = 1 - (angle_randomness / 45)
                scores['tacs1_score'] = 0.6 * curvature_score + 0.4 * randomness_score
            else:
                scores['tacs1_score'] = curvature_score
        
        # TACS-2: Straightened, parallel fibers
        if fiber_straightnesses and use_nearest_point and angles_to_tangent:
            # High straightness + parallel to boundary
            mean_straightness = np.mean(fiber_straightnesses)
            mean_parallel_angle = np.mean(angles_to_tangent)
            
            # Good TACS-2 = straight + parallel (angle to tangent < 30°)
            parallel_score = max(0, 1 - mean_parallel_angle / 30)
            scores['tacs2_score'] = 0.5 * mean_straightness + 0.5 * parallel_score
        
        # TACS-3: Perpendicular invasion fibers
        if use_nearest_point and angles_to_normal:
            # Low angle to normal = perpendicular to boundary
            mean_perp_angle = np.mean(angles_to_normal)
            perp_score = max(0, 1 - mean_perp_angle / 30)
            
            # Weight by straightness if available
            if fiber_straightnesses:
                mean_straightness = np.mean(fiber_straightnesses)
                scores['tacs3_score'] = 0.6 * perp_score + 0.4 * mean_straightness
            else:
                scores['tacs3_score'] = perp_score
            
            # Enhanced score for invasive front
            invasive_front_interactions = [
                i for i in interactions if i.at_invasive_front
            ]
            if invasive_front_interactions:
                invasive_angles = [
                    i.fiber.relative_angle_to_boundary_normal
                    for i in invasive_front_interactions
                    if i.fiber.relative_angle_to_boundary_normal is not None
                ]
                if invasive_angles:
                    invasive_perp = np.mean(invasive_angles)
                    invasive_score = max(0, 1 - invasive_perp / 30)
                    # Boost TACS-3 score for invasive front
                    scores['tacs3_score'] = 0.6 * scores['tacs3_score'] + 0.4 * invasive_score
        
        return scores