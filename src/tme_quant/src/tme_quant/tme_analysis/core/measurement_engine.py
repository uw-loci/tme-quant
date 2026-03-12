"""
Measurement engine for computing TME features from interaction pairs.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from shapely.geometry import Point, Polygon

from ..config.analysis_params import InteractionPair
from ...core.tme_models.cell_model import CellObject
from ...core.tme_models.fiber_model import FiberObject


class MeasurementEngine:
    """Compute measurements from interaction pairs and TME components."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def compute_tacs_features(
        self,
        interaction_pairs: List[InteractionPair],
        angle_threshold_perp: float = 30.0,
        angle_threshold_para: float = 60.0,
        straightness_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Compute TACS distribution features from fiber-tumor interactions."""
        del angle_threshold_perp, angle_threshold_para, straightness_threshold

        fiber_tumor_pairs = [
            p
            for p in interaction_pairs
            if p.source_type == "fiber" and p.target_type == "tumor_boundary"
        ]

        if not fiber_tumor_pairs:
            return {
                "tacs1_count": 0,
                "tacs2_count": 0,
                "tacs3_count": 0,
                "total_boundary_fibers": 0,
                "tacs1_ratio": 0.0,
                "tacs2_ratio": 0.0,
                "tacs3_ratio": 0.0,
                "dominant_tacs_type": None,
                "mean_tacs_score": 0.0,
                "tacs_heterogeneity": 0.0,
                "tacs1_score": 0.0,
                "tacs2_score": 0.0,
                "tacs3_score": 0.0,
            }

        tacs_counts = Counter([p.interaction_type for p in fiber_tumor_pairs])

        total = len(fiber_tumor_pairs)
        tacs1_count = tacs_counts.get("TACS-1", 0)
        tacs2_count = tacs_counts.get("TACS-2", 0)
        tacs3_count = tacs_counts.get("TACS-3", 0)

        tacs1_ratio = tacs1_count / total
        tacs2_ratio = tacs2_count / total
        tacs3_ratio = tacs3_count / total

        dominant_tacs = max(
            [("TACS-1", tacs1_count), ("TACS-2", tacs2_count), ("TACS-3", tacs3_count)],
            key=lambda x: x[1],
        )[0]

        tacs_score = tacs1_ratio * 1.0 + tacs2_ratio * 2.0 + tacs3_ratio * 3.0

        tacs_probs = [p for p in [tacs1_ratio, tacs2_ratio, tacs3_ratio] if p > 0]
        tacs_heterogeneity = entropy(tacs_probs) if tacs_probs else 0.0

        distances = [p.distance for p in fiber_tumor_pairs]
        mean_distance = float(np.mean(distances)) if distances else 0.0
        std_distance = float(np.std(distances)) if distances else 0.0

        angles_to_normal = [
            p.angle_to_boundary_normal for p in fiber_tumor_pairs if p.angle_to_boundary_normal is not None
        ]
        mean_angle_to_normal = float(np.mean(angles_to_normal)) if angles_to_normal else 0.0

        features = {
            "tacs1_count": tacs1_count,
            "tacs2_count": tacs2_count,
            "tacs3_count": tacs3_count,
            "total_boundary_fibers": total,
            "tacs1_ratio": tacs1_ratio,
            "tacs2_ratio": tacs2_ratio,
            "tacs3_ratio": tacs3_ratio,
            "dominant_tacs_type": dominant_tacs,
            "mean_tacs_score": tacs_score,
            "tacs_heterogeneity": float(tacs_heterogeneity),
            "mean_distance_to_boundary": mean_distance,
            "std_distance_to_boundary": std_distance,
            "mean_angle_to_normal": mean_angle_to_normal,
            "tacs1_score": tacs1_ratio * 1.0,
            "tacs2_score": tacs2_ratio * 2.0,
            "tacs3_score": tacs3_ratio * 3.0,
        }

        if self.verbose:
            print("TACS Features:")
            print(f"  TACS-1: {tacs1_ratio:.1%} ({tacs1_count})")
            print(f"  TACS-2: {tacs2_ratio:.1%} ({tacs2_count})")
            print(f"  TACS-3: {tacs3_ratio:.1%} ({tacs3_count})")
            print(f"  Dominant: {dominant_tacs}")

        return features

    def compute_morphological_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        interaction_pairs: List[InteractionPair],
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {}

        interacting_cell_ids = set(
            [p.source_id for p in interaction_pairs if p.source_type == "cell"]
            + [p.target_id for p in interaction_pairs if p.target_type == "cell"]
        )
        interacting_fiber_ids = set(
            [p.source_id for p in interaction_pairs if p.source_type == "fiber"]
            + [p.target_id for p in interaction_pairs if p.target_type == "fiber"]
        )

        if cells:
            interacting_cells = [c for c in cells if c.object_id in interacting_cell_ids]
            if interacting_cells:
                areas = [c.area for c in interacting_cells]
                circularities = [c.circularity for c in interacting_cells]
                eccentricities = [c.eccentricity for c in interacting_cells]
                features.update(
                    {
                        "interaction_cell_count": len(interacting_cells),
                        "mean_cell_area": float(np.mean(areas)),
                        "std_cell_area": float(np.std(areas)),
                        "mean_cell_circularity": float(np.mean(circularities)),
                        "mean_cell_eccentricity": float(np.mean(eccentricities)),
                    }
                )

        if fibers:
            interacting_fibers = [f for f in fibers if f.object_id in interacting_fiber_ids]
            if interacting_fibers:
                lengths = [f.length for f in interacting_fibers if hasattr(f, "length")]
                straightnesses = [f.straightness for f in interacting_fibers if hasattr(f, "straightness")]
                features.update(
                    {
                        "interaction_fiber_count": len(interacting_fibers),
                        "mean_fiber_length": float(np.mean(lengths)) if lengths else 0.0,
                        "std_fiber_length": float(np.std(lengths)) if lengths else 0.0,
                        "mean_fiber_straightness": float(np.mean(straightnesses)) if straightnesses else 0.0,
                    }
                )

        return features

    def compute_spatial_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        interaction_pairs: List[InteractionPair],
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {}

        if interaction_pairs:
            distances = [p.distance for p in interaction_pairs]
            features.update(
                {
                    "mean_interaction_distance": float(np.mean(distances)),
                    "median_interaction_distance": float(np.median(distances)),
                    "std_interaction_distance": float(np.std(distances)),
                    "min_interaction_distance": float(np.min(distances)),
                    "max_interaction_distance": float(np.max(distances)),
                }
            )
            interaction_types = [f"{p.source_type}-{p.target_type}" for p in interaction_pairs]
            features["interaction_type_distribution"] = dict(Counter(interaction_types))

        if cells and len(cells) > 1:
            centroids = np.array([c.centroid for c in cells])
            distances_matrix = cdist(centroids, centroids)
            np.fill_diagonal(distances_matrix, np.inf)
            nn_distances = distances_matrix.min(axis=1)
            features.update(
                {
                    "cell_mean_nn_distance": float(np.mean(nn_distances)),
                    "cell_std_nn_distance": float(np.std(nn_distances)),
                }
            )

            min_coords = centroids.min(axis=0)
            max_coords = centroids.max(axis=0)
            region_area = np.prod(max_coords - min_coords)
            if region_area > 0:
                expected_nn_dist = 0.5 / np.sqrt(len(cells) / region_area)
                features["cell_clustering_coefficient"] = float(np.mean(nn_distances) / expected_nn_dist)

        if fibers and len(fibers) > 1:
            angles = [f.angle for f in fibers if hasattr(f, "angle") and f.angle is not None]
            if angles:
                angles_rad = np.radians(angles)
                mean_cos = np.mean(np.cos(2 * angles_rad))
                mean_sin = np.mean(np.sin(2 * angles_rad))
                features["fiber_alignment_score"] = float(np.sqrt(mean_cos**2 + mean_sin**2))

        return features

    def compute_orientation_features(self, interaction_pairs: List[InteractionPair]) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        pairs_with_angles = [p for p in interaction_pairs if p.relative_angle is not None]
        if not pairs_with_angles:
            return features

        angles = [p.relative_angle for p in pairs_with_angles]
        features.update(
            {
                "mean_relative_angle": float(np.mean(angles)),
                "median_relative_angle": float(np.median(angles)),
                "std_relative_angle": float(np.std(angles)),
            }
        )

        perpendicular_count = sum(1 for a in angles if a < 30)
        parallel_count = sum(1 for a in angles if a > 60)
        oblique_count = len(angles) - perpendicular_count - parallel_count
        total = len(angles)
        features.update(
            {
                "perpendicular_ratio": perpendicular_count / total if total > 0 else 0.0,
                "parallel_ratio": parallel_count / total if total > 0 else 0.0,
                "oblique_ratio": oblique_count / total if total > 0 else 0.0,
            }
        )

        angles_rad = np.radians(angles)
        mean_cos = np.mean(np.cos(2 * angles_rad))
        mean_sin = np.mean(np.sin(2 * angles_rad))
        features["orientation_coherence"] = float(np.sqrt(mean_cos**2 + mean_sin**2))
        return features

    def compute_density_features(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        region_area: Optional[float] = None,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {}

        if region_area is None and (cells or fibers):
            all_points: List[Any] = []
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
            if cells:
                features["cell_density"] = float(len(cells) / (region_area / 1e6))
            if fibers:
                features["fiber_density"] = float(len(fibers) / (region_area / 1e6))
            if cells and fibers:
                features["cell_fiber_ratio"] = len(cells) / len(fibers) if fibers else 0.0

        return features

    def compute_prognostic_scores(
        self,
        tacs_features: Optional[Dict[str, Any]],
        spatial_features: Optional[Dict[str, Any]],
        interaction_pairs: List[InteractionPair],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        if tacs_features:
            tacs3_score = tacs_features.get("tacs3_score", 0.0)
            tacs2_score = tacs_features.get("tacs2_score", 0.0)
            tacs_heterogeneity = tacs_features.get("tacs_heterogeneity", 0.0)
            cps = tacs3_score * 0.5 + tacs2_score * 0.3 + tacs_heterogeneity * 0.2
            scores["collagen_prognostic_score"] = float(cps)
            scores["tacs3_prognostic"] = float(tacs3_score)

        if interaction_pairs:
            n_interactions = len(interaction_pairs)
            distances = [p.distance for p in interaction_pairs]
            mean_distance = np.mean(distances) if distances else 0.0
            interaction_norm = min(n_interactions / 100.0, 1.0)
            distance_score = max(0.0, 1.0 - mean_distance / 50.0)
            scores["tme_interaction_score"] = float(interaction_norm * 0.6 + distance_score * 0.4)

        if tacs_features and spatial_features:
            tacs3_ratio = tacs_features.get("tacs3_ratio", 0.0)
            fiber_alignment = spatial_features.get("fiber_alignment_score", 0.0)
            clustering_coeff = spatial_features.get("cell_clustering_coefficient", 1.0)
            clustering_score = min(clustering_coeff, 2.0) / 2.0
            scores["invasive_potential_score"] = float(
                tacs3_ratio * 0.5 + fiber_alignment * 0.3 + clustering_score * 0.2
            )

        if spatial_features:
            fiber_alignment = spatial_features.get("fiber_alignment_score", 0.0)
            interaction_density = min(len(interaction_pairs) / 100.0, 1.0)
            scores["mechanical_coupling_score"] = float(fiber_alignment * 0.6 + interaction_density * 0.4)

        if scores:
            scores["overall_tme_risk_score"] = float(np.mean(list(scores.values())))

        if self.verbose:
            print("Prognostic Scores:")
            for key, value in scores.items():
                print(f"  {key}: {value:.3f}")

        return scores

    def compute_distance_maps(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        tumor_regions: Optional[List[Any]] = None,
        grid_size: Tuple[int, int] = (100, 100),
    ) -> Dict[str, np.ndarray]:
        distance_maps: Dict[str, np.ndarray] = {}

        all_points: List[Any] = []
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

        x = np.linspace(x_min, x_max, grid_size[1])
        y = np.linspace(y_min, y_max, grid_size[0])
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        if cells:
            cell_centroids = np.array([c.centroid for c in cells])
            cell_distances = cdist(grid_points, cell_centroids).min(axis=1)
            distance_maps["cell_distance"] = cell_distances.reshape(grid_size)

        if fibers:
            fiber_midpoints = np.array(
                [fiber.centerline[len(fiber.centerline) // 2] for fiber in fibers if len(fiber.centerline) > 0]
            )
            if len(fiber_midpoints) > 0:
                fiber_distances = cdist(grid_points, fiber_midpoints).min(axis=1)
                distance_maps["fiber_distance"] = fiber_distances.reshape(grid_size)

        if tumor_regions:
            tumor_polygons: List[Polygon] = []
            for tumor in tumor_regions:
                if hasattr(tumor, "roi") and hasattr(tumor.roi, "polygon"):
                    tumor_polygons.append(tumor.roi.polygon)
                elif hasattr(tumor, "geometry") and hasattr(tumor.geometry, "coordinates"):
                    tumor_polygons.append(Polygon(tumor.geometry.coordinates))

            if tumor_polygons:
                min_tumor_distances = []
                for point in grid_points:
                    pt = Point(point)
                    min_tumor_distances.append(min(pt.distance(poly.boundary) for poly in tumor_polygons))
                distance_maps["tumor_boundary_distance"] = np.array(min_tumor_distances).reshape(grid_size)

        return distance_maps
