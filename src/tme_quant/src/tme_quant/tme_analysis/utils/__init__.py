"""
Utility functions for TME analysis.
"""

from .distance_utils import (
    compute_pairwise_distances,
    find_nearest_neighbors,
    compute_distance_to_boundary,
    points_within_distance,
    compute_distance_map,
)

from .statistical_utils import (
    compute_summary_statistics,
    compute_circular_statistics,
    test_spatial_randomness,
    bootstrap_confidence_interval,
)

# Geometry utilities live in core.geometry — re-exported here for convenience
from ...core.geometry import (
    compute_region_centroid,
    compute_region_area,
    point_in_polygon,
    compute_convex_hull,
    buffer_polygon,
)

from .validation import (
    validate_analysis_inputs,
    validate_distance_threshold,
    validate_interaction_pairs,
)

from .orientation_utils import (
    nearest_boundary_segment,
    discretize_roi_boundary,
    compute_orientation_relative_to_roi,
)

from ...fiber_analysis.utils.geometry_utils import (
    compute_boundary_tangent_angle,
    find_nearest_boundary_index,
    compute_relative_fiber_angles,
)

__all__ = [
    # Distance utilities
    'compute_pairwise_distances',
    'find_nearest_neighbors',
    'compute_distance_to_boundary',
    'points_within_distance',
    'compute_distance_map',
    # Statistical utilities
    'compute_summary_statistics',
    'compute_circular_statistics',
    'test_spatial_randomness',
    'bootstrap_confidence_interval',
    # Geometry utilities (from core.geometry)
    'compute_region_centroid',
    'compute_region_area',
    'point_in_polygon',
    'compute_convex_hull',
    'buffer_polygon',
    # Validation
    'validate_analysis_inputs',
    'validate_distance_threshold',
    'validate_interaction_pairs',
    # Pixel-map orientation utilities
    'nearest_boundary_segment',
    'discretize_roi_boundary',
    'compute_orientation_relative_to_roi',
    # Single-object relative-angle utilities (dense trace + polygon)
    'compute_boundary_tangent_angle',
    'find_nearest_boundary_index',
    'compute_relative_fiber_angles',
]
