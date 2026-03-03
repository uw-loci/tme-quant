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

from .geometry_utils import (
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
    
    # Geometry utilities
    'compute_region_centroid',
    'compute_region_area',
    'point_in_polygon',
    'compute_convex_hull',
    'buffer_polygon',
    
    # Validation
    'validate_analysis_inputs',
    'validate_distance_threshold',
    'validate_interaction_pairs',
]