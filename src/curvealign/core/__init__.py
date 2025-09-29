"""
CurveAlign core analysis modules.

This module provides the core analysis functionality organized into:
- algorithms: Low-level algorithmic functions
- processors: High-level processing orchestrators
"""

# Import high-level processors (most commonly used)
from .processors import (
    extract_curvelets, reconstruct_image,
    compute_features, measure_boundary_alignment
)

# Import specific algorithms for advanced users
from .algorithms import (
    apply_fdct, apply_ifdct, extract_parameters,
    threshold_coefficients_at_scale, create_empty_coeffs_like,
    extract_curvelets_from_coeffs, group_curvelets, 
    normalize_angles, filter_edge_curvelets
)

__all__ = [
    # High-level processors
    'extract_curvelets', 'reconstruct_image', 'compute_features', 'measure_boundary_alignment',
    # Low-level algorithms  
    'apply_fdct', 'apply_ifdct', 'extract_parameters',
    'threshold_coefficients_at_scale', 'create_empty_coeffs_like',
    'extract_curvelets_from_coeffs', 'group_curvelets', 
    'normalize_angles', 'filter_edge_curvelets'
]