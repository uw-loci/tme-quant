"""
Core algorithms for CurveAlign analysis.

This module provides low-level algorithmic functions organized by category:
- FDCT operations and transforms
- Coefficient processing and thresholding  
- Curvelet extraction and grouping
"""

from .fdct_wrapper import apply_fdct, apply_ifdct, extract_parameters
from .coefficient_processing import (
    threshold_coefficients_at_scale, 
    create_empty_coeffs_like,
    get_nonzero_coefficient_positions
)
from .curvelet_extraction import (
    extract_curvelets_from_coeffs,
    group_curvelets, 
    normalize_angles,
    filter_edge_curvelets
)

__all__ = [
    # FDCT operations
    'apply_fdct', 'apply_ifdct', 'extract_parameters',
    # Coefficient processing
    'threshold_coefficients_at_scale', 'create_empty_coeffs_like', 'get_nonzero_coefficient_positions',
    # Curvelet extraction
    'extract_curvelets_from_coeffs', 'group_curvelets', 'normalize_angles', 'filter_edge_curvelets'
]
