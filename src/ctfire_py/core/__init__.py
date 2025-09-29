"""
CT-FIRE core analysis modules.

This module provides the core CT-FIRE functionality organized into:
- algorithms: Low-level algorithmic functions (FIRE algorithm)
- processors: High-level processing orchestrators
"""

# Import high-level processors (most commonly used)
from .processors import (
    analyze_image_ctfire,
    analyze_fiber_network,
    compute_fiber_metrics,
    compute_ctfire_statistics
)

# Import specific algorithms for advanced users
from .algorithms import (
    extract_fibers_fire,
    enhance_image_with_curvelets
)

__all__ = [
    # High-level processors
    'analyze_image_ctfire', 'analyze_fiber_network', 
    'compute_fiber_metrics', 'compute_ctfire_statistics',
    # Low-level algorithms
    'extract_fibers_fire', 'enhance_image_with_curvelets'
]
