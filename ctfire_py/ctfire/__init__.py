"""
CT-FIRE Python API - Modern interface for individual fiber analysis.

This package provides a comprehensive Python API for CT-FIRE functionality,
featuring individual fiber extraction, network analysis, and visualization:

- Individual fiber extraction using FIRE algorithm
- Curvelet-based image enhancement
- Fiber network connectivity analysis
- Detailed fiber metrics (length, width, angle, straightness)
- Integration with CurveAlign for combined analysis

Basic Usage
-----------
>>> import ctfire
>>> result = ctfire.analyze_image(image)
>>> print(f"Found {len(result.fibers)} individual fibers")

With Configuration
------------------
>>> options = ctfire.CTFireOptions(run_mode="ctfire", thresh_flen=50.0)
>>> result = ctfire.analyze_image(image, options=options)
"""

# High-level API functions (most commonly used)
from .api import (
    analyze_image, batch_analyze,
    extract_fibers, get_fiber_metrics, get_network_analysis
)

# Core types (commonly needed)
from .types import (
    # Core data structures
    Fiber, FiberNetwork, FiberGraph,
    # Configuration
    CTFireOptions,
    # Results
    CTFireResult, FiberMetrics
)

# Core processing functions (for advanced users)
from .core.processors import (
    analyze_image_ctfire as core_analyze_image_ctfire,
    analyze_fiber_network as core_analyze_fiber_network,
    compute_fiber_metrics as core_compute_fiber_metrics,
    compute_ctfire_statistics as core_compute_ctfire_statistics
)

# Package metadata
__version__ = "0.1.0"
__author__ = "CT-FIRE Development Team"
__email__ = "ctfire@loci.wisc.edu"

# Main public API
__all__ = [
    # High-level API
    'analyze_image', 'batch_analyze',
    'extract_fibers', 'get_fiber_metrics', 'get_network_analysis',
    
    # Core types
    'Fiber', 'FiberNetwork', 'FiberGraph',
    'CTFireOptions', 'CTFireResult', 'FiberMetrics',
    
    # Advanced/core functions
    'core_analyze_image_ctfire', 'core_analyze_fiber_network',
    'core_compute_fiber_metrics', 'core_compute_ctfire_statistics',
    
    # Package info
    '__version__', '__author__', '__email__'
]
