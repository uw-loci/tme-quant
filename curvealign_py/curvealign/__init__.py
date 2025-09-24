"""
CurveAlign Python API - Modern interface for collagen fiber analysis.

This package provides a comprehensive Python API for CurveAlign functionality,
featuring a clean separation of concerns:

- Core analysis algorithms (visualization-free)
- Organized type system (core, config, results)
- Pluggable visualization backends (matplotlib, napari, ImageJ)
- High-level user-facing API functions

Basic Usage
-----------
>>> import curvealign
>>> result = curvealign.analyze_image(image)
>>> print(f"Found {len(result.curvelets)} fiber segments")

With Visualization
------------------
>>> from curvealign.visualization import standalone
>>> overlay = standalone.create_overlay(image, result.curvelets)
"""

# High-level API functions (most commonly used)
from .api import (
    analyze_image, analyze_roi, batch_analyze,
    get_curvelets, reconstruct,
    overlay, angle_map
)

# Core types (commonly needed)
from .types import (
    # Core data structures
    Curvelet, Boundary, CtCoeffs,
    # Configuration
    CurveAlignOptions, FeatureOptions,
    # Results
    AnalysisResult, ROIResult, FeatureTable, BoundaryMetrics
)

# Core processing functions (for advanced users)
from .core.processors import (
    extract_curvelets as core_extract_curvelets,
    reconstruct_image as core_reconstruct_image,
    compute_features as core_compute_features,
    measure_boundary_alignment as core_measure_boundary_alignment
)

# Package metadata
__version__ = "0.1.0"
__author__ = "CurveAlign Development Team"
__email__ = "curvealign@loci.wisc.edu"

# Main public API
__all__ = [
    # High-level API
    'analyze_image', 'analyze_roi', 'batch_analyze',
    'get_curvelets', 'reconstruct',
    'overlay', 'angle_map',
    
    # Core types
    'Curvelet', 'Boundary', 'CtCoeffs',
    'CurveAlignOptions', 'FeatureOptions',
    'AnalysisResult', 'ROIResult', 'FeatureTable', 'BoundaryMetrics',
    
    # Advanced/core functions
    'core_extract_curvelets', 'core_reconstruct_image',
    'core_compute_features', 'core_measure_boundary_alignment',
    
    # Package info
    '__version__', '__author__', '__email__'
]