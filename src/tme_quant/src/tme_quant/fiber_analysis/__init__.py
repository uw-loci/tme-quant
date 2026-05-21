"""
Fiber analysis module for collagen fiber quantification.

Provides orientation analysis and fiber extraction for microscopy images.
Supports CT-FIRE, CurveAlign, skeleton-based extraction, ridge detection,
structure tensor, and gradient-based orientation methods, plus TACS
classification.
"""

from .extraction import FiberExtractionAnalyzer, BaseExtractionMethod
from .orientation import FiberOrientationAnalyzer, BaseOrientationMethod
from .methods.ctfire import CTFireExtraction
from .methods.curvealign import CurveAlignOrientation
from .methods.skeleton import SkeletonExtractionMethod
from .methods.ridge_detection import RidgeDetectionMethod
from .methods.gradient import GradientOrientationMethod
from .methods.structure_tensor import StructureTensorMethod
from .methods.orientationj import OrientationJMethod
from .tacs import classify_fiber_tacs, classify_fiber_segment_tacs_like, get_tacs_color
from .config import (
    ExtractionParams, ExtractionResult,
    OrientationParams, OrientationResult,
    CTFireParams, CurveAlignParams,
    SkeletonParams, RidgeDetectionParams,
    OrientationJParams, GradientParams, StructureTensorParams,
    FiberProperties, FiberFeatureParams,
)
from .results import FiberAnalysisResult
from .utils.fiber_dataframe_utils import (
    build_fiber_structure_from_curvelets,
    compute_fiber_density_and_alignment,
    flatten_numeric,
)
from .utils.boundary_tif_utils import extract_tif_boundary, extract_boundary_coords_from_mask
from .visualization.draw_utils import generate_fiber_overlay, generate_fiber_heatmap

# Backwards-compatible alias (examples and external code may use FiberAnalyzer)
FiberAnalyzer = FiberExtractionAnalyzer

__all__ = [
    # Analyzers
    'FiberExtractionAnalyzer', 'FiberAnalyzer', 'BaseExtractionMethod',
    'FiberOrientationAnalyzer', 'BaseOrientationMethod',
    # Extraction methods
    'CTFireExtraction', 'SkeletonExtractionMethod', 'RidgeDetectionMethod',
    # Orientation methods
    'CurveAlignOrientation', 'GradientOrientationMethod',
    'StructureTensorMethod', 'OrientationJMethod',
    # TACS
    'classify_fiber_tacs', 'classify_fiber_segment_tacs_like', 'get_tacs_color',
    # Params (general)
    'ExtractionParams', 'ExtractionResult',
    'OrientationParams', 'OrientationResult',
    # Params (method-specific)
    'CTFireParams', 'CurveAlignParams',
    'SkeletonParams', 'RidgeDetectionParams',
    'OrientationJParams', 'GradientParams', 'StructureTensorParams',
    # Fiber data
    'FiberProperties', 'FiberFeatureParams', 'FiberAnalysisResult',
    # Utilities
    'build_fiber_structure_from_curvelets',
    'compute_fiber_density_and_alignment',
    'flatten_numeric',
    'extract_tif_boundary',
    'extract_boundary_coords_from_mask',
    # Visualization
    'generate_fiber_overlay', 'generate_fiber_heatmap',
]
