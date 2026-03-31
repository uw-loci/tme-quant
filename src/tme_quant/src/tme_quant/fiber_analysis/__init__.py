"""
Fiber analysis module for collagen fiber quantification.

Provides orientation analysis and fiber extraction for microscopy images.
Supports CT-FIRE, CurveAlign, skeleton-based extraction, ridge detection,
structure tensor, and gradient-based orientation methods, plus TACS
classification.
"""

from .extraction import FiberExtractionAnalyzer, BaseExtractionMethod
from .orientation import FiberOrientationAnalyzer, BaseOrientationMethod
from .ctfire import CTFireExtraction
from .curvealign import CurveAlignOrientation
from .tacs import classify_fiber_tacs, classify_fiber_segment_tacs_like, get_tacs_color
from .config import ExtractionParams, ExtractionResult, OrientationParams, OrientationResult
from .results import FiberAnalysisResult

# Backwards-compatible alias (examples and external code may use FiberAnalyzer)
FiberAnalyzer = FiberExtractionAnalyzer

__all__ = [
    'FiberExtractionAnalyzer', 'FiberAnalyzer', 'BaseExtractionMethod',
    'FiberOrientationAnalyzer', 'BaseOrientationMethod',
    'CTFireExtraction',
    'CurveAlignOrientation',
    'classify_fiber_tacs', 'classify_fiber_segment_tacs_like', 'get_tacs_color',
    'ExtractionParams', 'ExtractionResult',
    'OrientationParams', 'OrientationResult',
    'FiberAnalysisResult',
]
