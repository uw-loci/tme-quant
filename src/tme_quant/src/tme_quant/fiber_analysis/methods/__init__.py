"""
Fiber analysis method implementations.

Each module provides a concrete subclass of BaseExtractionMethod or
BaseOrientationMethod.  Use FiberExtractionAnalyzer / FiberOrientationAnalyzer
(in the parent package) to select and run methods by name.

Extraction methods
------------------
CTFireExtraction         — curvelet-based individual fiber extraction
RidgeDetectionMethod     — Fiji Ridge Detection plugin (NumPy fallback)
SkeletonExtractionMethod — skeletonization-based centerline extraction

Orientation methods
-------------------
CurveAlignOrientation    — curvelet-based pixel orientation maps
GradientOrientationMethod — Sobel/Scharr gradient orientation
StructureTensorMethod    — windowed structure tensor orientation
OrientationJMethod       — Fiji OrientationJ plugin (NumPy fallback)

External-tool bridge
--------------------
FijiBridge has moved to ``tme_quant.integrations.fiji_bridge``.
Import it from there: ``from tme_quant.integrations import FijiBridge``.
"""

from .ctfire import CTFireExtraction
from .curvealign import CurveAlignOrientation
from .skeleton import SkeletonExtractionMethod
from .gradient import GradientOrientationMethod
from .structure_tensor import StructureTensorMethod
from .orientationj import OrientationJMethod
from .ridge_detection import RidgeDetectionMethod

__all__ = [
    'CTFireExtraction',
    'CurveAlignOrientation',
    'SkeletonExtractionMethod',
    'GradientOrientationMethod',
    'StructureTensorMethod',
    'OrientationJMethod',
    'RidgeDetectionMethod',
]
