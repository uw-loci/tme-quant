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
FijiBridge               — unified Fiji / ImageJ subprocess bridge
"""

from .ctfire import CTFireExtraction
from .curvealign import CurveAlignOrientation
from .skeleton import SkeletonExtractionMethod
from .gradient import GradientOrientationMethod
from .structure_tensor import StructureTensorMethod
from .orientationj import OrientationJMethod
from .ridge_detection import RidgeDetectionMethod
from .fiji_bridge import FijiBridge

__all__ = [
    'CTFireExtraction',
    'CurveAlignOrientation',
    'SkeletonExtractionMethod',
    'GradientOrientationMethod',
    'StructureTensorMethod',
    'OrientationJMethod',
    'RidgeDetectionMethod',
    'FijiBridge',
]
