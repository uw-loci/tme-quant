"""Library-wide exception hierarchy for tme_quant.

Ported from pycurvelets ``models/models.py``.
"""


class FiberAnalysisError(Exception):
    """Base exception for all tme_quant fiber analysis operations."""


class ROIProcessingError(FiberAnalysisError):
    """Raised when ROI construction or validation fails."""


class BoundaryAnalysisError(FiberAnalysisError):
    """Raised when boundary extraction or measurement fails."""


class FeatureExtractionError(FiberAnalysisError):
    """Raised when curvelet/fiber feature extraction fails."""


class ImageProcessingError(FiberAnalysisError):
    """Raised when image loading, resizing, or preprocessing fails."""


__all__ = [
    "FiberAnalysisError",
    "ROIProcessingError",
    "BoundaryAnalysisError",
    "FeatureExtractionError",
    "ImageProcessingError",
]
