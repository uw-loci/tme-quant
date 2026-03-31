"""
Cell analysis module for segmentation, classification, and quantification.

Supports StarDist, Cellpose, and custom segmentation methods with
marker-based and ML classification.
"""

from .cell_analyzer import CellAnalyzer
from .segmentation import CellSegmentationAnalyzer
from .classification import CellClassificationAnalyzer
from .quantification import CellQuantificationAnalyzer
from .results import CellAnalysisResult
from .config import (
    SegmentationParams,
    ClassificationParams,
    QuantificationParams,
)

__all__ = [
    'CellAnalyzer',
    'CellSegmentationAnalyzer',
    'CellClassificationAnalyzer',
    'CellQuantificationAnalyzer',
    'CellAnalysisResult',
    'SegmentationParams',
    'ClassificationParams',
    'QuantificationParams',
]
