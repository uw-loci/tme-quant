## Cell Analysis Core
"""
Core cell analysis classes.
"""

from .cell_analyzer import CellAnalyzer
from .segmentation_analyzer import CellSegmentationAnalyzer
from .classification_analyzer import CellClassificationAnalyzer
from .quantification_analyzer import CellQuantificationAnalyzer

__all__ = [
    'CellAnalyzer',
    'CellSegmentationAnalyzer',
    'CellClassificationAnalyzer',
    'CellQuantificationAnalyzer',
]

# Optional: result classes (module not yet created)
try:
    from .results import (
        SegmentationResult,
        ClassificationResult,
        QuantificationResult,
        CellAnalysisResult,
    )
    __all__ += [
        'SegmentationResult',
        'ClassificationResult',
        'QuantificationResult',
        'CellAnalysisResult',
    ]
except ImportError:
    pass