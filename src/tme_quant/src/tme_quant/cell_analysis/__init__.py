# Cell Analysis Module
"""
Cell analysis module for TME quantification.

Provides comprehensive cell segmentation, classification, and quantification
with integration into the TME project hierarchy.
"""

from .core.cell_analyzer import CellAnalyzer
from .core.segmentation_analyzer import CellSegmentationAnalyzer
from .core.classification_analyzer import CellClassificationAnalyzer
from .core.quantification_analyzer import CellQuantificationAnalyzer

# Export main classes
__all__ = [
    'CellAnalyzer',
    'CellSegmentationAnalyzer',
    'CellClassificationAnalyzer',
    'CellQuantificationAnalyzer',
]

__version__ = '0.1.0'