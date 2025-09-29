"""
CurveAlign type system - organized by functional area.

This module provides all type definitions used in the CurveAlign API,
organized into logical categories:
- core: Fundamental data structures (Curvelet, Boundary, CtCoeffs)
- config: Configuration and option classes
- results: Result and output structures
"""

# Core data structures
from .core import Curvelet, Boundary, CtCoeffs

# Configuration classes
from .config import CurveAlignOptions, FeatureOptions

# Result structures
from .results import FeatureTable, BoundaryMetrics, AnalysisResult, ROIResult

__all__ = [
    # Core types
    'Curvelet', 'Boundary', 'CtCoeffs',
    # Configuration
    'CurveAlignOptions', 'FeatureOptions', 
    # Results
    'FeatureTable', 'BoundaryMetrics', 'AnalysisResult', 'ROIResult'
]