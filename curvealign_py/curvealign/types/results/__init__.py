"""
Result structures for CurveAlign analysis.

This module provides all result and output data structures returned
by CurveAlign analysis functions.
"""

from .feature_table import FeatureTable
from .boundary_metrics import BoundaryMetrics
from .analysis_result import AnalysisResult
from .roi_result import ROIResult

__all__ = ['FeatureTable', 'BoundaryMetrics', 'AnalysisResult', 'ROIResult']
