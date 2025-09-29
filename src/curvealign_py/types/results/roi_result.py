"""
ROI analysis results.

This module defines the result structure for ROI-based analysis operations.
"""

from typing import NamedTuple, List, Dict, Any

from .analysis_result import AnalysisResult


class ROIResult(NamedTuple):
    """
    Results from ROI analysis.
    
    Attributes
    ----------
    roi_results : List[AnalysisResult]
        Results for each individual ROI
    comparison_stats : Dict[str, Any]
        Statistics comparing ROIs
    """
    roi_results: List[AnalysisResult]
    comparison_stats: Dict[str, Any]
