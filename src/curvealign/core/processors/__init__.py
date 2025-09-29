"""
High-level processing orchestrators for CurveAlign analysis.

This module provides processing classes and functions that coordinate
various algorithms to perform complete analysis workflows.
"""

from .curvelet_processor import extract_curvelets, reconstruct_image
from .feature_processor import compute_features
from .boundary_processor import measure_boundary_alignment

__all__ = [
    'extract_curvelets', 'reconstruct_image',
    'compute_features', 
    'measure_boundary_alignment'
]
