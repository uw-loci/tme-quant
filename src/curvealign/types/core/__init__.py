"""
Core data structures for CurveAlign.

This module provides fundamental data types used throughout the API.
"""

from .curvelet import Curvelet
from .boundary import Boundary
from .coefficients import CtCoeffs

__all__ = ['Curvelet', 'Boundary', 'CtCoeffs']
