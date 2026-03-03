"""
TME analysis module for comprehensive tumor microenvironment quantification.

Provides analysis of spatial interactions between cells, fibers, and tumor boundaries
with focus on TACS (Tumor-Associated Collagen Signatures) classification.
"""

from .core.tme_analyzer import TMEAnalyzer

# Export main class
__all__ = [
    'TMEAnalyzer',
]

__version__ = '0.1.0'