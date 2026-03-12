"""
TMEQuant - Tumor Microenvironment Quantification Platform

A comprehensive Python platform for quantitative analysis of the tumor
microenvironment, including:
    - Fiber analysis (collagen orientation and extraction)
    - Cell analysis (segmentation, classification, quantification)
    - TME analysis (TACS classification, prognostic scoring)
    - Image registration (multimodal microscopy alignment)

Example:
    >>> from tme_quant.fiber_analysis import FiberAnalyzer
    >>> from tme_quant.cell_analysis import CellAnalyzer
    >>> from tme_quant.tme_analysis import TMEAnalyzer
    >>> from tme_quant.image_registration import RegistrationManager
"""

from .fiber_analysis import FiberAnalyzer
from .cell_analysis import CellAnalyzer
from .tme_analysis import TMEAnalyzer
from .image_registration import RegistrationManager

__version__ = '1.0.0'
__author__ = 'TMEQuant Development Team'

__all__ = [
    'FiberAnalyzer',
    'CellAnalyzer',
    'TMEAnalyzer',
    'RegistrationManager',
]