"""
TME analysis module for comprehensive tumor microenvironment quantification.
"""

from .tme_analyzer import TMEAnalyzer
from .interaction_detector import InteractionDetector
from .interaction_network import InteractionNetworkAnalyzer
from .region_manager import RegionManager
from .measurement_engine import MeasurementEngine
from .pipelines import StandardTMEPipeline, InteractionAnalysisPipeline

__version__ = '0.1.0'

__all__ = [
    'TMEAnalyzer',
    'InteractionDetector',
    'InteractionNetworkAnalyzer',
    'RegionManager',
    'MeasurementEngine',
    'StandardTMEPipeline',
    'InteractionAnalysisPipeline',
]
