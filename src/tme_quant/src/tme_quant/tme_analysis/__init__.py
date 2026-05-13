"""
TME analysis module for comprehensive tumor microenvironment quantification.
"""

from .tme_analyzer import TMEAnalyzer
from .interaction_detector import InteractionDetector
from .interaction_network import InteractionNetworkAnalyzer
from .region_manager import RegionManager
from .measurement_engine import MeasurementEngine
from .pipelines import (
    StandardTMEPipeline,
    InteractionAnalysisPipeline,
    CurveAlignPipelineResult,
    curvealign_curvelets_mode_pipeline,
    analyze_tacs_zone,
)
from .config import TMEAnalysisParams, TMEAnalysisResult, AnalysisMode
from .utils.alignment_utils import compute_fiber_alignment_to_roi
from .utils.orientation_utils import compute_orientation_relative_to_roi

__version__ = '0.1.0'

__all__ = [
    # Analyzers / orchestrators
    'TMEAnalyzer',
    'InteractionDetector',
    'InteractionNetworkAnalyzer',
    'RegionManager',
    'MeasurementEngine',
    # Pipelines
    'StandardTMEPipeline',
    'InteractionAnalysisPipeline',
    'CurveAlignPipelineResult',
    'curvealign_curvelets_mode_pipeline',
    'analyze_tacs_zone',
    # Config
    'TMEAnalysisParams',
    'TMEAnalysisResult',
    'AnalysisMode',
    # Spatial utilities
    'compute_fiber_alignment_to_roi',
    'compute_orientation_relative_to_roi',
]
