"""
Configuration and parameters for TME analysis.
"""

from .analysis_params import (
    # Enums
    AnalysisMode,
    InteractionStrategy,
    TumorDetectionMethod,
    
    # Parameters
    TMEAnalysisParams,
    TumorDetectionParams,
    
    # Results
    InteractionPair,
    TMEAnalysisResult,
)

__all__ = [
    # Enums
    'AnalysisMode',
    'InteractionStrategy',
    'TumorDetectionMethod',
    
    # Parameters
    'TMEAnalysisParams',
    'TumorDetectionParams',
    
    # Results
    'InteractionPair',
    'TMEAnalysisResult',
]