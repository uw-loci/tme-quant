"""
High-level TME analysis pipelines.
"""

from .standard_tme_pipeline import StandardTMEPipeline
from .interaction_analysis_pipeline import InteractionAnalysisPipeline
from .tacs_pipeline import analyze_tacs_zone
from .curvealign_curveletsMode_pipeline import (
    CurveAlignPipelineResult,
    curvealign_curvelets_mode_pipeline,
)
from .curvealign_ctfireMode_pipeline import (
    CTFirePipelineResult,
    curvealign_ctfire_mode_pipeline,
)

# Backward-compatible alias — callers using curvealign_pipeline() still work.
curvealign_pipeline = curvealign_curvelets_mode_pipeline

__all__ = [
    'StandardTMEPipeline',
    'InteractionAnalysisPipeline',
    'analyze_tacs_zone',
    'CurveAlignPipelineResult',
    'curvealign_curvelets_mode_pipeline',
    'curvealign_pipeline',   # deprecated alias for curvealign_curvelets_mode_pipeline
    'CTFirePipelineResult',
    'curvealign_ctfire_mode_pipeline',
]
