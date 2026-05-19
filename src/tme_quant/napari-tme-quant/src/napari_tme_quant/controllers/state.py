"""PluginState — single source of truth for all plugin data.

All widgets read and write state through controllers only;
no widget holds a direct reference to PluginState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

from tme_quant import (
    TMEHierarchy,
    TMEProject,
    FiberAnalysisResult,
    CurveAlignPipelineResult,
)
from tme_quant.cell_analysis import CellAnalysisResult
from tme_quant.tme_analysis.config import TMEAnalysisResult


class ImageType(Enum):
    FIBER = auto()
    CELL = auto()
    MASK = auto()
    TWO_CHANNEL = auto()
    UNKNOWN = auto()


@dataclass
class PluginState:
    """Centralised plugin state.  Mutated on the Qt main thread only."""

    project: Optional[TMEProject] = None
    hierarchy: TMEHierarchy = field(default_factory=TMEHierarchy)

    # Per-image raw results (before hierarchy commit)
    fiber_results: Dict[str, FiberAnalysisResult] = field(default_factory=dict)
    cell_results: Dict[str, CellAnalysisResult] = field(default_factory=dict)
    tme_results: Dict[str, TMEAnalysisResult] = field(default_factory=dict)

    # CurveAlign full-pipeline results (typed separately from FiberAnalysisResult)
    curvealign_pipeline_results: Dict[str, CurveAlignPipelineResult] = field(
        default_factory=dict
    )

    # Image metadata
    image_pairs: Dict[str, str] = field(default_factory=dict)   # fiber_id → cell_id
    image_types: Dict[str, ImageType] = field(default_factory=dict)
    active_image_id: Optional[str] = None

    # napari layer map: layer_name → TMEObject.object_id
    layer_map: Dict[str, str] = field(default_factory=dict)

    # Active parameter presets (one per analysis step)
    presets: Dict[str, dict] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear all transient state (keep hierarchy and project)."""
        self.fiber_results.clear()
        self.cell_results.clear()
        self.tme_results.clear()
        self.curvealign_pipeline_results.clear()
        self.layer_map.clear()
        self.active_image_id = None
