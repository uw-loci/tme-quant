"""PluginState — single source of truth for all plugin data.

All widgets read and write state through controllers only;
no widget holds a direct reference to PluginState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional

import numpy as np

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

    # Raw image arrays keyed by image_id (populated on add_image; used for
    # heatmap/overlay generation without hitting the napari layer API)
    images: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-image parameter snapshots: image_id → {step → params_dict}
    # e.g. state.per_image_params["fiber_001"]["curvealign_tacs"] = {"keep": 0.05, ...}
    # Preserved across reset() so re-runs use the same settings.
    per_image_params: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    # Absolute file paths for project save/restore: image_id → path string
    image_paths: Dict[str, str] = field(default_factory=dict)

    # Project output folder — auto-save target for results and figures
    project_dir: Optional[Path] = None

    def reset(self) -> None:
        """Clear transient analysis state.

        Preserves: hierarchy, project, image_types, image_pairs, per_image_params,
        image_paths, images (arrays still available for re-run without reload).
        Clears: analysis results, layer_map, active_image_id.
        """
        self.fiber_results.clear()
        self.cell_results.clear()
        self.tme_results.clear()
        self.curvealign_pipeline_results.clear()
        self.layer_map.clear()
        self.active_image_id = None
