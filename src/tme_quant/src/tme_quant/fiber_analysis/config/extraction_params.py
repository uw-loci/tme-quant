"""Configuration parameters for fiber extraction analysis."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np


class ExtractionMode(Enum):
    """Fiber extraction methods."""
    CTFIRE = "ctfire"
    RIDGE_DETECTION = "ridge_detection"
    SKELETONIZATION = "skeletonization"
    FIBER_TRACING = "fiber_tracing"


@dataclass
class FiberData:
    """Properties of a single extracted fiber."""
    fiber_id: str = ""
    length: float = 0.0          # microns
    width: float = 0.0           # microns
    straightness: float = 1.0    # 0–1 (1 = perfectly straight)
    orientation: float = 0.0     # degrees
    coordinates: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Alias for backwards compatibility
FiberProperties = FiberData


@dataclass
class ExtractionParams:
    """Parameters for fiber extraction analysis."""
    mode: ExtractionMode = ExtractionMode.CTFIRE

    # Physical scale
    pixel_size: float = 1.0  # microns per pixel

    # Fiber filtering
    min_fiber_length: float = 10.0   # microns
    max_fiber_length: float = 1000.0  # microns

    # CT-FIRE specific
    ctfire_threshold: float = 0.1
    straightness_threshold: float = 0.5

    # Ridge detection specific
    ridge_scale: float = 1.0
    ridge_threshold: float = 0.1

    # Additional method-specific kwargs
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of a fiber extraction analysis run."""

    # Extracted fibers
    fibers: List[FiberData] = field(default_factory=list)

    # Summary statistics (populated by FiberExtractionAnalyzer)
    total_fiber_count: int = 0
    mean_fiber_length: float = 0.0
    mean_fiber_width: float = 0.0
    mean_straightness: float = 0.0

    # Metadata filled in by the analyzer
    dimension: str = "2D"
    mode: Optional[ExtractionMode] = None
    processing_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
