"""Configuration parameters for fiber orientation analysis."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

import numpy as np


class OrientationMode(Enum):
    """Fiber orientation analysis methods."""
    CURVEALIGN = "curvealign"
    ORIENTATIONJ = "orientationj"
    GRADIENT = "gradient"
    STRUCTURE_TENSOR = "structure_tensor"


@dataclass
class OrientationParams:
    """Parameters for fiber orientation analysis."""
    mode: OrientationMode = OrientationMode.CURVEALIGN

    # Window / patch settings
    window_size: int = 128

    # CurveAlign / curvelet settings
    curvelet_levels: int = 4
    curvelet_angles: int = 16

    # Physical scale
    pixel_size: float = 1.0  # microns per pixel

    # Output options
    compute_coherency: bool = True
    compute_energy: bool = False

    # Additional method-specific kwargs
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrientationResult:
    """Result of a fiber orientation analysis run."""

    # Orientation map (degrees, same shape as input image)
    orientation_map: Optional[np.ndarray] = None
    coherency_map: Optional[np.ndarray] = None
    energy_map: Optional[np.ndarray] = None

    # Summary statistics
    mean_orientation: float = 0.0
    alignment_score: float = 0.0
    orientation_distribution: Optional[np.ndarray] = None

    # Metadata filled in by the analyzer
    dimension: str = "2D"
    mode: Optional[OrientationMode] = None
    processing_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_dominant_orientation(self) -> float:
        """Return mean orientation angle in degrees."""
        return self.mean_orientation

    def get_alignment_score(self) -> float:
        """Return circular alignment score (0–1)."""
        return self.alignment_score
