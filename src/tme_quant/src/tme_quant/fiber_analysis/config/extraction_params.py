"""
Fiber extraction parameters and result classes.

Design
------
Mirrors the orientation_params module: each ExtractionMode has its own
parameter subclass and result subclass.  The base classes define only
the fields that are common to every extraction method.

Class hierarchy
---------------
ExtractionParams              — base: mode, pixel_size, size filters,
                                measure flags
  ├── CTFireParams            — ctfire_threshold, ctfire_n_levels,
  │                             ctfire_n_angles, straightness_threshold,
  │                             use_matlab_backend
  ├── RidgeDetectionParams    — ridge_sigma, lower_threshold,
  │                             upper_threshold, extend_line,
  │                             correct_position
  └── SkeletonParams          — skeleton_method, min_branch_length,
                                smooth_skeleton

ExtractionResult              — base: fibers, summary statistics,
                                provenance metadata
  ├── CTFireResult            — fiber_mask, labeled_fibers,
  │                             curvelet_energy_map, n_candidates
  ├── RidgeDetectionResult    — line_width_map, junction_points
  └── SkeletonResult          — skeleton_mask, branch_points,
                                end_points

FiberProperties               — single fiber data object (mode-agnostic)

Factory methods
---------------
ExtractionParams.for_mode(mode, **kwargs)  →  correct subclass
ExtractionResult.for_mode(mode, **kwargs)  →  correct subclass

Backward compatibility
----------------------
Existing code that constructs ``ExtractionParams(mode=..., ctfire_threshold=...)``
continues to work — ExtractionParams is the common base and all subclass
fields have defaults.  The factory is the recommended path for new code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Mode enum
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionMode(Enum):
    """Supported individual fiber extraction algorithms."""
    CTFIRE          = "ctfire"           # Curvelet-based (CT-FIRE)
    RIDGE_DETECTION = "ridge_detection"  # Fiji Ridge Detection plugin
    SKELETON        = "skeleton"         # Skeletonization-based


# ─────────────────────────────────────────────────────────────────────────────
# FiberProperties  (single fiber, mode-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FiberProperties:
    """
    Geometric and morphological properties of one extracted fiber.

    This class is the same regardless of which extraction method produced
    the fiber.  All physical measurements are in µm (after pixel_size scaling).

    Attributes
    ----------
    fiber_id : int
        Unique integer identifier within the result.
    length : float
        Arc length along the centerline (µm).
    width : float
        Mean fiber width estimated from perpendicular intensity profiles (µm).
    straightness : float
        End-to-end distance / arc length ∈ [0, 1].  1 = perfectly straight.
    angle : float
        Orientation of the end-to-end vector in degrees (−90 to +90).
    curvature : float
        Mean absolute turning angle per µm (degrees/µm).
    centerline : ndarray or None
        ``(N, 2)`` array of (row, col) pixel coordinates tracing the fiber.
    boundary : ndarray or None
        Optional ``(M, 2)`` outline of the fiber body.
    aspect_ratio : float or None
        length / width.
    tortuosity : float or None
        arc_length / end_to_end_distance  (= 1 / straightness when > 0).
    confidence : float or None
        Detection confidence in [0, 1] if the method provides it.
    """
    fiber_id:    int   = 0
    length:      float = 0.0
    width:       float = 0.0
    straightness: float = 0.0
    angle:       float = 0.0
    curvature:   float = 0.0
    centerline:  Optional[np.ndarray] = None
    boundary:    Optional[np.ndarray] = None
    aspect_ratio: Optional[float]    = None
    tortuosity:  Optional[float]     = None
    confidence:  Optional[float]     = None

    @property
    def center_coordinates(self) -> np.ndarray:
        """Midpoint of the centerline as a (2,) array."""
        if self.centerline is None or len(self.centerline) == 0:
            return np.array([])
        return self.centerline[len(self.centerline) // 2]

    def to_dict(self) -> Dict[str, Any]:
        center = self.center_coordinates
        return {
            'fiber_id':     self.fiber_id,
            'center_x':     float(center[0]) if len(center) > 0 else None,
            'center_y':     float(center[1]) if len(center) > 1 else None,
            'length':       self.length,
            'width':        self.width,
            'straightness': self.straightness,
            'angle':        self.angle,
            'curvature':    self.curvature,
            'aspect_ratio': self.aspect_ratio,
            'tortuosity':   self.tortuosity,
            'confidence':   self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Base parameter class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionParams:
    """
    Base parameters shared by every fiber extraction mode.

    Use :meth:`for_mode` to get a correctly-typed subclass instance.

    Attributes
    ----------
    mode : ExtractionMode
        Which algorithm to run.
    pixel_size : float
        µm per pixel — used to convert pixel measurements to physical units.
    min_fiber_length : float
        Minimum fiber arc length (µm).  Shorter fibers are discarded.
    max_fiber_length : float
        Maximum fiber arc length (µm).  Longer fibers are discarded.
    min_fiber_width : float
        Minimum accepted fiber width (µm).
    max_fiber_width : float
        Maximum accepted fiber width (µm).
    measure_length : bool
        Whether to compute arc length.
    measure_width : bool
        Whether to estimate width from intensity profiles.
    measure_straightness : bool
        Whether to compute straightness (end-to-end / arc length).
    measure_angle : bool
        Whether to compute the end-to-end orientation angle.
    measure_curvature : bool
        Whether to compute mean curvature along the centerline.
    extract_centerlines : bool
        Whether to retain the full centerline coordinate array.
    """
    mode:       ExtractionMode = ExtractionMode.CTFIRE
    pixel_size: float          = 1.0

    # Size filters (µm) — applied after extraction
    min_fiber_length: float = 5.0
    max_fiber_length: float = 1000.0
    min_fiber_width:  float = 0.5
    max_fiber_width:  float = 20.0

    # Measurement flags
    measure_length:      bool = True
    measure_width:       bool = True
    measure_straightness: bool = True
    measure_angle:       bool = True
    measure_curvature:   bool = False
    extract_centerlines: bool = True

    @property
    def fiber_width_range(self) -> Tuple[float, float]:
        """Convenience tuple ``(min_fiber_width, max_fiber_width)``."""
        return (self.min_fiber_width, self.max_fiber_width)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_mode(cls, mode: ExtractionMode, **kwargs: Any) -> "ExtractionParams":
        """
        Return a correctly-typed parameter object for *mode*.

        Examples
        --------
        >>> p = ExtractionParams.for_mode(
        ...     ExtractionMode.CTFIRE, pixel_size=0.5, ctfire_threshold=0.15
        ... )
        >>> type(p).__name__
        'CTFireParams'

        >>> p = ExtractionParams.for_mode(
        ...     ExtractionMode.RIDGE_DETECTION, ridge_sigma=3.0
        ... )
        >>> type(p).__name__
        'RidgeDetectionParams'
        """
        _map: Dict[ExtractionMode, Type[ExtractionParams]] = {
            ExtractionMode.CTFIRE:          CTFireParams,
            ExtractionMode.RIDGE_DETECTION: RidgeDetectionParams,
            ExtractionMode.SKELETON:        SkeletonParams,
        }
        return _map.get(mode, cls)(mode=mode, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode':              self.mode.value,
            'pixel_size':        self.pixel_size,
            'min_fiber_length':  self.min_fiber_length,
            'max_fiber_length':  self.max_fiber_length,
            'min_fiber_width':   self.min_fiber_width,
            'max_fiber_width':   self.max_fiber_width,
            'measure_length':    self.measure_length,
            'measure_width':     self.measure_width,
            'measure_straightness': self.measure_straightness,
            'measure_angle':     self.measure_angle,
            'measure_curvature': self.measure_curvature,
            'extract_centerlines': self.extract_centerlines,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific parameter subclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CTFireParams(ExtractionParams):
    """
    Parameters for CT-FIRE curvelet-based fiber extraction.

    CT-FIRE applies a multi-scale curvelet transform to enhance fiber-like
    structures, thresholds the reconstructed image to create a fiber mask,
    extracts the skeleton, and traces individual fibers along the skeleton.

    Attributes
    ----------
    ctfire_threshold : float
        Threshold applied to the curvelet-reconstructed image to create
        the binary fiber mask.  Higher values = stricter detection.
    ctfire_n_levels : int
        Number of curvelet decomposition levels.  More levels capture
        finer-scale fibers.  Typical: 4–6.
    ctfire_n_angles : int
        Number of angular bins in the curvelet decomposition.
        Must be a power of 2 ≥ 4.  Typical: 8–16.
    straightness_threshold : float
        Minimum straightness (0–1) for a traced fiber to be kept.
        0 = keep all; 0.7 = keep only relatively straight fibers.
    use_matlab_backend : bool
        If ``True``, attempt to call the original MATLAB CT-FIRE
        binary via the MATLAB Engine for Python.  Falls back to the
        Python implementation when the engine is unavailable.
    z_spacing : float
        Inter-slice spacing in µm for anisotropic 3-D volumes.  Used by
        the 3-D FIRE tracer to compute physically correct arc lengths along
        the Z axis.  Set equal to ``pixel_size`` for isotropic voxels.
        Only relevant for ``extract_3d``; ignored in 2-D analysis.

    Notes
    -----
    3-D volumetric extraction (``extract_3d``) requires the CT-FIRE C++
    extension (``_ctfire_cpp``).  Check availability at runtime::

        from tme_quant.fiber_analysis.utils.ctfire_utils import ctfire_backend_status
        print(ctfire_backend_status())
    """
    mode: ExtractionMode = ExtractionMode.CTFIRE

    ctfire_threshold:     float = 0.1
    ctfire_n_levels:      int   = 5
    ctfire_n_angles:      int   = 16
    straightness_threshold: float = 0.0
    use_matlab_backend:     bool  = False
    z_spacing:              float = 1.0  # inter-slice spacing in µm (3-D only)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'ctfire_threshold':       self.ctfire_threshold,
            'ctfire_n_levels':        self.ctfire_n_levels,
            'ctfire_n_angles':        self.ctfire_n_angles,
            'straightness_threshold': self.straightness_threshold,
            'use_matlab_backend':     self.use_matlab_backend,
            'z_spacing':              self.z_spacing,
        })
        return d


@dataclass
class RidgeDetectionParams(ExtractionParams):
    """
    Parameters for Ridge Detection (Fiji plugin) fiber extraction.

    Uses Fiji's Ridge Detection plugin to find curvilinear structures.
    Falls back to a Hessian/Frangi-based NumPy implementation when
    Fiji is not available.

    Attributes
    ----------
    ridge_sigma : float
        Gaussian smoothing sigma (pixels) that controls the scale of
        ridges to detect.  Set to approximately half the expected fiber
        width in pixels.
    lower_threshold : float
        Lower hysteresis threshold for ridge linking (0–1, fraction of
        max response).  Ridges between lower and upper thresholds are
        included only if connected to a ridge above the upper threshold.
    upper_threshold : float
        Upper hysteresis threshold (0–1).  Ridges above this are always
        included.
    extend_line : bool
        Whether to extend detected lines to the point of highest
        curvature at their endpoints (Fiji Ridge Detection option).
    correct_position : bool
        Whether to apply sub-pixel position correction (Fiji option).
    """
    mode: ExtractionMode = ExtractionMode.RIDGE_DETECTION

    ridge_sigma:       float = 2.0
    lower_threshold:   float = 0.1
    upper_threshold:   float = 0.5
    extend_line:       bool  = True
    correct_position:  bool  = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'ridge_sigma':      self.ridge_sigma,
            'lower_threshold':  self.lower_threshold,
            'upper_threshold':  self.upper_threshold,
            'extend_line':      self.extend_line,
            'correct_position': self.correct_position,
        })
        return d


@dataclass
class SkeletonParams(ExtractionParams):
    """
    Parameters for skeletonization-based fiber extraction.

    Extracts fibers by thresholding the image, skeletonizing the binary
    mask, and tracing connected components.  Requires no external tools.

    Attributes
    ----------
    skeleton_method : str
        Skeletonization algorithm.  Options: ``'lee'`` (default, 3-D
        compatible), ``'zhang'`` (2-D only, faster).
    threshold_method : str
        How to binarize the image before skeletonization.
        Options: ``'otsu'``, ``'li'``, ``'yen'``, ``'manual'``.
    manual_threshold : float or None
        Absolute intensity threshold when ``threshold_method='manual'``.
    min_branch_length : float
        Skeleton branches shorter than this (µm) are pruned as noise
        before fiber tracing.
    smooth_skeleton : bool
        Whether to apply a light Gaussian smooth before skeletonization
        to reduce spurs from noise.
    """
    mode: ExtractionMode = ExtractionMode.SKELETON

    skeleton_method:   str            = "lee"
    threshold_method:  str            = "otsu"
    manual_threshold:  Optional[float] = None
    min_branch_length: float           = 3.0
    smooth_skeleton:   bool            = True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'skeleton_method':   self.skeleton_method,
            'threshold_method':  self.threshold_method,
            'manual_threshold':  self.manual_threshold,
            'min_branch_length': self.min_branch_length,
            'smooth_skeleton':   self.smooth_skeleton,
        })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Base result class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """
    Base result returned by every fiber extraction method.

    Attributes
    ----------
    fibers : list of FiberProperties
        All extracted fibers that passed the size and quality filters.
    total_fiber_count : int
        ``len(fibers)`` — set by the analyzer after extraction.
    mean_fiber_length : float
        Mean arc length of extracted fibers (µm).
    std_fiber_length : float
        Standard deviation of arc lengths.
    mean_fiber_width : float
        Mean fiber width (µm).
    mean_straightness : float
        Mean straightness index.
    std_straightness : float
        Standard deviation of straightness.
    mode, dimension, pixel_size, processing_time, parameters :
        Provenance metadata set by the analyzer.
    """
    fibers: List[FiberProperties] = field(default_factory=list)

    # Summary statistics (populated by FiberExtractionAnalyzer)
    total_fiber_count: int   = 0
    mean_fiber_length: float = 0.0
    std_fiber_length:  float = 0.0
    mean_fiber_width:  float = 0.0
    mean_straightness: float = 0.0
    std_straightness:  float = 0.0

    # Provenance
    mode:            Optional[ExtractionMode] = None
    dimension:       str            = "2D"
    pixel_size:      float          = 1.0
    processing_time: float          = 0.0
    parameters:      Dict[str, Any] = field(default_factory=dict)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_mode(cls, mode: ExtractionMode, **kwargs: Any) -> "ExtractionResult":
        """Return a correctly-typed result object for *mode*."""
        _map: Dict[ExtractionMode, Type[ExtractionResult]] = {
            ExtractionMode.CTFIRE:          CTFireResult,
            ExtractionMode.RIDGE_DETECTION: RidgeDetectionResult,
            ExtractionMode.SKELETON:        SkeletonResult,
        }
        return _map.get(mode, cls)(mode=mode, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_fiber_count': self.total_fiber_count,
            'mean_fiber_length': self.mean_fiber_length,
            'std_fiber_length':  self.std_fiber_length,
            'mean_fiber_width':  self.mean_fiber_width,
            'mean_straightness': self.mean_straightness,
            'std_straightness':  self.std_straightness,
            'mode':              self.mode.value if self.mode else None,
            'dimension':         self.dimension,
            'pixel_size':        self.pixel_size,
            'processing_time':   self.processing_time,
            'parameters':        self.parameters,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific result subclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CTFireResult(ExtractionResult):
    """
    Result from CT-FIRE curvelet-based fiber extraction.

    Adds intermediate images and diagnostics specific to the CT-FIRE
    algorithm on top of the base fields.

    Attributes
    ----------
    fiber_mask : ndarray or None
        Binary mask of the curvelet-reconstructed, thresholded fiber
        signal.  Shape ``(H, W)``.
    labeled_fibers : ndarray or None
        Integer-labeled image where each fiber occupies pixels with its
        ``fiber_id + 1``.  Shape ``(H, W)``, dtype int32.
    curvelet_energy_map : ndarray or None
        Total curvelet energy per pixel summed across all angles.
        Useful for visualising where fiber signal is strongest.
    n_candidates : int
        Total number of skeleton segments considered before applying
        length and straightness filters.
    """
    mode: Optional[ExtractionMode] = ExtractionMode.CTFIRE

    fiber_mask:          Optional[np.ndarray] = None
    labeled_fibers:      Optional[np.ndarray] = None
    curvelet_energy_map: Optional[np.ndarray] = None
    n_candidates:        int                  = 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({'n_candidates': self.n_candidates})
        return d


@dataclass
class RidgeDetectionResult(ExtractionResult):
    """
    Result from Ridge Detection fiber extraction.

    Attributes
    ----------
    line_width_map : ndarray or None
        Per-pixel estimated line width from the Ridge Detection plugin.
        Shape ``(H, W)``.  ``None`` when using the NumPy fallback.
    junction_points : ndarray or None
        ``(K, 2)`` array of (row, col) pixel coordinates where detected
        ridges intersect or branch.  Useful for network analysis.
    """
    mode: Optional[ExtractionMode] = ExtractionMode.RIDGE_DETECTION

    line_width_map:  Optional[np.ndarray] = None
    junction_points: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'n_junction_points': (
                len(self.junction_points) if self.junction_points is not None else 0
            ),
        })
        return d


@dataclass
class SkeletonResult(ExtractionResult):
    """
    Result from skeletonization-based fiber extraction.

    Attributes
    ----------
    skeleton_mask : ndarray or None
        Binary skeleton image from which fibers were traced.
        Shape ``(H, W)``.
    branch_points : ndarray or None
        ``(B, 2)`` array of skeleton branch/junction point coordinates.
    end_points : ndarray or None
        ``(E, 2)`` array of skeleton endpoint coordinates.
    """
    mode: Optional[ExtractionMode] = ExtractionMode.SKELETON

    skeleton_mask:  Optional[np.ndarray] = None
    branch_points:  Optional[np.ndarray] = None
    end_points:     Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'n_branch_points': (
                len(self.branch_points) if self.branch_points is not None else 0
            ),
            'n_end_points': (
                len(self.end_points) if self.end_points is not None else 0
            ),
        })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enum
    'ExtractionMode',
    # Single-fiber data
    'FiberProperties',
    # Parameter base + subclasses
    'ExtractionParams',
    'CTFireParams',
    'RidgeDetectionParams',
    'SkeletonParams',
    # Result base + subclasses
    'ExtractionResult',
    'CTFireResult',
    'RidgeDetectionResult',
    'SkeletonResult',
]