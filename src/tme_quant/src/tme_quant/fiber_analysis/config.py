"""
Configuration and parameter dataclasses for fiber analysis.
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
# ========================================================
# ORIENTATION PARAMS
# ========================================================


# ─────────────────────────────────────────────────────────────────────────────
# Mode enum
# ─────────────────────────────────────────────────────────────────────────────

class OrientationMode(Enum):
    """Supported fiber orientation analysis algorithms."""
    CURVEALIGN       = "curvealign"        # Curvelet-based (CT-FIRE / CurveAlign)
    ORIENTATIONJ     = "orientationj"      # Fiji OrientationJ plugin
    GRADIENT         = "gradient"          # Pixel-wise gradient (Sobel / Scharr …)
    STRUCTURE_TENSOR = "structure_tensor"  # Windowed structure tensor


# ─────────────────────────────────────────────────────────────────────────────
# Base parameter class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrientationParams:
    """
    Base parameters shared by every orientation analysis mode.

    Instantiate via :meth:`for_mode` to get a correctly-typed subclass,
    or construct a subclass directly.

    Attributes
    ----------
    mode : OrientationMode
        Which algorithm to run.
    pixel_size : float
        Image pixel size in µm/pixel.  Used when reporting physical
        measurements in the result.
    compute_statistics : bool
        Whether to compute summary statistics (circular mean orientation,
        alignment score, orientation histogram) after the maps are built.
    keep_values : list of str
        Which output arrays to retain after analysis.  Options:
        ``'angles'``, ``'alignment'``, ``'energy'``, ``'all'``.
        Unused arrays are set to ``None`` to save memory.
    """
    mode:               OrientationMode = OrientationMode.CURVEALIGN
    pixel_size:         float           = 1.0
    compute_statistics: bool            = True
    keep_values:        List[str]       = field(
        default_factory=lambda: ['angles', 'alignment']
    )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_mode(cls, mode: OrientationMode, **kwargs: Any) -> "OrientationParams":
        """
        Return a correctly-typed parameter object for *mode*.

        Parameters
        ----------
        mode : OrientationMode
            Target analysis mode.
        **kwargs
            Forwarded to the subclass constructor.

        Returns
        -------
        OrientationParams subclass

        Examples
        --------
        >>> p = OrientationParams.for_mode(
        ...     OrientationMode.CURVEALIGN, pixel_size=0.5, curvelet_levels=5
        ... )
        >>> type(p).__name__
        'CurveAlignParams'

        >>> p = OrientationParams.for_mode(
        ...     OrientationMode.STRUCTURE_TENSOR, sigma_spatial=3.0
        ... )
        >>> type(p).__name__
        'StructureTensorParams'
        """
        _map: Dict[OrientationMode, Type[OrientationParams]] = {
            OrientationMode.CURVEALIGN:       CurveAlignParams,
            OrientationMode.ORIENTATIONJ:     OrientationJParams,
            OrientationMode.GRADIENT:         GradientParams,
            OrientationMode.STRUCTURE_TENSOR: StructureTensorParams,
        }
        return _map.get(mode, cls)(mode=mode, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode':               self.mode.value,
            'pixel_size':         self.pixel_size,
            'compute_statistics': self.compute_statistics,
            'keep_values':        list(self.keep_values),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific parameter subclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveAlignParams(OrientationParams):
    """
    Parameters for CurveAlign curvelet-based orientation analysis.

    CurveAlign decomposes the image with a multi-scale curvelet transform
    inside a sliding window, then picks the dominant angular energy bin as
    the local fiber orientation.  It is the reference method for SHG
    collagen images and the default in TMEQuant.

    Attributes
    ----------
    window_size : int
        Side length (pixels) of the sliding analysis window.
        Smaller → finer spatial resolution but noisier estimates.
        Typical range: 32–128 pixels.
    overlap : float
        Fractional overlap between adjacent windows (0–1).
        ``0.5`` = 50 % overlap.  Higher overlap gives smoother maps
        but proportionally longer run time.
    curvelet_levels : int
        Number of curvelet decomposition scales (frequency bands).
        More levels capture finer structures but are slower.
        Typical: 3–6.
    curvelet_angles : int
        Number of angular bins in the curvelet decomposition.
        Must be a power of 2 ≥ 4.  More angles → finer angular
        resolution.  Typical: 8–16.
    compute_coherency : bool
        Whether to fill a per-pixel coherency (energy concentration)
        map in the result.
    compute_energy : bool
        Whether to fill a per-pixel total curvelet energy map.
    use_matlab_backend : bool
        If ``True``, attempt to call the original MATLAB CT-FIRE /
        CurveAlign via the MATLAB Engine for Python.  Falls back to
        the NumPy FFT approximation when the engine is unavailable.
    return_fiber_segments : bool
        If ``True``, trace and return individual fiber segment
        coordinates from the curvelet maxima.  Only meaningful for
        CurveAlign; ignored by other modes.
    """
    mode: OrientationMode = OrientationMode.CURVEALIGN

    # Sliding-window settings
    window_size: int   = 64
    overlap:     float = 0.5

    # Curvelet decomposition settings
    curvelet_levels: int = 4
    curvelet_angles: int = 8

    # Output flags
    compute_coherency: bool = True
    compute_energy:    bool = True

    # Backend and optional outputs
    use_matlab_backend:    bool = False
    return_fiber_segments: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'window_size':         self.window_size,
            'overlap':             self.overlap,
            'curvelet_levels':     self.curvelet_levels,
            'curvelet_angles':     self.curvelet_angles,
            'compute_coherency':   self.compute_coherency,
            'compute_energy':      self.compute_energy,
            'use_matlab_backend':  self.use_matlab_backend,
            'return_fiber_segments': self.return_fiber_segments,
        })
        return d


@dataclass
class OrientationJParams(OrientationParams):
    """
    Parameters for OrientationJ (Fiji plugin) orientation analysis.

    OrientationJ computes pixel-wise orientations using a local structure
    tensor or gradient method, called via Fiji (pyimagej or subprocess).
    Requires Fiji to be installed and ``FIJI_PATH`` set, or pyimagej.
    Falls back to a NumPy structure-tensor implementation when Fiji is
    unavailable.

    Attributes
    ----------
    gradient_method : str
        Gradient estimator passed to OrientationJ.  Options:
        ``'Gaussian'``, ``'Finite difference'``, ``'Fourier'``,
        ``'Riesz'``, ``'Cubic spline'``.
    coherency_threshold : float
        Minimum coherency (0–1) for a pixel to be considered reliably
        oriented.  Pixels below this value are set to NaN in the map.
    energy_threshold : float
        Minimum structure tensor energy (0–1) for pixel inclusion.
    sigma_tensor : float
        Gaussian smoothing sigma (pixels) applied to the structure
        tensor components before eigenvector computation.  Larger
        values give spatially smoother but lower-resolution maps.
    compute_color_survey : bool
        Whether to request OrientationJ's HSB-encoded colour survey
        image (useful for visual QC).  Adds run time.
    """
    mode: OrientationMode = OrientationMode.ORIENTATIONJ

    gradient_method:      str   = "Gaussian"
    coherency_threshold:  float = 0.1
    energy_threshold:     float = 0.0
    sigma_tensor:         float = 2.0
    compute_color_survey: bool  = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'gradient_method':      self.gradient_method,
            'coherency_threshold':  self.coherency_threshold,
            'energy_threshold':     self.energy_threshold,
            'sigma_tensor':         self.sigma_tensor,
            'compute_color_survey': self.compute_color_survey,
        })
        return d


@dataclass
class GradientParams(OrientationParams):
    """
    Parameters for gradient-based pixel-wise orientation analysis.

    The fastest method.  Derives fiber orientation directly from the
    image gradient direction: ``θ = arctan2(gy, gx) + 90°``.
    No external tools required.

    Suitable for quick previews or images with strong, well-separated
    fibers.  Less robust than the structure tensor or curvelet methods
    on noisy or dense fiber networks.

    Attributes
    ----------
    gradient_operator : str
        Derivative filter to use.  Options: ``'sobel'``, ``'scharr'``,
        ``'prewitt'``, ``'farid'``.
    smoothing_sigma : float
        Gaussian pre-smoothing sigma (pixels) before gradient
        computation.  ``0`` = no pre-smoothing.
    min_gradient_magnitude : float
        Pixels whose gradient magnitude is below this threshold are
        classified as background and excluded from statistics.
    """
    mode: OrientationMode = OrientationMode.GRADIENT

    gradient_operator:      str   = "sobel"
    smoothing_sigma:        float = 1.0
    min_gradient_magnitude: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'gradient_operator':      self.gradient_operator,
            'smoothing_sigma':        self.smoothing_sigma,
            'min_gradient_magnitude': self.min_gradient_magnitude,
        })
        return d


@dataclass
class StructureTensorParams(OrientationParams):
    """
    Parameters for windowed structure tensor orientation analysis.

    The structure tensor (second-moment matrix) averages the outer
    product of image gradients over a local neighbourhood.  The
    dominant eigenvector gives orientation; the eigenvalue ratio gives
    a well-defined coherency (anisotropy index).  Implemented entirely
    in NumPy/SciPy — no external tools required.

    Attributes
    ----------
    sigma_derivative : float
        Inner-scale Gaussian sigma (pixels) used for the derivative.
        Controls sensitivity to fine vs. coarse structures.
    sigma_spatial : float
        Outer-scale (integration) Gaussian sigma (pixels) for smoothing
        the tensor components.  Should be larger than
        ``sigma_derivative``.  Larger → spatially smoother but coarser
        orientation maps.
    compute_anisotropy : bool
        Whether to compute the per-pixel anisotropy index
        ``(λ_max − λ_min) / (λ_max + λ_min)`` ∈ [0, 1].
    compute_eigenvalues : bool
        Whether to return the full per-pixel eigenvalue maps
        (``lambda_max``, ``lambda_min``) in the result.
    """
    mode: OrientationMode = OrientationMode.STRUCTURE_TENSOR

    sigma_derivative:    float = 1.0
    sigma_spatial:       float = 3.0
    compute_anisotropy:  bool  = True
    compute_eigenvalues: bool  = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'sigma_derivative':    self.sigma_derivative,
            'sigma_spatial':       self.sigma_spatial,
            'compute_anisotropy':  self.compute_anisotropy,
            'compute_eigenvalues': self.compute_eigenvalues,
        })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Base result class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrientationResult:
    """
    Base result returned by every orientation analysis method.

    All modes populate the fields defined here.  Mode-specific
    subclasses add extra fields that are only meaningful for their
    algorithm (e.g. raw curvelet coefficients for CurveAlign).

    Attributes
    ----------
    orientation_map : ndarray or None
        Per-pixel dominant fiber orientation in degrees (−90 to +90).
        Unreliable pixels (below coherency / magnitude threshold) are
        set to ``NaN``.
    alignment_map : ndarray or None
        Per-pixel alignment strength in [0, 1].  Interpretation varies
        by mode:
        - CurveAlign: angular energy concentration per window
        - OrientationJ / StructureTensor: coherency = (λ_max−λ_min) /
          (λ_max+λ_min)
        - Gradient: normalised gradient magnitude
    mean_orientation : float
        Circular mean of all valid orientation values (degrees).
    alignment_score : float
        Global mean resultant length R ∈ [0, 1]: 0 = random, 1 =
        perfectly aligned.
    std_orientation : float
        Circular standard deviation of orientations (degrees).
    orientation_distribution : ndarray or None
        36-bin histogram of orientations over [−90°, 90°].
    mode, dimension, pixel_size, processing_time, parameters :
        Provenance metadata written by the analyzer after the method
        returns.
    """
    # Core spatial outputs
    orientation_map: Optional[np.ndarray] = None
    alignment_map:   Optional[np.ndarray] = None

    # Scalar summary statistics
    mean_orientation:         float = 0.0
    alignment_score:          float = 0.0
    mean_alignment:           float = 0.0  # kept as alias of alignment_score
    std_orientation:          float = 0.0
    orientation_distribution: Optional[np.ndarray] = None

    # Provenance (written by FiberOrientationAnalyzer, not the method)
    mode:            Optional[OrientationMode] = None
    dimension:       str            = "2D"
    pixel_size:      float          = 1.0
    processing_time: float          = 0.0
    parameters:      Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the two alignment aliases in sync if only one is set
        if self.mean_alignment == 0.0 and self.alignment_score != 0.0:
            self.mean_alignment = self.alignment_score
        elif self.alignment_score == 0.0 and self.mean_alignment != 0.0:
            self.alignment_score = self.mean_alignment

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_mode(cls, mode: OrientationMode, **kwargs: Any) -> "OrientationResult":
        """Return a correctly-typed result object for *mode*."""
        _map: Dict[OrientationMode, Type[OrientationResult]] = {
            OrientationMode.CURVEALIGN:       CurveAlignResult,
            OrientationMode.ORIENTATIONJ:     OrientationJResult,
            OrientationMode.GRADIENT:         GradientResult,
            OrientationMode.STRUCTURE_TENSOR: StructureTensorResult,
        }
        return _map.get(mode, cls)(mode=mode, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_orientation':         self.mean_orientation,
            'alignment_score':          self.alignment_score,
            'std_orientation':          self.std_orientation,
            'orientation_distribution': (
                self.orientation_distribution.tolist()
                if self.orientation_distribution is not None else None
            ),
            'mode':             self.mode.value if self.mode else None,
            'dimension':        self.dimension,
            'pixel_size':       self.pixel_size,
            'processing_time':  self.processing_time,
            'parameters':       self.parameters,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific result subclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveAlignResult(OrientationResult):
    """
    Result from CurveAlign curvelet-based orientation analysis.

    Adds multi-scale curvelet-specific outputs on top of the base fields.

    Attributes
    ----------
    alignment_map : ndarray or None
        Per-pixel angular energy concentration ∈ [0, 1].  For each sliding
        window, this equals ``peak_bin_energy / total_energy`` — how tightly
        the curvelet energy is concentrated in the dominant angular bin.  All
        pixels covered by the same window share the same value.

        .. note::
            This is a **per-window pixel map**, not the same as the
            *per-region* alignment score (mean resultant length R of fiber
            angles within a TACS zone, ROI, or k-nearest-neighbor group).
            That per-region metric is computed by
            ``CurveAlignOrientation.compute_region_alignment``.

    energy_map : ndarray or None
        Per-pixel total curvelet energy (summed across all angles and
        scales).  Higher energy indicates stronger fiber signal.
    scale_energy_maps : list of ndarray or None
        One energy map per curvelet decomposition level.  Useful for
        multi-scale analysis of fiber length distributions.
    curvelet_coefficients : ndarray or None
        Raw ``(H, W, n_angles)`` coefficient array from the transform.
        Retained only when ``'all'`` is in ``keep_values``; otherwise
        ``None`` to save memory.
    fiber_segments : list of ndarray or None
        List of ``(N, 2)`` arrays giving (row, col) coordinates of
        individual fiber segments traced from curvelet maxima.
        Populated only when ``CurveAlignParams.return_fiber_segments``
        is ``True``.
    window_orientations : dict or None
        Maps ``(row, col)`` window top-left coordinates to a dict with
        keys ``'orientation'`` and ``'coherency'`` for that window.
    n_windows_analyzed : int
        Total number of windows processed.
    mean_energy : float
        Mean curvelet energy across all analysed windows.
    """
    mode: Optional[OrientationMode] = OrientationMode.CURVEALIGN

    energy_map:            Optional[np.ndarray]       = None
    scale_energy_maps:     Optional[List[np.ndarray]] = None
    curvelet_coefficients: Optional[np.ndarray]       = None
    fiber_segments:        Optional[List[np.ndarray]] = None
    window_orientations:   Optional[Dict[Any, Any]]   = None

    n_windows_analyzed: int   = 0
    mean_energy:        float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'n_windows_analyzed': self.n_windows_analyzed,
            'mean_energy':        self.mean_energy,
            'n_fiber_segments':   (
                len(self.fiber_segments) if self.fiber_segments else 0
            ),
        })
        return d


@dataclass
class OrientationJResult(OrientationResult):
    """
    Result from OrientationJ (Fiji plugin) orientation analysis.

    OrientationJ natively outputs orientation, coherency and energy maps
    as separate images.  These are exposed as dedicated fields here in
    addition to the base alignment_map.

    Attributes
    ----------
    coherency_map : ndarray or None
        Per-pixel coherency C = (λ_max − λ_min) / (λ_max + λ_min) ∈ [0, 1].
        Identical to ``alignment_map``; exposed under both names for
        compatibility with OrientationJ nomenclature.
    energy_map : ndarray or None
        Per-pixel structure tensor trace (λ_max + λ_min), normalised to
        [0, 1].  Higher values indicate stronger local contrast / fiber
        intensity.
    color_survey : ndarray or None
        HSB-encoded RGB orientation image produced by OrientationJ's
        colour survey.  Shape ``(H, W, 3)``.  ``None`` unless
        ``OrientationJParams.compute_color_survey`` is ``True``.
    mean_coherency : float
        Mean coherency over pixels that exceed the coherency threshold.
    mean_energy : float
        Mean energy over pixels that exceed the energy threshold.
    """
    mode: Optional[OrientationMode] = OrientationMode.ORIENTATIONJ

    coherency_map: Optional[np.ndarray] = None   # same data as alignment_map
    energy_map:    Optional[np.ndarray] = None
    color_survey:  Optional[np.ndarray] = None   # RGB (H, W, 3)

    mean_coherency: float = 0.0
    mean_energy:    float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep coherency_map and alignment_map in sync
        if self.coherency_map is not None and self.alignment_map is None:
            self.alignment_map = self.coherency_map
        elif self.alignment_map is not None and self.coherency_map is None:
            self.coherency_map = self.alignment_map

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'mean_coherency': self.mean_coherency,
            'mean_energy':    self.mean_energy,
        })
        return d


@dataclass
class GradientResult(OrientationResult):
    """
    Result from gradient-based orientation analysis.

    The simplest result type.  No coherency information is available
    because the gradient method is pixel-wise; reliability is estimated
    from the gradient magnitude instead.

    Attributes
    ----------
    gradient_magnitude_map : ndarray or None
        Per-pixel gradient magnitude, normalised to [0, 1].  Acts as a
        confidence proxy: high magnitude → strong edge → reliable
        orientation.
    background_mask : ndarray or None
        Boolean mask; ``True`` where gradient magnitude is below
        ``GradientParams.min_gradient_magnitude`` and the orientation
        estimate is therefore unreliable.
    mean_gradient_magnitude : float
        Mean magnitude over foreground (non-masked) pixels.
    foreground_fraction : float
        Fraction of pixels classified as foreground.
    """
    mode: Optional[OrientationMode] = OrientationMode.GRADIENT

    gradient_magnitude_map:  Optional[np.ndarray] = None
    background_mask:         Optional[np.ndarray] = None

    mean_gradient_magnitude: float = 0.0
    foreground_fraction:     float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # alignment_map doubles as normalised gradient magnitude
        if self.gradient_magnitude_map is not None and self.alignment_map is None:
            self.alignment_map = self.gradient_magnitude_map

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'mean_gradient_magnitude': self.mean_gradient_magnitude,
            'foreground_fraction':     self.foreground_fraction,
        })
        return d


@dataclass
class StructureTensorResult(OrientationResult):
    """
    Result from windowed structure tensor orientation analysis.

    The structure tensor provides both a well-defined orientation and a
    coherency (anisotropy index) without requiring Fiji.

    Attributes
    ----------
    coherency_map : ndarray or None
        Per-pixel anisotropy index
        ``(λ_max − λ_min) / (λ_max + λ_min)`` ∈ [0, 1].
        Identical to ``alignment_map``.
    lambda_max_map : ndarray or None
        Per-pixel dominant eigenvalue.  Populated only when
        ``StructureTensorParams.compute_eigenvalues`` is ``True``.
    lambda_min_map : ndarray or None
        Per-pixel minor eigenvalue.  Populated only when
        ``StructureTensorParams.compute_eigenvalues`` is ``True``.
    mean_anisotropy : float
        Mean coherency / anisotropy index across all pixels.
    isotropy_fraction : float
        Fraction of pixels with coherency < 0.1 (approximately
        isotropic background).
    """
    mode: Optional[OrientationMode] = OrientationMode.STRUCTURE_TENSOR

    coherency_map:  Optional[np.ndarray] = None   # same data as alignment_map
    lambda_max_map: Optional[np.ndarray] = None
    lambda_min_map: Optional[np.ndarray] = None

    mean_anisotropy:   float = 0.0
    isotropy_fraction: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coherency_map is not None and self.alignment_map is None:
            self.alignment_map = self.coherency_map
        elif self.alignment_map is not None and self.coherency_map is None:
            self.coherency_map = self.alignment_map

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'mean_anisotropy':   self.mean_anisotropy,
            'isotropy_fraction': self.isotropy_fraction,
        })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enum
    'OrientationMode',
    # Parameter base + subclasses
    'OrientationParams',
    'CurveAlignParams',
    'OrientationJParams',
    'GradientParams',
    'StructureTensorParams',
    # Result base + subclasses
    'OrientationResult',
    'CurveAlignResult',
    'OrientationJResult',
    'GradientResult',
    'StructureTensorResult',
]