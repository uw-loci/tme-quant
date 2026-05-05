"""
Fiber orientation: base protocol and analyzer coordinator.
"""

"""Base class for fiber orientation analysis methods."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import OrientationParams, OrientationResult


def _report(cb: Optional[Callable], step: int, total: int, msg: str) -> None:
    if cb is not None:
        cb(step, total, msg)


class BaseOrientationMethod(ABC):
    """Abstract base class for all fiber orientation analysis methods."""

    @abstractmethod
    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationResult:
        """Analyse fiber orientation in a 2-D image."""

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationResult:
        """Analyse fiber orientation in a 3-D image."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 3-D analysis"
        )

    def supports_3d(self) -> bool:
        """Return True if this method implements analyze_3d."""
        return False

    # ── Shared statistics helper ──────────────────────────────────────────────

    def _compute_statistics(
        self,
        orientation_map: np.ndarray,
        coherency_map: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Compute circular statistics from an orientation map.

        Parameters
        ----------
        orientation_map : ndarray
            1-D array of valid orientation values in degrees, **or** a
            2-D / 3-D map (NaN pixels are ignored automatically).
        coherency_map : ndarray or None
            Not currently used in the calculation; reserved for
            coherency-weighted statistics in a future release.

        Returns
        -------
        dict with keys:
            ``mean_orientation``       – circular mean (degrees)
            ``alignment_score``        – mean resultant length R ∈ [0, 1]
            ``std_orientation``        – circular std dev (degrees)
            ``orientation_distribution`` – 36-bin histogram over [−90°, 90°]
        """
        flat = orientation_map.ravel()
        valid = flat[~np.isnan(flat)] if np.issubdtype(flat.dtype, np.floating) \
                else flat

        if valid.size == 0:
            return {
                'mean_orientation':         0.0,
                'alignment_score':          0.0,
                'std_orientation':          0.0,
                'orientation_distribution': np.zeros(36, dtype=np.int64),
            }

        angles_rad = np.deg2rad(valid)

        # Circular mean using the doubling trick (handles 180° periodicity)
        mean_x = float(np.mean(np.cos(2.0 * angles_rad)))
        mean_y = float(np.mean(np.sin(2.0 * angles_rad)))
        mean_orientation = float(np.rad2deg(np.arctan2(mean_y, mean_x) / 2.0))

        # Mean resultant length (order parameter R)
        alignment_score = float(np.sqrt(mean_x ** 2 + mean_y ** 2))

        # Circular standard deviation
        # Clamp R to (0, 1] — avoids sqrt of negative for near-zero R
        r_clamped = float(np.clip(alignment_score, 1e-10, 1.0 - 1e-10))
        std_orientation = float(np.rad2deg(np.sqrt(-2.0 * np.log(r_clamped))))

        # Orientation histogram
        hist, _ = np.histogram(valid, bins=36, range=(-90.0, 90.0))

        return {
            'mean_orientation':         mean_orientation,
            'alignment_score':          alignment_score,
            'std_orientation':          std_orientation,
            'orientation_distribution': hist,
        }

# ── FiberOrientationAnalyzer ──

"""Fiber orientation analysis coordinator."""

import time
from typing import Dict, Type

import numpy as np

# base_orientation merged: BaseOrientationMethod
from .config import (
    OrientationMode,
    OrientationParams, CurveAlignParams, OrientationJParams,
    GradientParams, StructureTensorParams,
    OrientationResult,
)


class FiberOrientationAnalyzer:
    """
    Coordinates fiber orientation analysis across all supported modes.

    Supported modes
    ---------------
    - CURVEALIGN      – curvelet-based (2-D / 3-D)
    - ORIENTATIONJ    – Fiji plugin (2-D only)
    - GRADIENT        – pixel-wise Sobel/Scharr (2-D)
    - STRUCTURE_TENSOR – windowed structure tensor (2-D / 3-D)
    """

    def __init__(self) -> None:
        self._orientation_methods: Dict[OrientationMode, Type[BaseOrientationMethod]] = {}
        self._register_methods()

    def _register_methods(self) -> None:
        """Lazily register all available orientation method classes."""
        from .methods.curvealign import CurveAlignOrientation
        from .methods.orientationj import OrientationJMethod

        self._orientation_methods[OrientationMode.CURVEALIGN]   = CurveAlignOrientation
        self._orientation_methods[OrientationMode.ORIENTATIONJ] = OrientationJMethod
        # GRADIENT and STRUCTURE_TENSOR are registered on first use to
        # avoid importing scipy at module load time.

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> OrientationResult:
        """
        Analyse fiber orientation in a 2-D image.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.
        params : OrientationParams subclass
            Use ``OrientationParams.for_mode()`` or instantiate a subclass
            directly (``CurveAlignParams``, ``OrientationJParams``, etc.).
        progress_callback : callable or None
            Optional ``(current, total, message)`` callback for progress reporting.

        Returns
        -------
        OrientationResult subclass matching params.mode
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image.shape}")

        method = self._get_method(params.mode)

        _report(progress_callback, 1, 2, f"Analyzing fiber orientation ({params.mode.value})…")
        start = time.perf_counter()
        result = method.analyze_2d(image, params)
        elapsed = time.perf_counter() - start

        # Write provenance metadata (overwrite anything the method set)
        _report(progress_callback, 2, 2, "Finalizing orientation result…")
        result.dimension        = "2D"
        result.mode             = params.mode
        result.processing_time  = elapsed
        result.parameters       = params.to_dict()   # JSON-safe dict

        return result

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> OrientationResult:
        """
        Analyse fiber orientation in a 3-D image.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
        params : OrientationParams subclass
        progress_callback : callable or None
            Optional ``(current, total, message)`` callback for progress reporting.

        Returns
        -------
        OrientationResult subclass matching params.mode
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3-D image, got shape {image.shape}")

        method = self._get_method(params.mode)

        if not method.supports_3d():
            raise ValueError(
                f"Mode '{params.mode.value}' does not support 3-D analysis"
            )

        _report(progress_callback, 1, 2, f"Analyzing fiber orientation 3D ({params.mode.value})…")
        start = time.perf_counter()
        result = method.analyze_3d(image, params)
        elapsed = time.perf_counter() - start

        _report(progress_callback, 2, 2, "Finalizing orientation result…")
        result.dimension       = "3D"
        result.mode            = params.mode
        result.processing_time = elapsed
        result.parameters      = params.to_dict()

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_method(self, mode: OrientationMode) -> BaseOrientationMethod:
        """
        Return an instantiated orientation method for *mode*.

        Registers GRADIENT and STRUCTURE_TENSOR on first use so that scipy
        is not imported at module load time.
        """
        if mode not in (OrientationMode.CURVEALIGN, OrientationMode.ORIENTATIONJ):
            self._ensure_extra_methods_registered()

        if mode not in self._orientation_methods:
            raise ValueError(
                f"Unsupported orientation mode: {mode!r}. "
                f"Available: {list(self._orientation_methods)}"
            )
        return self._orientation_methods[mode]()

    def _ensure_extra_methods_registered(self) -> None:
        """Register GRADIENT and STRUCTURE_TENSOR if not already done."""
        if OrientationMode.GRADIENT not in self._orientation_methods:
            from .methods.gradient import GradientOrientationMethod
            self._orientation_methods[OrientationMode.GRADIENT] = GradientOrientationMethod
        if OrientationMode.STRUCTURE_TENSOR not in self._orientation_methods:
            from .methods.structure_tensor import StructureTensorMethod
            self._orientation_methods[OrientationMode.STRUCTURE_TENSOR] = StructureTensorMethod