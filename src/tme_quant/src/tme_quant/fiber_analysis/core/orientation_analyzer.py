"""Fiber orientation analysis coordinator."""

import time
from typing import Dict, Type

import numpy as np

from ..methods.orientation.base_orientation import BaseOrientationMethod
from ..methods.registry import MethodRegistry
from ..config.orientation_params import (
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
        self.registry = MethodRegistry()
        self._register_methods()

    def _register_methods(self) -> None:
        """Lazily register all available orientation method classes."""
        from ..methods.orientation.curvealign import CurveAlignOrientation
        from ..methods.orientation.orientationj import OrientationJMethod

        self.registry.register_orientation_method(
            OrientationMode.CURVEALIGN,   CurveAlignOrientation
        )
        self.registry.register_orientation_method(
            OrientationMode.ORIENTATIONJ, OrientationJMethod
        )
        # GRADIENT and STRUCTURE_TENSOR are registered on first use to
        # avoid importing scipy at module load time.

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
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

        Returns
        -------
        OrientationResult subclass matching params.mode
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image.shape}")

        method = self._get_method(params.mode)

        start = time.perf_counter()
        result = method.analyze_2d(image, params)
        elapsed = time.perf_counter() - start

        # Write provenance metadata (overwrite anything the method set)
        result.dimension        = "2D"
        result.mode             = params.mode
        result.processing_time  = elapsed
        result.parameters       = params.to_dict()   # JSON-safe dict

        return result

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationResult:
        """
        Analyse fiber orientation in a 3-D image.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
        params : OrientationParams subclass

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

        start = time.perf_counter()
        result = method.analyze_3d(image, params)
        elapsed = time.perf_counter() - start

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

        method_class = self.registry.get_orientation_method(mode)
        return method_class()

    def _ensure_extra_methods_registered(self) -> None:
        """Register GRADIENT and STRUCTURE_TENSOR if not already done."""
        if OrientationMode.GRADIENT not in self.registry._orientation_methods:
            from ..methods.orientation.gradient import GradientOrientationMethod
            self.registry.register_orientation_method(
                OrientationMode.GRADIENT, GradientOrientationMethod
            )
        if OrientationMode.STRUCTURE_TENSOR not in self.registry._orientation_methods:
            from ..methods.orientation.structure_tensor import StructureTensorMethod
            self.registry.register_orientation_method(
                OrientationMode.STRUCTURE_TENSOR, StructureTensorMethod
            )