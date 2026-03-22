"""Fiber extraction analysis coordinator."""

import time
from typing import Dict, Type

import numpy as np

from ..methods.extraction.base_extraction import BaseExtractionMethod
from ..methods.registry import MethodRegistry
from ..config.extraction_params import (
    ExtractionMode,
    ExtractionParams, CTFireParams, RidgeDetectionParams, SkeletonParams,
    ExtractionResult, FiberProperties,
)


class FiberExtractionAnalyzer:
    """
    Coordinates individual fiber extraction across all supported modes.

    Supported modes
    ---------------
    - CTFIRE          — curvelet-based (2-D / 3-D)
    - RIDGE_DETECTION — Fiji Ridge Detection plugin (2-D only)
    - SKELETON        — skeletonization-based (2-D / 3-D)
    """

    def __init__(self) -> None:
        self.registry = MethodRegistry()
        self._register_methods()

    def _register_methods(self) -> None:
        """Register all available extraction method classes."""
        from ..methods.extraction.ctfire import CTFireExtraction
        from ..methods.extraction.ridge_detection import RidgeDetectionMethod
        from ..methods.extraction.skeleton import SkeletonExtractionMethod

        self.registry.register_extraction_method(
            ExtractionMode.CTFIRE,          CTFireExtraction
        )
        self.registry.register_extraction_method(
            ExtractionMode.RIDGE_DETECTION, RidgeDetectionMethod
        )
        self.registry.register_extraction_method(
            ExtractionMode.SKELETON,        SkeletonExtractionMethod
        )

    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> ExtractionResult:
        """
        Extract individual fibers from a 2-D image.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.
        params : ExtractionParams subclass
            Use ``ExtractionParams.for_mode()`` or a subclass directly
            (``CTFireParams``, ``RidgeDetectionParams``, ``SkeletonParams``).

        Returns
        -------
        ExtractionResult subclass matching params.mode
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {image.shape}")

        method = self._get_method(params.mode)

        start = time.perf_counter()
        result = method.extract_2d(image, params)
        elapsed = time.perf_counter() - start

        result.dimension       = "2D"
        result.mode            = params.mode
        result.processing_time = elapsed
        result.parameters      = params.to_dict()   # JSON-safe dict

        self._compute_summary_statistics(result)

        # Convert FiberProperties → FiberObject so downstream TME analysis
        # (interaction_detector, annotate_interaction_pairs, etc.) receives
        # the expected TMEObject subclass with .object_id and hierarchy methods.
        from ...core.tme_models.fiber_model import FiberObject
        from ...fiber_analysis.config.extraction_params import FiberProperties
        image_id = params.to_dict().get('mode', 'fiber')
        result.fibers = [
            FiberObject.from_fiber_properties(
                fp,
                object_id=f"{image_id}_fiber_{fp.fiber_id}",
            ) if isinstance(fp, FiberProperties) and not isinstance(fp, FiberObject)
            else fp
            for fp in result.fibers
        ]

        return result

    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> ExtractionResult:
        """
        Extract individual fibers from a 3-D image.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
        params : ExtractionParams subclass

        Returns
        -------
        ExtractionResult subclass matching params.mode
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3-D image, got shape {image.shape}")

        method = self._get_method(params.mode)

        if not method.supports_3d():
            raise ValueError(
                f"Mode '{params.mode.value}' does not support 3-D extraction"
            )

        start = time.perf_counter()
        result = method.extract_3d(image, params)
        elapsed = time.perf_counter() - start

        result.dimension       = "3D"
        result.mode            = params.mode
        result.processing_time = elapsed
        result.parameters      = params.to_dict()

        self._compute_summary_statistics(result)

        # Convert FiberProperties → FiberObject so downstream TME analysis
        # (interaction_detector, annotate_interaction_pairs, etc.) receives
        # the expected TMEObject subclass with .object_id and hierarchy methods.
        from ...core.tme_models.fiber_model import FiberObject
        from ...fiber_analysis.config.extraction_params import FiberProperties
        image_id = params.to_dict().get('mode', 'fiber')
        result.fibers = [
            FiberObject.from_fiber_properties(
                fp,
                object_id=f"{image_id}_fiber_{fp.fiber_id}",
            ) if isinstance(fp, FiberProperties) and not isinstance(fp, FiberObject)
            else fp
            for fp in result.fibers
        ]

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_method(self, mode: ExtractionMode) -> BaseExtractionMethod:
        return self.registry.get_extraction_method(mode)()

    def _compute_summary_statistics(self, result: ExtractionResult) -> None:
        """
        Populate summary statistics on *result* from ``result.fibers``.

        Computes mean and std for length, width, and straightness.
        All fields are set to 0.0 / 0 when no fibers are present.
        """
        fibers = result.fibers
        result.total_fiber_count = len(fibers)

        if not fibers:
            result.mean_fiber_length = 0.0
            result.std_fiber_length  = 0.0
            result.mean_fiber_width  = 0.0
            result.mean_straightness = 0.0
            result.std_straightness  = 0.0
            return

        lengths      = np.array([f.length      for f in fibers], dtype=np.float64)
        widths       = np.array([f.width       for f in fibers], dtype=np.float64)
        straightness = np.array([f.straightness for f in fibers], dtype=np.float64)

        result.mean_fiber_length = float(lengths.mean())
        result.std_fiber_length  = float(lengths.std())
        result.mean_fiber_width  = float(widths.mean())
        result.mean_straightness = float(straightness.mean())
        result.std_straightness  = float(straightness.std())