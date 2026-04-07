"""
Fiber extraction: base protocol and analyzer coordinator.
"""

"""Base class for fiber extraction methods."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .config import (
    ExtractionParams, ExtractionResult, FiberProperties,
)


class BaseExtractionMethod(ABC):
    """Abstract base class for all fiber extraction methods."""

    @abstractmethod
    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> ExtractionResult:
        """Extract individual fibers from a 2-D image."""

    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> ExtractionResult:
        """Extract individual fibers from a 3-D image."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 3-D extraction"
        )

    def supports_3d(self) -> bool:
        """Return True if this method implements extract_3d."""
        return False

    # ── Shared helpers available to all subclasses ────────────────────────────

    def _apply_size_filters(
        self,
        fibers: List[FiberProperties],
        params: ExtractionParams,
    ) -> List[FiberProperties]:
        """
        Remove fibers that fall outside the length or width bounds in *params*.

        Parameters
        ----------
        fibers : list of FiberProperties
        params : ExtractionParams (or subclass)

        Returns
        -------
        Filtered list of FiberProperties
        """
        return [
            f for f in fibers
            if (params.min_fiber_length <= f.length  <= params.max_fiber_length
                and params.min_fiber_width  <= f.width   <= params.max_fiber_width)
        ]

    def _build_labeled_image(
        self,
        fibers: List[FiberProperties],
        shape: tuple,
    ) -> np.ndarray:
        """
        Create an integer-labeled image from fiber centerlines.

        Pixel value = ``fiber.fiber_id + 1`` (0 = background).

        Parameters
        ----------
        fibers : list of FiberProperties with non-None centerlines
        shape  : (H, W) output image shape

        Returns
        -------
        ndarray of shape *shape*, dtype int32
        """
        labeled = np.zeros(shape, dtype=np.int32)
        for f in fibers:
            if f.centerline is not None and len(f.centerline) > 0:
                coords = np.clip(
                    f.centerline.astype(int),
                    [0, 0],
                    [shape[0] - 1, shape[1] - 1],
                )
                labeled[coords[:, 0], coords[:, 1]] = f.fiber_id + 1
        return labeled

# ── FiberExtractionAnalyzer ──

"""Fiber extraction analysis coordinator."""

import time
from typing import Dict, Type

import numpy as np

# base_extraction merged: BaseExtractionMethod
from .config import (
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
        self._extraction_methods: Dict[ExtractionMode, Type[BaseExtractionMethod]] = {}
        self._register_methods()

    def _register_methods(self) -> None:
        """Register all available extraction method classes."""
        from .methods.ctfire import CTFireExtraction
        from .methods.ridge_detection import RidgeDetectionMethod
        from .methods.skeleton import SkeletonExtractionMethod

        self._extraction_methods[ExtractionMode.CTFIRE]          = CTFireExtraction
        self._extraction_methods[ExtractionMode.RIDGE_DETECTION] = RidgeDetectionMethod
        self._extraction_methods[ExtractionMode.SKELETON]        = SkeletonExtractionMethod

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
        from ..core.tme_objects.fiber_objects import FiberObject
        from .config import FiberProperties
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
        from ..core.tme_objects.fiber_objects import FiberObject
        from .config import FiberProperties
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
        if mode not in self._extraction_methods:
            raise ValueError(
                f"Unsupported extraction mode: {mode!r}. "
                f"Available: {list(self._extraction_methods)}"
            )
        return self._extraction_methods[mode]()

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