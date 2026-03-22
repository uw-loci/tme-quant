"""Base class for fiber extraction methods."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ...config.extraction_params import (
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