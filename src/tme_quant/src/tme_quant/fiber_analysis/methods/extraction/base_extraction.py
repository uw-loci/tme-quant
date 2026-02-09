"""Base class for fiber extraction methods."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List
from ...config.extraction_params import ExtractionParams, ExtractionResult, FiberProperties


class BaseExtractionMethod(ABC):
    """Abstract base class for fiber extraction methods."""
    
    @abstractmethod
    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams
    ) -> ExtractionResult:
        """Extract individual fibers from 2D image."""
        pass
    
    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams
    ) -> ExtractionResult:
        """Extract individual fibers from 3D image."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 3D extraction"
        )
    
    def supports_3d(self) -> bool:
        """Check if method supports 3D extraction."""
        return False
    
    def _filter_by_length(
        self,
        fibers: List[FiberProperties],
        min_length: float
    ) -> List[FiberProperties]:
        """Filter fibers by minimum length."""
        return [f for f in fibers if f.length >= min_length]
    
    def _compute_fiber_properties(
        self,
        centerline: np.ndarray,
        pixel_size: float
    ) -> FiberProperties:
        """Compute properties for a single fiber."""
        # This is a placeholder - actual implementation in specific methods
        pass