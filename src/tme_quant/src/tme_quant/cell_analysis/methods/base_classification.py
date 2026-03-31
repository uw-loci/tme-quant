"""
Abstract base class for cell classification methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..config import ClassificationParams, ClassificationResult
from tme_quant.core.tme_objects.cell_objects import SegmentationResult


class BaseClassificationMethod(ABC):
    """Abstract base class for classification methods."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @abstractmethod
    def classify(
        self,
        segmentation_result: SegmentationResult,
        params: ClassificationParams,
        image: Optional[np.ndarray] = None,
    ) -> ClassificationResult:
        """
        Classify segmented cells.

        Args:
            segmentation_result: Segmentation result
            params: Classification parameters
            image: Original image (required for intensity-based methods)

        Returns:
            ClassificationResult with cell type assignments
        """
        pass
