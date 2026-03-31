"""
Cell analysis methods.

All segmentation and classification method classes are importable directly
from this package, alongside the MethodRegistry.
"""

from typing import Dict, Type

from ...core.tme_objects.cell_objects import SegmentationMode, ClassificationMode

# ── Segmentation methods ──────────────────────────────────────────────────────
from .base_segmentation import BaseSegmentationMethod
from .stardist_segmentation import StarDistSegmentation
from .cellpose_segmentation import CellposeSegmentation
from .threshold_segmentation import ThresholdingSegmentation
from .watershed_segmentation import WatershedSegmentation

# ── Classification methods ────────────────────────────────────────────────────
from .base_classification import BaseClassificationMethod
from .morphology_classification import MorphologyClassifier
from .marker_classification import MarkerClassifier


# ── Registry ──────────────────────────────────────────────────────────────────

class MethodRegistry:
    """
    Registry for cell analysis methods.

    Manages registration and retrieval of segmentation and classification
    methods by their enum mode key.
    """

    def __init__(self):
        self._segmentation_methods: Dict[SegmentationMode, Type] = {}
        self._classification_methods: Dict[ClassificationMode, Type] = {}

    # Segmentation ─────────────────────────────────────────────────────────────

    def register_segmentation_method(
        self, mode: SegmentationMode, method_class: Type
    ) -> None:
        self._segmentation_methods[mode] = method_class

    def get_segmentation_method(self, mode: SegmentationMode) -> Type:
        if mode not in self._segmentation_methods:
            raise ValueError(
                f"Segmentation method {mode.value!r} not registered. "
                f"Available: {list(self._segmentation_methods.keys())}"
            )
        return self._segmentation_methods[mode]

    def list_segmentation_methods(self) -> list:
        return list(self._segmentation_methods.keys())

    # Classification ───────────────────────────────────────────────────────────

    def register_classification_method(
        self, mode: ClassificationMode, method_class: Type
    ) -> None:
        self._classification_methods[mode] = method_class

    def get_classification_method(self, mode: ClassificationMode) -> Type:
        if mode not in self._classification_methods:
            raise ValueError(
                f"Classification method {mode.value!r} not registered. "
                f"Available: {list(self._classification_methods.keys())}"
            )
        return self._classification_methods[mode]

    def list_classification_methods(self) -> list:
        return list(self._classification_methods.keys())


# Global registry instance
_global_registry = MethodRegistry()


def get_registry() -> MethodRegistry:
    """Return the global method registry."""
    return _global_registry


__all__ = [
    # Registry
    'MethodRegistry',
    'get_registry',
    # Segmentation
    'BaseSegmentationMethod',
    'StarDistSegmentation',
    'CellposeSegmentation',
    'ThresholdingSegmentation',
    'WatershedSegmentation',
    # Classification
    'BaseClassificationMethod',
    'MorphologyClassifier',
    'MarkerClassifier',
]
