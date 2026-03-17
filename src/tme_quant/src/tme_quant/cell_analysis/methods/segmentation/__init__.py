"""
Cell segmentation methods.

All implementations live in base_segmentation.py.
StarDist and Cellpose wrap their respective third-party libraries
(pip install stardist / pip install cellpose) and are imported directly
from the canonical module rather than maintained as separate files.
"""

from .base_segmentation import (
    BaseSegmentationMethod,
    StarDistSegmentation,
    CellposeSegmentation,
    ThresholdingSegmentation,
    WatershedSegmentation,
)

__all__ = [
    'BaseSegmentationMethod',
    'StarDistSegmentation',
    'CellposeSegmentation',
    'ThresholdingSegmentation',
    'WatershedSegmentation',
]