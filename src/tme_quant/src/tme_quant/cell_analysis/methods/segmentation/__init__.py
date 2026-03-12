## Cell segmentation methods
"""
Cell segmentation methods.
"""

from .base_segmentation import BaseSegmentationMethod
from .stardist import StarDistSegmentation
from .cellpose import CellposeSegmentation

__all__ = [
    'BaseSegmentationMethod',
    'StarDistSegmentation',
    'CellposeSegmentation',
]

# Optional: additional segmentation methods (not yet created)
try:
    from .thresholding import ThresholdingSegmentation
    __all__.append('ThresholdingSegmentation')
except ImportError:
    pass

try:
    from .watershed import WatershedSegmentation
    __all__.append('WatershedSegmentation')
except ImportError:
    pass