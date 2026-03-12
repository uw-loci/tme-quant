"""
Configuration and parameters for cell analysis.
"""

__all__ = []

# Re-export enums from cell_model for convenience
try:
    from ...core.tme_models.cell_model import (
        SegmentationMode,
        ImageModality,
        CellType,
        ClassificationMode,
    )
    __all__ += ['SegmentationMode', 'ImageModality', 'CellType', 'ClassificationMode']
except ImportError:
    pass

# Parameter classes (modules not yet created)
try:
    from .segmentation_params import SegmentationParams
    __all__.append('SegmentationParams')
except ImportError:
    pass

try:
    from .classification_params import ClassificationParams
    __all__.append('ClassificationParams')
except ImportError:
    pass

try:
    from .quantification_params import QuantificationParams
    __all__.append('QuantificationParams')
except ImportError:
    pass