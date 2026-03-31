"""
Configuration and parameter dataclasses for cell analysis.
"""

from __future__ import annotations


# ========================================================
# CLASSIFICATION PARAMS
# ========================================================

from ..core.tme_models.cell_model import (
    ClassificationMode,
    CellType,
    ClassificationParams,
)

# ClassificationResult may not exist yet; provide a minimal stub if absent.
try:
    from ..core.tme_models.cell_model import ClassificationResult
except ImportError:
    from dataclasses import dataclass, field
    from typing import List, Dict, Any, Optional

    @dataclass
    class ClassificationResult:
        """Minimal stub for classification results."""
        cell_type_counts: Dict[str, int] = field(default_factory=dict)
        parameters: Optional[Dict[str, Any]] = None

__all__ = [
    "ClassificationMode",
    "CellType",
    "ClassificationParams",
    "ClassificationResult",
]

# ========================================================
# QUANTIFICATION PARAMS
# ========================================================

from ..core.tme_models.cell_model import QuantificationParams

# QuantificationResult may not exist yet; provide a minimal stub if absent.
try:
    from ..core.tme_models.cell_model import QuantificationResult
except ImportError:
    from dataclasses import dataclass, field
    from typing import Dict, Any, Optional

    @dataclass
    class QuantificationResult:
        """Minimal stub for quantification results."""
        measurements: Dict[str, Any] = field(default_factory=dict)
        parameters: Optional[Dict[str, Any]] = None

__all__ = [
    "QuantificationParams",
    "QuantificationResult",
]

# ========================================================
# SEGMENTATION PARAMS
# ========================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Re-export enums from the canonical location
from ..core.tme_models.cell_model import (  # noqa: F401
    SegmentationMode, ImageModality, CellType,
)


@dataclass
class SegmentationParams:
    """Parameters for cell segmentation."""
    mode: SegmentationMode = None
    image_modality: ImageModality = None
    pixel_size: float = 1.0
    target: str = "nucleus"
    stardist_model:   str = "2D_versatile_he"
    cellpose_model:   str = "cyto"
    threshold_method: str = "otsu"
    min_cell_size:    float = 20.0
    max_cell_size:    float = 5000.0
    probability_threshold: float = 0.5
    stardist_prob_thresh:  float = 0.5
    stardist_nms_thresh:   float = 0.4
    cellpose_diameter:     float = 30.0
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    return_probabilities: bool = False
    use_gpu: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SegmentationResult:
    """Results from cell segmentation."""
    from ..core.tme_models.cell_model import CellProperties  # local import avoids circular
    cells: List[Any] = field(default_factory=list)
    label_mask: Optional[np.ndarray] = None
    probability_map: Optional[np.ndarray] = None
    total_cell_count: int = 0
    mean_cell_area: float = 0.0
    mean_circularity: float = 0.0
    pixel_size: float = 1.0
    mode: Any = None
    dimension: str = "2D"
    image_modality: Any = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_cell_count': self.total_cell_count,
            'mean_cell_area':   self.mean_cell_area,
            'mean_circularity': self.mean_circularity,
            'pixel_size':       self.pixel_size,
            'dimension':        self.dimension,
            'mode':             self.mode.value if self.mode else None,
            'processing_time':  self.processing_time,
        }


__all__ = [
    'SegmentationMode', 'ImageModality', 'CellType',
    'SegmentationParams', 'SegmentationResult',
]