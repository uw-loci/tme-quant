"""Combined cell analysis result container."""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class CellAnalysisResult:
    """Combined result from segmentation, classification, and quantification."""

    image_id: str = ""
    segmentation_result: Optional[Any] = None   # SegmentationResult
    classification_result: Optional[Any] = None  # ClassificationResult
    quantification_result: Optional[Any] = None  # QuantificationResult

    combined_metrics: dict = field(default_factory=dict)
