"""Combined fiber analysis result container."""

from dataclasses import dataclass, field
from typing import Optional, Any

# Re-export FiberProperties so importers can find it here
from ..config.extraction_params import FiberData as FiberProperties


@dataclass
class FiberAnalysisResult:
    """Combined result from orientation and extraction analysis."""

    image_id: str = ""
    orientation_result: Optional[Any] = None   # OrientationResult
    extraction_result: Optional[Any] = None    # ExtractionResult

    # Combined / derived metrics
    combined_metrics: dict = field(default_factory=dict)
