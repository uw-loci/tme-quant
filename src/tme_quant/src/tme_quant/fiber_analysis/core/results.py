"""Combined fiber analysis result container."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config.orientation_params import OrientationResult
from ..config.extraction_params import ExtractionResult, FiberProperties


@dataclass
class FiberAnalysisResult:
    """Combined result from orientation + extraction analysis."""
    image_id: str = ""

    orientation_result: Optional[OrientationResult] = None
    extraction_result:  Optional[ExtractionResult]  = None

    measurements:  Dict[str, Any] = field(default_factory=dict)
    export_paths:  Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {'image_id': self.image_id}
        if self.orientation_result:
            d['orientation'] = self.orientation_result.to_dict()
        if self.extraction_result:
            d['extraction'] = self.extraction_result.to_dict()
        d['measurements'] = self.measurements
        return d


# Re-export for importers that do: from ..core.results import FiberProperties
__all__ = ['FiberAnalysisResult', 'FiberProperties', 'OrientationResult', 'ExtractionResult']