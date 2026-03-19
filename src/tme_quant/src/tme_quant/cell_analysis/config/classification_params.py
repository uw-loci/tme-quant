"""Classification parameters — re-exported from the central cell model."""

from ...core.tme_models.cell_model import (
    ClassificationMode,
    CellType,
    ClassificationParams,
)

# ClassificationResult may not exist yet; provide a minimal stub if absent.
try:
    from ...core.tme_models.cell_model import ClassificationResult
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
