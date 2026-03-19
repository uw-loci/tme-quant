"""Quantification parameters — re-exported from the central cell model."""

from ...core.tme_models.cell_model import QuantificationParams

# QuantificationResult may not exist yet; provide a minimal stub if absent.
try:
    from ...core.tme_models.cell_model import QuantificationResult
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
