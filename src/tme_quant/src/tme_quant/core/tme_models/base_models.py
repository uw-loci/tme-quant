"""Compatibility base models for tme_models package."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ObjectType(Enum):
    CELL = "cell"
    FIBER = "fiber"
    REGION = "region"
    ORIENTATION_MAP = "orientation_map"
    FIBER_POPULATION = "fiber_population"
    UNKNOWN = "unknown"


@dataclass
class TMEObject:
    """Compatibility object base with object_id-oriented fields."""

    object_id: str = ""
    object_type: ObjectType = ObjectType.UNKNOWN
    parent_id: Optional[str] = None
    roi: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
