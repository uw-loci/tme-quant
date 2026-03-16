"""
Base models for TME objects with QuPath-like hierarchy support.

Design principles:
- Single TMEObject base class for ALL hierarchy nodes (project, image, tumor,
  fiber, cell, stroma, region, orientation map, etc.).
- Regular class (not dataclass) to avoid MRO / default-field pitfalls with
  deep inheritance chains.
- object_id is the primary key (string, caller-supplied or auto-generated UUID).
- Full parent/child tree traversal on every object — mirrors QuPath's model
  where every object is a first-class hierarchy node.
- TMEType and ObjectType are unified in one file; ObjectType is kept as a
  separate enum for backward compatibility with existing analysis code.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, Iterator, List, Optional, Type, TypeVar, Union
)

import numpy as np

T = TypeVar("T", bound="TMEObject")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TMEType(Enum):
    """Coarse-grained object types following the QuPath hierarchy."""
    PROJECT = "project"
    SAMPLE = "sample"
    IMAGE = "image"
    ANNOTATION = "annotation"
    DETECTION = "detection"
    CELL = "cell"
    FIBER = "fiber"
    VESSEL = "vessel"
    STROMA = "stroma"
    REGION = "region"
    TISSUE = "tissue"
    HIERARCHY = "hierarchy"
    ORIENTATION_MAP = "orientation_map"
    FIBER_POPULATION = "fiber_population"
    TUMOR = "tumor"
    TUMOR_REGION = "tumor_region"
    UNKNOWN = "unknown"


class ObjectType(Enum):
    """
    Fine-grained object type used by analysis modules.

    Kept as a separate enum for backward compatibility with
    interaction_detector, measurement_engine, and pipeline code.
    """
    CELL = "cell"
    FIBER = "fiber"
    REGION = "region"
    ORIENTATION_MAP = "orientation_map"
    FIBER_POPULATION = "fiber_population"
    TUMOR_REGION = "tumor_region"
    COLLAGEN_FIBER = "fiber"      # alias used in pipeline code
    UNKNOWN = "unknown"


class GeometryType(Enum):
    """Supported geometry types (2-D and 3-D)."""
    POINT = "point"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    LINE = "line"
    MULTIPOINT = "multipoint"
    PATH = "path"
    CUBOID = "cuboid"   # 3-D
    MESH = "mesh"       # 3-D


# ---------------------------------------------------------------------------
# Core hierarchy node
# ---------------------------------------------------------------------------

class TMEObject:
    """
    Base class for every node in the TME object hierarchy.

    Every concrete TME object (image, tumor region, fiber, cell, stroma, ...)
    inherits from this class and therefore participates in the same tree.
    This mirrors the QuPath model:
        project -> image -> annotation/tumor -> detection/fiber/cell
    with arbitrary nesting depth.

    Primary key
    -----------
    ``object_id`` — caller-supplied string or auto-generated UUID4.
    Analysis modules that build IDs programmatically (e.g.
    "tumor_1_fiber_42") should pass the id explicitly.

    Backward-compatibility shims
    ----------------------------
    ``.id``   property -> object_id   (old core/base_models used ``id``)
    ``.type`` property -> tme_type    (old code set ``self.type = TMEType.X``)
    """

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        tme_type: TMEType = TMEType.UNKNOWN,
        object_type: ObjectType = ObjectType.UNKNOWN,
        roi: Optional[Any] = None,
        parent: Optional["TMEObject"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Identity
        self.object_id: str = object_id if object_id else str(uuid.uuid4())
        self.name: str = name

        # Type bookkeeping — both enums co-exist for compatibility
        self.tme_type: TMEType = tme_type
        self.object_type: ObjectType = object_type

        # Spatial reference (geometry.ROI or shapely geometry)
        self.roi: Optional[Any] = roi

        # Hierarchy
        self.parent: Optional["TMEObject"] = None
        self.children: List["TMEObject"] = []

        # Data bags
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.properties: Dict[str, Any] = (
            properties if properties is not None else {}
        )

        # Timestamps
        self.created: datetime = datetime.now()
        self.modified: datetime = datetime.now()

        # Wire into parent immediately if supplied
        if parent is not None:
            parent.add_child(self)

    # ------------------------------------------------------------------
    # Backward-compatibility shims
    # ------------------------------------------------------------------

    @property
    def type(self) -> TMEType:
        """Alias for tme_type (legacy attribute name)."""
        return self.tme_type

    @type.setter
    def type(self, value: TMEType) -> None:
        self.tme_type = value

    @property
    def id(self) -> str:
        """Alias for object_id (legacy attribute name)."""
        return self.object_id

    @id.setter
    def id(self, value: str) -> None:
        self.object_id = value

    # ------------------------------------------------------------------
    # Hierarchy management
    # ------------------------------------------------------------------

    def add_child(self, child: "TMEObject") -> None:
        """Add *child* to this node; keeps both sides of the link consistent."""
        if child is self:
            raise ValueError("An object cannot be its own child.")
        if child not in self.children:
            # Detach from previous parent first
            if child.parent is not None and child.parent is not self:
                child.parent.children.remove(child)
            self.children.append(child)
            child.parent = self
            self.modified = datetime.now()

    def remove_child(self, child: "TMEObject") -> None:
        """Remove *child* and clear its parent reference."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self.modified = datetime.now()

    def detach(self) -> None:
        """Detach this object from its parent (no-op if already root)."""
        if self.parent is not None:
            self.parent.remove_child(self)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_ancestors(self) -> List["TMEObject"]:
        """Return ancestors ordered from immediate parent up to root."""
        ancestors: List[TMEObject] = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_root(self) -> "TMEObject":
        """Return the root node of this object's hierarchy."""
        current: TMEObject = self
        while current.parent is not None:
            current = current.parent
        return current

    def get_descendants(self, include_self: bool = False) -> List["TMEObject"]:
        """
        Return all descendants in depth-first order.

        Parameters
        ----------
        include_self:
            When True the list begins with *self*.
        """
        result: List[TMEObject] = []
        if include_self:
            result.append(self)
        for child in self.children:
            result.append(child)
            result.extend(child.get_descendants())
        return result

    def iter_descendants(self) -> Iterator["TMEObject"]:
        """Depth-first iterator over all descendants (self not included)."""
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def depth(self) -> int:
        """Depth of this node in the hierarchy (root = 0)."""
        return len(self.get_ancestors())

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def find_by_id(self, object_id: str) -> Optional["TMEObject"]:
        """Find a descendant (or self) by object_id. Returns None if absent."""
        if self.object_id == object_id:
            return self
        for child in self.children:
            result = child.find_by_id(object_id)
            if result is not None:
                return result
        return None

    def filter_by_type(
        self,
        tme_type: Optional[TMEType] = None,
        object_type: Optional[ObjectType] = None,
        include_self: bool = False,
    ) -> List["TMEObject"]:
        """
        Return descendants matching the given type(s).

        Pass *tme_type*, *object_type*, or both.  Passing neither returns
        all descendants (equivalent to get_descendants).

        Examples
        --------
        >>> fibers = image.filter_by_type(tme_type=TMEType.FIBER)
        >>> cells  = tumor.filter_by_type(object_type=ObjectType.CELL)
        """
        candidates = self.get_descendants(include_self=include_self)
        if tme_type is None and object_type is None:
            return candidates
        result = []
        for obj in candidates:
            t_match = (tme_type is None) or (obj.tme_type == tme_type)
            o_match = (object_type is None) or (obj.object_type == object_type)
            if t_match and o_match:
                result.append(obj)
        return result

    def filter_by_class(
        self, cls: Type[T], include_self: bool = False
    ) -> List[T]:
        """Return all descendants that are instances of *cls*."""
        return [
            obj  # type: ignore[return-value]
            for obj in self.get_descendants(include_self=include_self)
            if isinstance(obj, cls)
        ]

    def get_children_of_type(self, tme_type: TMEType) -> List["TMEObject"]:
        """Return direct (non-recursive) children with the given TMEType."""
        return [c for c in self.children if c.tme_type == tme_type]

    # ------------------------------------------------------------------
    # Properties / measurements
    # ------------------------------------------------------------------

    def update_properties(self, **kwargs: Any) -> None:
        """Merge *kwargs* into the properties dict."""
        self.properties.update(kwargs)
        self.modified = datetime.now()

    def get_property(self, key: str, default: Any = None) -> Any:
        """Return a property value, falling back to *default*."""
        return self.properties.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a single metadata entry."""
        self.metadata[key] = value
        self.modified = datetime.now()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_hierarchy_dict(self) -> Dict[str, Any]:
        """
        Recursively serialise this subtree to a nested dict.

        Format is compatible with QuPath's JSON object export.
        """
        return {
            "object_id": self.object_id,
            "name": self.name,
            "tme_type": self.tme_type.value,
            "object_type": self.object_type.value,
            "metadata": self.metadata,
            "properties": self.properties,
            "children": [c.to_hierarchy_dict() for c in self.children],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Shallow dict (no children). Subclasses should override."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "tme_type": self.tme_type.value,
            "object_type": self.object_type.value,
            "metadata": self.metadata,
            "properties": self.properties,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"object_id={self.object_id!r}, "
            f"tme_type={self.tme_type.value!r}, "
            f"children={len(self.children)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TMEObject):
            return NotImplemented
        return self.object_id == other.object_id

    def __hash__(self) -> int:
        return hash(self.object_id)


# ---------------------------------------------------------------------------
# Supporting data classes
# These are pure data holders, NOT hierarchy nodes, so dataclass is fine.
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    """
    Geometric representation of a TME object's spatial extent.

    Supports 2-D and 3-D coordinate arrays.
    bounds layout: [x_min, y_min, (z_min,) x_max, y_max, (z_max)]
    """
    type: GeometryType
    coordinates: Union[np.ndarray, List[Any]]
    bounds: Optional[np.ndarray] = None
    is_3d: bool = False

    def __post_init__(self) -> None:
        if self.bounds is None:
            self._calculate_bounds()

    def _calculate_bounds(self) -> None:
        coords = (
            np.array(self.coordinates)
            if isinstance(self.coordinates, list)
            else self.coordinates
        )
        if coords.ndim == 1:
            self.bounds = np.concatenate([coords, coords])
        elif coords.ndim == 2:
            self.bounds = np.concatenate(
                [np.min(coords, axis=0), np.max(coords, axis=0)]
            )
        if coords.ndim > 0:
            self.is_3d = coords.shape[-1] >= 3

    def area(self) -> float:
        """Bounding-box area (2-D) or volume (3-D)."""
        if self.bounds is None:
            return 0.0
        if not self.is_3d:
            return float(
                (self.bounds[3] - self.bounds[0]) *
                (self.bounds[4] - self.bounds[1])
            )
        return float(
            (self.bounds[3] - self.bounds[0]) *
            (self.bounds[4] - self.bounds[1]) *
            (self.bounds[5] - self.bounds[2])
        )

    def centroid(self) -> np.ndarray:
        """Bounding-box centroid as a numpy array."""
        if self.bounds is None:
            return np.zeros(3)
        if not self.is_3d:
            return np.array([
                (self.bounds[0] + self.bounds[3]) / 2,
                (self.bounds[1] + self.bounds[4]) / 2,
                0.0,
            ])
        return np.array([
            (self.bounds[0] + self.bounds[3]) / 2,
            (self.bounds[1] + self.bounds[4]) / 2,
            (self.bounds[2] + self.bounds[5]) / 2,
        ])


@dataclass
class Measurement:
    """A single quantitative measurement attached to a TME object."""
    name: str
    value: Union[float, int, str]
    unit: str = ""
    description: str = ""
    method: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "method": self.method,
            "confidence": self.confidence,
        }


@dataclass
class Classification:
    """Hierarchical classification label with optional probability score."""
    name: str
    probability: float = 1.0
    parent_class: Optional["Classification"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Colon-separated full path (e.g. ``Immune:T_Cell``)."""
        if self.parent_class:
            return f"{self.parent_class.full_name}:{self.name}"
        return self.name

    def is_subclass_of(self, classification: "Classification") -> bool:
        current = self.parent_class
        while current is not None:
            if current == classification:
                return True
            current = current.parent_class
        return False


@dataclass
class TMEMetadata:
    """
    Comprehensive metadata for a TME image/sample entry.

    Covers acquisition parameters, sample provenance, processing history,
    and clinical annotations.
    """
    # Acquisition
    image_dimensions: Optional[tuple] = None  # (W, H[, D, C])
    pixel_size: Optional[tuple] = None        # (x, y[, z]) µm/pixel
    magnification: Optional[float] = None
    channels: List[str] = field(default_factory=list)

    # Sample provenance
    sample_id: str = ""
    patient_id: str = ""
    tissue_type: str = ""
    stain_type: str = ""
    diagnosis: str = ""

    # Processing history
    preprocessing_steps: List[str] = field(default_factory=list)
    analysis_parameters: Dict[str, Any] = field(default_factory=dict)

    # Clinical annotations
    clinical_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_dimensions": self.image_dimensions,
            "pixel_size": self.pixel_size,
            "magnification": self.magnification,
            "channels": self.channels,
            "sample_id": self.sample_id,
            "patient_id": self.patient_id,
            "tissue_type": self.tissue_type,
            "stain_type": self.stain_type,
            "diagnosis": self.diagnosis,
            "preprocessing_steps": self.preprocessing_steps,
            "analysis_parameters": self.analysis_parameters,
            "clinical_data": self.clinical_data,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "TMEType",
    "ObjectType",
    "GeometryType",
    # Core hierarchy node
    "TMEObject",
    # Supporting data classes
    "Geometry",
    "Measurement",
    "Classification",
    "TMEMetadata",
]