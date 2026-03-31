"""
ROI Manager for TMEQuant — arbitrary annotation shapes and hierarchy integration.

Design
------
Every ROI is a first-class ``TMEObject`` in the hierarchy so it can be
queried, filtered, traversed, and serialised with the same API as cells
and fibers.  The ``ROIManager`` class owns the creation, validation,
and removal of ROI nodes; it delegates hierarchy placement to
``TMEHierarchy``.

Supported shape families
------------------------
    RECTANGLE   — axis-aligned bounding box  (cx, cy, width, height)
    CIRCLE      — filled circle / ellipse     (cx, cy, rx, ry)
    POLYGON     — arbitrary closed polygon    (list of (x, y) vertices)
    FREEHAND    — digitised free-hand stroke  (same as polygon, tagged differently)
    LINE        — open polyline               (list of (x, y) points)
    POINT       — single landmark             (x, y)

All shapes are stored internally as a ``Geometry`` dataclass (from
``core.base_models``) with the ``GeometryType`` enum as the tag.
The coordinate array is always a plain NumPy float32 array so no shapely
dependency is required at construction time.  Shapely is used *optionally*
for spatial queries (contains, intersects) when it is available.

Annotation types
----------------
ROI objects carry an ``annotation_type`` property string that records the
intended biological meaning:

    "tumor_boundary"   — delineates a tumour region
    "stroma"           — stromal zone
    "invasive_front"   — transition zone
    "necrosis"         — necrotic area
    "artifact"         — tissue/slide artifact (exclude from analysis)
    "landmark"         — reference point
    "measurement"      — distance / area measurement
    "custom"           — user-defined (free label via properties bag)

ROIObject
---------
A thin ``TMEObject`` subclass that adds:
    shape_type      : GeometryType
    annotation_type : str
    geometry        : Geometry (coordinates + bounds)
    label           : str      (display name, e.g. "Tumor core #1")
    locked          : bool     (prevent accidental edits)
    visible         : bool     (for rendering layers)

ROIManager
----------
    # Construction
    add_rectangle(cx, cy, w, h, ...)
    add_circle(cx, cy, r, ...)          # isotropic circle
    add_ellipse(cx, cy, rx, ry, ...)    # axis-aligned ellipse
    add_polygon(vertices, ...)          # arbitrary vertices
    add_freehand(points, ...)           # free-hand path (auto-closed)
    add_line(points, ...)               # open polyline
    add_point(x, y, ...)                # landmark point

    # Import from external formats
    from_qupath_geojson(path_or_dict)   # QuPath annotation GeoJSON
    from_bbox_list(bboxes, ...)         # [(x0,y0,x1,y1), ...]

    # Removal
    remove(roi_id)
    remove_all(annotation_type=None)    # remove all, or all of a type

    # Query
    get_all()                           -> List[ROIObject]
    get_by_type(annotation_type)        -> List[ROIObject]
    get_containing(x, y)               -> List[ROIObject]  (point-in-ROI)
    get_overlapping(other_roi)         -> List[ROIObject]

    # Hierarchy integration
    attach_to_hierarchy(hierarchy, parent_id=None)
    detach_from_hierarchy()

Usage example
-------------
    from tme_quant.core.roi_manager import ROIManager

    manager = ROIManager(image_id="img_001", pixel_size=0.5)

    # Manual annotations drawn in napari / QuPath
    t1 = manager.add_polygon(
        vertices=[(100,80),(200,80),(200,180),(100,180)],
        annotation_type="tumor_boundary",
        label="Tumor core A",
    )
    t2 = manager.add_circle(300, 300, radius=60,
        annotation_type="tumor_boundary", label="Tumor B")
    s1 = manager.add_freehand(
        points=[(50,50),(60,55),(70,52),(80,60),(70,70),(55,65),(50,50)],
        annotation_type="stroma", label="Stroma zone 1")

    # Import from QuPath GeoJSON export
    manager.from_qupath_geojson("annotations.geojson")

    # Remove a specific ROI
    manager.remove(t1.object_id)

    # Query
    tumors = manager.get_by_type("tumor_boundary")
    rois_at_point = manager.get_containing(150, 130)

    # Integrate with the TME hierarchy
    manager.attach_to_hierarchy(hierarchy, parent_id="img_001")
    # Now every ROI is a node under img_001 and can be queried with
    # hierarchy.get_objects_by_type(TMEType.ANNOTATION) etc.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base_models import (
    Geometry,
    GeometryType,
    ObjectType,
    TMEObject,
    TMEType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation type constants
# ─────────────────────────────────────────────────────────────────────────────

ANNOTATION_TYPES = {
    "tumor_boundary",
    "stroma",
    "invasive_front",
    "necrosis",
    "artifact",
    "landmark",
    "measurement",
    "custom",
}


# ─────────────────────────────────────────────────────────────────────────────
# ROIObject — a TMEObject node that carries a shape
# ─────────────────────────────────────────────────────────────────────────────

class ROIObject(TMEObject):
    """
    A single region-of-interest annotation as a hierarchy node.

    Inherits all TMEObject capabilities (add_child, filter_by_type,
    update_properties, set_metadata, …) and adds shape-specific fields.

    Parameters
    ----------
    object_id : str
    label : str
        Human-readable display name shown in the ROI manager list.
    shape_type : GeometryType
        Rectangle, polygon, circle, ellipse, line, point, freehand.
    geometry : Geometry
        Coordinate array + bounds.  Always stored as float32 NumPy array.
    annotation_type : str
        Biological meaning: "tumor_boundary", "stroma", etc.
    locked : bool
        When True the ROI cannot be moved or resized (prevents accidents).
    visible : bool
        Display flag for rendering layers (does not affect analysis).
    parent : TMEObject, optional
        Hierarchy parent (set by ROIManager.attach_to_hierarchy).
    """

    def __init__(
        self,
        object_id: str,
        label: str = "",
        shape_type: GeometryType = GeometryType.POLYGON,
        geometry: Optional[Geometry] = None,
        annotation_type: str = "custom",
        locked: bool = False,
        visible: bool = True,
        parent: Optional[TMEObject] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=label or object_id,
            tme_type=TMEType.ANNOTATION,
            object_type=ObjectType.REGION,
            parent=parent,
            metadata=metadata,
        )
        self.label:           str             = label or object_id
        self.shape_type:      GeometryType    = shape_type
        self.geometry:        Optional[Geometry] = geometry
        self.annotation_type: str             = annotation_type
        self.locked:          bool            = locked
        self.visible:         bool            = visible

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """Vertex coordinates as (N, 2) float32 array, or None."""
        if self.geometry is None:
            return None
        c = self.geometry.coordinates
        return np.asarray(c, dtype=np.float32) if c is not None else None

    @property
    def area(self) -> float:
        """Shoelace polygon area (0 for non-closed shapes)."""
        coords = self.coordinates
        if coords is None or len(coords) < 3:
            return 0.0
        x, y = coords[:, 0], coords[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @property
    def centroid(self) -> Optional[Tuple[float, float]]:
        """Centroid of the bounding box, or the single point for POINT ROIs."""
        if self.geometry is None:
            return None
        if self.shape_type == GeometryType.POINT:
            coords = self.coordinates
            return (float(coords[0, 0]), float(coords[0, 1])) if coords is not None else None
        b = self.geometry.bounds
        if b is None:
            return None
        return (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2))

    # ── Point containment (no shapely required) ───────────────────────────────

    def contains_point(self, x: float, y: float) -> bool:
        """
        Return True if (x, y) lies inside this ROI.

        Uses ray-casting for polygons / freehand / rectangles, radial test
        for circles / ellipses, and always False for lines / points.
        """
        coords = self.coordinates
        if coords is None:
            return False

        if self.shape_type in (GeometryType.POLYGON,
                                GeometryType.PATH,
                                GeometryType.RECTANGLE):
            return _ray_cast(x, y, coords)

        if self.shape_type == GeometryType.ELLIPSE:
            # Prefer parametric metadata (set by add_circle / add_ellipse).
            # Falls back to polygon ray-cast for externally constructed ROIs.
            cx = self.metadata.get('cx') if self.metadata else None
            cy = self.metadata.get('cy') if self.metadata else None
            rx = self.metadata.get('rx') if self.metadata else None
            ry = self.metadata.get('ry') if self.metadata else None
            if cx is not None and cy is not None and rx and ry:
                return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
            # Fallback: ray-cast on polygon approximation
            return _ray_cast(x, y, coords) if coords is not None else False

        return False

    # ── Overlap test ──────────────────────────────────────────────────────────

    def overlaps(self, other: "ROIObject") -> bool:
        """
        Return True if bounding boxes overlap (fast conservative test).

        When shapely is available a precise polygon intersection is used.
        """
        a, b = self.geometry, other.geometry
        if a is None or b is None or a.bounds is None or b.bounds is None:
            return False

        # Bounding-box overlap
        ax0, ay0, ax1, ay1 = a.bounds[0], a.bounds[1], a.bounds[2], a.bounds[3]
        bx0, by0, bx1, by1 = b.bounds[0], b.bounds[1], b.bounds[2], b.bounds[3]
        if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
            return False

        # Precise test if shapely is present
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            ca, cb = self.coordinates, other.coordinates
            if ca is not None and cb is not None and len(ca) >= 3 and len(cb) >= 3:
                return ShapelyPolygon(ca).intersects(ShapelyPolygon(cb))
        except Exception:
            pass

        return True  # bounding boxes overlap — conservative

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        coords = self.coordinates
        d.update({
            "label":           self.label,
            "shape_type":      self.shape_type.value,
            "annotation_type": self.annotation_type,
            "locked":          self.locked,
            "visible":         self.visible,
            "coordinates":     coords.tolist() if coords is not None else None,
            "area":            self.area,
            "centroid":        self.centroid,
        })
        return d

    def to_geojson_feature(self) -> Dict[str, Any]:
        """Return a GeoJSON Feature dict compatible with QuPath annotations."""
        coords = self.coordinates
        if coords is None:
            geom_json: Any = None
        elif self.shape_type == GeometryType.POINT:
            geom_json = {"type": "Point",
                         "coordinates": [float(coords[0, 0]), float(coords[0, 1])]}
        elif self.shape_type == GeometryType.LINE:
            geom_json = {"type": "LineString",
                         "coordinates": coords.tolist()}
        else:
            # Close the ring if not already closed
            ring = coords.tolist()
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            geom_json = {"type": "Polygon", "coordinates": [ring]}

        # Include image_id and pixel_size from metadata if present so the
        # saved file carries enough context to round-trip back to the right image.
        image_id   = self.metadata.get("image_id")   if self.metadata else None
        pixel_size = self.metadata.get("pixel_size") if self.metadata else None
        extra = {}
        if image_id:
            extra["image_id"]   = image_id
        if pixel_size is not None:
            extra["pixel_size"] = pixel_size

        return {
            "type": "Feature",
            "id": self.object_id,
            "geometry": geom_json,
            "properties": {
                "name":            self.label,
                "classification":  {"name": self.annotation_type},
                "isLocked":        self.locked,
                "measurements":    [],
                **extra,
                **self.properties,
            },
        }

    def __repr__(self) -> str:
        return (
            f"ROIObject(id={self.object_id!r}, "
            f"shape={self.shape_type.value}, "
            f"type={self.annotation_type!r}, "
            f"label={self.label!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ray_cast(px: float, py: float, coords: np.ndarray) -> bool:
    """Ray-casting point-in-polygon (no shapely required)."""
    n, inside, j = len(coords), False, len(coords) - 1
    for i in range(n):
        xi, yi = coords[i, 0], coords[i, 1]
        xj, yj = coords[j, 0], coords[j, 1]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _ellipse_to_polygon(cx, cy, rx, ry, n=64) -> np.ndarray:
    """Approximate an ellipse as a closed polygon with *n* vertices."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([
        cx + rx * np.cos(angles),
        cy + ry * np.sin(angles),
    ]).astype(np.float32)


def _rect_to_polygon(cx, cy, w, h) -> np.ndarray:
    """Convert a centre-based rectangle to a 4-vertex polygon."""
    hw, hh = w / 2, h / 2
    return np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)


def _unique_id(prefix: str, counter: list) -> str:
    counter[0] += 1
    return f"{prefix}_{counter[0]:04d}"


# ─────────────────────────────────────────────────────────────────────────────
# ROIManager
# ─────────────────────────────────────────────────────────────────────────────

class ROIManager:
    """
    Creates, stores, and manages ROI annotations for a single image.

    All ROIs are kept in an ordered list ``_rois`` and also in a flat
    ``_by_id`` dict for O(1) lookup.  When attached to a TMEHierarchy the
    ROI nodes become first-class hierarchy nodes queryable with the
    standard API.

    Parameters
    ----------
    image_id : str
        Identifier of the parent image.  Used as a prefix for auto-generated
        object_ids (``<image_id>_roi_0001``, etc.).
    pixel_size : float
        µm per pixel.  Stored as metadata on each ROI; used by area / length
        calculations when converting pixel coordinates to physical units.
    """

    def __init__(self, image_id: str = "image", pixel_size: float = 1.0) -> None:
        self.image_id   = image_id
        self.pixel_size = pixel_size
        self._rois:  List[ROIObject]     = []
        self._by_id: Dict[str, ROIObject] = {}
        self._counter = [0]             # mutable counter for id generation
        self._hierarchy_parent: Optional[TMEObject] = None

    # ── ID helpers ────────────────────────────────────────────────────────────

    def _next_id(self) -> str:
        return _unique_id(f"{self.image_id}_roi", self._counter)

    def _register(self, roi: ROIObject) -> ROIObject:
        self._rois.append(roi)
        self._by_id[roi.object_id] = roi
        # If already attached to a hierarchy parent, wire in immediately
        if self._hierarchy_parent is not None:
            self._hierarchy_parent.add_child(roi)
            # Also update the hierarchy index if it exists on the parent's root
            root = self._hierarchy_parent.get_root()
            if root is not None and hasattr(root, '_hierarchy_ref'):
                h = root._hierarchy_ref
                if hasattr(h, '_index'):
                    h._index.add(roi)
        return roi

    # ── Shape constructors ────────────────────────────────────────────────────

    def add_rectangle(
        self,
        cx: float, cy: float,
        width: float, height: float,
        annotation_type: str = "custom",
        label: str = "",
        object_id: Optional[str] = None,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add an axis-aligned rectangle.

        Parameters
        ----------
        cx, cy : float
            Centre of the rectangle in pixel coordinates.
        width, height : float
            Dimensions in pixels.
        """
        coords = _rect_to_polygon(cx, cy, width, height)
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Rectangle {self._counter[0]}",
            shape_type=GeometryType.RECTANGLE,
            geometry=Geometry(type=GeometryType.RECTANGLE, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size, **kwargs},
        ))

    def add_circle(
        self,
        cx: float, cy: float,
        radius: float,
        annotation_type: str = "custom",
        label: str = "",
        object_id: Optional[str] = None,
        n_vertices: int = 64,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add a circle, represented as a regular polygon with *n_vertices*.

        Parameters
        ----------
        cx, cy : float
            Centre in pixel coordinates.
        radius : float
            Radius in pixels.
        """
        coords = _ellipse_to_polygon(cx, cy, radius, radius, n=n_vertices)
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Circle {self._counter[0]}",
            shape_type=GeometryType.ELLIPSE,
            geometry=Geometry(type=GeometryType.ELLIPSE,
                              # Store (cx,cy) and (rx,ry) as first two rows
                              # alongside the polygon approximation
                              coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size,
                      "cx": cx, "cy": cy, "rx": radius, "ry": radius,
                      **kwargs},
        ))

    def add_ellipse(
        self,
        cx: float, cy: float,
        rx: float, ry: float,
        annotation_type: str = "custom",
        label: str = "",
        object_id: Optional[str] = None,
        n_vertices: int = 64,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add an axis-aligned ellipse.

        Parameters
        ----------
        cx, cy : float   Centre in pixel coordinates.
        rx, ry : float   Semi-axes in pixels (horizontal, vertical).
        """
        coords = _ellipse_to_polygon(cx, cy, rx, ry, n=n_vertices)
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Ellipse {self._counter[0]}",
            shape_type=GeometryType.ELLIPSE,
            geometry=Geometry(type=GeometryType.ELLIPSE, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size,
                      "cx": cx, "cy": cy, "rx": rx, "ry": ry, **kwargs},
        ))

    def add_polygon(
        self,
        vertices: Union[List[Tuple[float, float]], np.ndarray],
        annotation_type: str = "custom",
        label: str = "",
        object_id: Optional[str] = None,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add an arbitrary closed polygon.

        Parameters
        ----------
        vertices : list of (x, y) tuples or (N, 2) ndarray
            Vertices in pixel coordinates.  The polygon is auto-closed
            (first vertex is not required to be repeated at the end).
        """
        coords = np.asarray(vertices, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                "vertices must be a list of (x, y) pairs or a (N, 2) array."
            )
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Polygon {self._counter[0]}",
            shape_type=GeometryType.POLYGON,
            geometry=Geometry(type=GeometryType.POLYGON, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size, **kwargs},
        ))

    def add_freehand(
        self,
        points: Union[List[Tuple[float, float]], np.ndarray],
        annotation_type: str = "custom",
        label: str = "",
        object_id: Optional[str] = None,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add a free-hand digitised contour (treated as a closed polygon).

        Parameters
        ----------
        points : list of (x, y) tuples or (N, 2) ndarray
            The raw digitised path.  Will be stored as-is (no simplification).
        """
        coords = np.asarray(points, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("points must be (x, y) pairs or a (N, 2) array.")
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Freehand {self._counter[0]}",
            shape_type=GeometryType.PATH,
            geometry=Geometry(type=GeometryType.PATH, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size, **kwargs},
        ))

    def add_line(
        self,
        points: Union[List[Tuple[float, float]], np.ndarray],
        annotation_type: str = "measurement",
        label: str = "",
        object_id: Optional[str] = None,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """
        Add an open polyline (e.g. for distance measurements).

        Parameters
        ----------
        points : list of (x, y) tuples or (N, 2) ndarray
            Ordered path points.
        """
        coords = np.asarray(points, dtype=np.float32)
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Line {self._counter[0]}",
            shape_type=GeometryType.LINE,
            geometry=Geometry(type=GeometryType.LINE, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size, **kwargs},
        ))

    def add_point(
        self,
        x: float, y: float,
        annotation_type: str = "landmark",
        label: str = "",
        object_id: Optional[str] = None,
        locked: bool = False,
        **kwargs,
    ) -> ROIObject:
        """Add a single landmark point."""
        coords = np.array([[x, y]], dtype=np.float32)
        return self._register(ROIObject(
            object_id=object_id or self._next_id(),
            label=label or f"Point {self._counter[0]}",
            shape_type=GeometryType.POINT,
            geometry=Geometry(type=GeometryType.POINT, coordinates=coords),
            annotation_type=annotation_type,
            locked=locked,
            metadata={"pixel_size": self.pixel_size, **kwargs},
        ))

    # ── Bulk import ───────────────────────────────────────────────────────────

    def from_qupath_geojson(
        self,
        source: Union[str, Path, Dict],
    ) -> List[ROIObject]:
        """
        Import annotations from a QuPath GeoJSON export.

        Supports FeatureCollection and individual Feature dicts.
        Geometry types mapped: Polygon → polygon, Point → point,
        LineString → line, MultiPolygon → one ROI per sub-polygon.

        Parameters
        ----------
        source : str, Path, or dict
            Path to a ``.geojson`` file, or an already-parsed dict.

        Returns
        -------
        list of newly created ROIObject instances.
        """
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = source

        features = (
            data.get("features", [data])
            if data.get("type") == "FeatureCollection"
            else [data]
        )
        created: List[ROIObject] = []
        for feat in features:
            geom  = feat.get("geometry") or {}
            props = feat.get("properties") or {}
            name  = (props.get("name") or
                     props.get("label") or
                     feat.get("id") or "")
            ann_type = (
                (props.get("classification") or {}).get("name", "custom")
                .lower().replace(" ", "_")
            )
            if ann_type not in ANNOTATION_TYPES:
                ann_type = "custom"

            gtype = geom.get("type", "")
            coords_raw = geom.get("coordinates")
            oid = feat.get("id") or self._next_id()

            if gtype == "Polygon" and coords_raw:
                ring = np.array(coords_raw[0], dtype=np.float32)[:, :2]
                roi = self.add_polygon(ring, annotation_type=ann_type,
                                       label=name, object_id=str(oid))
                created.append(roi)

            elif gtype == "MultiPolygon" and coords_raw:
                for k, sub in enumerate(coords_raw):
                    ring = np.array(sub[0], dtype=np.float32)[:, :2]
                    roi = self.add_polygon(ring, annotation_type=ann_type,
                                           label=f"{name}_{k}",
                                           object_id=f"{oid}_{k}")
                    created.append(roi)

            elif gtype == "Point" and coords_raw:
                x, y = float(coords_raw[0]), float(coords_raw[1])
                roi = self.add_point(x, y, annotation_type=ann_type,
                                     label=name, object_id=str(oid))
                created.append(roi)

            elif gtype == "LineString" and coords_raw:
                pts = np.array(coords_raw, dtype=np.float32)[:, :2]
                roi = self.add_line(pts, annotation_type=ann_type,
                                    label=name, object_id=str(oid))
                created.append(roi)

        return created

    def from_bbox_list(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        annotation_type: str = "tumor_boundary",
        label_prefix: str = "bbox",
    ) -> List[ROIObject]:
        """
        Import a list of axis-aligned bounding boxes.

        Parameters
        ----------
        bboxes : list of (x0, y0, x1, y1)
            Top-left and bottom-right corners in pixel coordinates.
        """
        created = []
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w, h   = abs(x1 - x0), abs(y1 - y0)
            roi = self.add_rectangle(
                cx, cy, w, h,
                annotation_type=annotation_type,
                label=f"{label_prefix}_{i+1}",
            )
            created.append(roi)
        return created

    def from_tumor_region(
        self,
        tumor_region,
        label: Optional[str] = None,
        locked: bool = True,
    ) -> Optional["ROIObject"]:
        """
        Convert a TumorRegion (from RegionManager.detect_tumor_regions()) into
        an ROIObject and register it in this manager.

        Parameters
        ----------
        tumor_region : TumorRegion
            A TumorRegion produced by RegionManager.detect_tumor_regions().
            Must have a non-None geometry with (N, 2) polygon coordinates.
        label : str, optional
            Display name.  Defaults to ``tumor_region.name`` or
            ``tumor_region.object_id``.
        locked : bool
            Lock the ROI to prevent accidental editing.  Default True for
            auto-detected boundaries (manual annotations default to False).

        Returns
        -------
        ROIObject, or None if the TumorRegion has no valid polygon geometry.
        """
        import warnings

        geom = getattr(tumor_region, 'geometry', None)
        if geom is None or geom.coordinates is None:
            warnings.warn(
                f"TumorRegion '{tumor_region.object_id}' has no polygon geometry "
                "\u2014 skipping conversion to ROIObject."
            )
            return None

        coords = np.asarray(geom.coordinates, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 3:
            warnings.warn(
                f"TumorRegion '{tumor_region.object_id}' geometry has insufficient "
                f"vertices ({coords.shape}) \u2014 skipping."
            )
            return None

        return self.add_polygon(
            vertices=coords[:, :2],
            annotation_type="tumor_boundary",
            label=label or getattr(tumor_region, 'name', tumor_region.object_id),
            object_id=f"auto_{tumor_region.object_id}",
            locked=locked,
            source="region_manager_auto",
            original_id=tumor_region.object_id,
        )

    # ── Removal ───────────────────────────────────────────────────────────────

    def remove(self, roi_id: str) -> bool:
        """
        Remove the ROI with the given id.

        Also detaches it from the TME hierarchy if attached.
        Returns True if found and removed, False if not found.
        """
        roi = self._by_id.pop(roi_id, None)
        if roi is None:
            return False
        self._rois.remove(roi)
        # Update hierarchy index before detaching
        if roi.parent is not None:
            root = roi.parent.get_root()
            if root is not None and hasattr(root, '_hierarchy_ref'):
                h = root._hierarchy_ref
                if hasattr(h, '_index'):
                    h._index.remove(roi)
        roi.detach()
        return True

    def remove_all(self, annotation_type: Optional[str] = None) -> int:
        """
        Remove all ROIs, or all ROIs of a specific annotation type.

        Returns the number of ROIs removed.
        """
        if annotation_type is None:
            targets = list(self._rois)
        else:
            targets = [r for r in self._rois if r.annotation_type == annotation_type]
        for roi in targets:
            self._by_id.pop(roi.object_id, None)
            self._rois.remove(roi)
            roi.detach()
        return len(targets)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_all(self) -> List[ROIObject]:
        """Return all ROIs in insertion order."""
        return list(self._rois)

    def get_by_id(self, roi_id: str) -> Optional[ROIObject]:
        """Return the ROI with the given id, or None."""
        return self._by_id.get(roi_id)

    def get_by_type(self, annotation_type: str) -> List[ROIObject]:
        """Return all ROIs of a specific annotation type."""
        return [r for r in self._rois if r.annotation_type == annotation_type]

    def get_by_shape(self, shape_type: GeometryType) -> List[ROIObject]:
        """Return all ROIs of a specific shape type."""
        return [r for r in self._rois if r.shape_type == shape_type]

    def get_containing(self, x: float, y: float) -> List[ROIObject]:
        """
        Return all ROIs whose interior contains the point (x, y).

        Only closed shapes (polygon, rectangle, ellipse, freehand) are
        tested; lines and points are skipped.
        """
        closed = {GeometryType.POLYGON, GeometryType.RECTANGLE,
                  GeometryType.ELLIPSE, GeometryType.PATH}
        return [
            r for r in self._rois
            if r.shape_type in closed and r.contains_point(x, y)
        ]

    def get_overlapping(self, roi: ROIObject) -> List[ROIObject]:
        """Return all ROIs that overlap with *roi* (bounding box + polygon)."""
        return [r for r in self._rois if r is not roi and r.overlaps(roi)]

    def count(self, annotation_type: Optional[str] = None) -> int:
        """Return total count, or count for a specific annotation type."""
        if annotation_type is None:
            return len(self._rois)
        return sum(1 for r in self._rois if r.annotation_type == annotation_type)

    # ── Hierarchy integration ─────────────────────────────────────────────────

    def attach_to_hierarchy(
        self,
        hierarchy: Any,          # TMEHierarchy — avoid circular import
        parent_id: Optional[str] = None,
    ) -> None:
        """
        Wire all current (and future) ROI nodes into *hierarchy*.

        Parameters
        ----------
        hierarchy : TMEHierarchy
            The hierarchy manager to attach to.
        parent_id : str, optional
            object_id of the parent node.  Defaults to the hierarchy root.
        """
        parent = (
            hierarchy.get_object(parent_id)
            if parent_id is not None
            else hierarchy.root
        )
        if parent is None:
            raise ValueError(
                f"Parent '{parent_id}' not found in hierarchy."
            )
        self._hierarchy_parent = parent
        # Store back-reference on root so late adds/removes can update the index
        root = parent.get_root()
        if root is not None:
            root._hierarchy_ref = hierarchy
        _has_index = hasattr(hierarchy, '_index')
        for roi in self._rois:
            if roi.parent is None:
                parent.add_child(roi)
                if _has_index:
                    hierarchy._index.add(roi)

    def detach_from_hierarchy(self) -> None:
        """
        Remove all ROI nodes from the TME hierarchy (keeps them in the manager).
        """
        for roi in self._rois:
            roi.detach()
        self._hierarchy_parent = None

    # ── Export ────────────────────────────────────────────────────────────────


    # ── ROI transfer between images ───────────────────────────────────────────

    def transfer_to(
        self,
        target_manager: "ROIManager",
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        roi_ids: Optional[List[str]] = None,
        copy_metadata: bool = True,
    ) -> List["ROIObject"]:
        """
        Copy ROIs from this manager into *target_manager*, optionally
        rescaling and/or shifting their coordinates.

        This is the base transfer primitive.  Use the higher-level helpers:
          ``transfer_same_size()``      — no transform needed
          ``transfer_scaled()``         — known pixel-size or dimension ratio
          ``transfer_with_transform()`` — registration affine matrix

        Parameters
        ----------
        target_manager : ROIManager
            Destination manager (may belong to a different image).
        scale_x, scale_y : float
            Multiplicative scale applied to x and y coordinates.
            1.0 = no scaling (same size images or same pixel size).
        offset_x, offset_y : float
            Additive offset applied after scaling (pixels in target space).
        roi_ids : list of str, optional
            Subset of ROI ids to transfer.  None = transfer all.
        copy_metadata : bool
            Copy the source ROI's metadata to the new ROI (except image_id
            which is updated to target_manager.image_id).

        Returns
        -------
        list of newly created ROIObject instances in *target_manager*.
        """
        sources = (
            [self._by_id[i] for i in roi_ids if i in self._by_id]
            if roi_ids is not None
            else list(self._rois)
        )
        created: List[ROIObject] = []
        for roi in sources:
            coords = roi.coordinates
            if coords is None:
                continue

            new_coords = coords.copy().astype(np.float32)
            new_coords[:, 0] = new_coords[:, 0] * scale_x + offset_x
            new_coords[:, 1] = new_coords[:, 1] * scale_y + offset_y

            meta = {}
            if copy_metadata and roi.metadata:
                meta = dict(roi.metadata)
            meta["image_id"]      = target_manager.image_id
            meta["pixel_size"]    = target_manager.pixel_size
            meta["transferred_from"] = self.image_id
            meta["transferred_from_id"] = roi.object_id

            new_roi = target_manager.add_polygon(
                vertices=new_coords,
                annotation_type=roi.annotation_type,
                label=roi.label,
                locked=roi.locked,
                **{k: v for k, v in meta.items()
                   if k not in ("image_id", "pixel_size",
                                "transferred_from", "transferred_from_id")},
            )
            # Overwrite metadata with the full dict (add_polygon only passes **kwargs)
            new_roi.metadata.update(meta)
            created.append(new_roi)

        return created

    def transfer_same_size(
        self,
        target_manager: "ROIManager",
        roi_ids: Optional[List[str]] = None,
    ) -> List["ROIObject"]:
        """
        Copy ROIs to *target_manager* with no coordinate change.

        Use when source and target images have the same pixel dimensions
        AND the same pixel size (e.g. the registered H&E and the SHG image
        from the same acquisition, both 2048×2048 at 0.5 µm/px).

        Parameters
        ----------
        target_manager : ROIManager
            Destination manager.
        roi_ids : list of str, optional
            Subset to transfer.  None = all.
        """
        return self.transfer_to(
            target_manager,
            scale_x=1.0, scale_y=1.0,
            offset_x=0.0, offset_y=0.0,
            roi_ids=roi_ids,
        )

    def transfer_scaled(
        self,
        target_manager: "ROIManager",
        source_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        source_pixel_size: Optional[float] = None,
        target_pixel_size: Optional[float] = None,
        roi_ids: Optional[List[str]] = None,
    ) -> List["ROIObject"]:
        """
        Copy ROIs to *target_manager*, rescaling coordinates to fit a
        different image size or pixel size.

        Two scaling modes — use whichever information you have:

        **Pixel-size mode** (recommended when pixel sizes are known):
            ``scale = source_pixel_size / target_pixel_size``
            A 0.5 µm/px source ROI transferred to a 0.25 µm/px target
            doubles all coordinates (the same physical region covers twice
            as many pixels at finer resolution).

        **Dimension mode** (fallback when pixel sizes are unknown):
            ``scale_x = target_width  / source_width``
            ``scale_y = target_height / source_height``
            Proportionally rescales so the ROI covers the same *fraction*
            of the image area.

        If both pixel sizes and shapes are supplied, pixel-size mode takes
        priority.

        Parameters
        ----------
        target_manager : ROIManager
            Destination manager.
        source_shape : (H, W)
            Pixel dimensions of the source image.
        target_shape : (H, W)
            Pixel dimensions of the target image.
        source_pixel_size : float, optional
            µm per pixel of the source image.
        target_pixel_size : float, optional
            µm per pixel of the target image.
        roi_ids : list of str, optional
            Subset to transfer.  None = all.
        """
        if (source_pixel_size is not None
                and target_pixel_size is not None
                and target_pixel_size > 0):
            # Physical-unit scaling: keep µm coordinates constant
            s = source_pixel_size / target_pixel_size
            scale_x = scale_y = s
        else:
            # Dimension-ratio scaling
            src_h, src_w = source_shape
            tgt_h, tgt_w = target_shape
            scale_x = tgt_w / src_w if src_w > 0 else 1.0
            scale_y = tgt_h / src_h if src_h > 0 else 1.0

        return self.transfer_to(
            target_manager,
            scale_x=scale_x, scale_y=scale_y,
            roi_ids=roi_ids,
        )

    def transfer_with_transform(
        self,
        target_manager: "ROIManager",
        transform_matrix: "np.ndarray",
        roi_ids: Optional[List[str]] = None,
    ) -> List["ROIObject"]:
        """
        Copy ROIs to *target_manager*, warping each vertex through a 2-D
        affine transform matrix.

        Use this when the target image was registered to the source (or
        vice versa) and you have the registration transform.

        Parameters
        ----------
        target_manager : ROIManager
            Destination manager.
        transform_matrix : (3, 3) or (2, 3) ndarray
            Affine transform matrix in homogeneous coordinates.
            Follows the scikit-image / OpenCV convention:
              output_point = M @ [x, y, 1]^T

            If your registration produces a (2, 3) matrix (e.g. from
            cv2.getAffineTransform), pad it to (3, 3):
              M = np.vstack([M23, [0, 0, 1]])

            Typical source: the ``transform`` field of
            ``HESHGRegistration.register()`` result, converted to a matrix
            via ``result.transform.params`` (skimage AffineTransform).
        roi_ids : list of str, optional
            Subset to transfer.  None = all.
        """
        M = np.asarray(transform_matrix, dtype=np.float64)
        if M.shape == (2, 3):
            M = np.vstack([M, [0.0, 0.0, 1.0]])
        if M.shape != (3, 3):
            raise ValueError(
                f"transform_matrix must be (3,3) or (2,3), got {M.shape}."
            )

        sources = (
            [self._by_id[i] for i in roi_ids if i in self._by_id]
            if roi_ids is not None
            else list(self._rois)
        )
        created: List[ROIObject] = []
        for roi in sources:
            coords = roi.coordinates
            if coords is None:
                continue

            # Apply affine: [x', y', 1]^T = M @ [x, y, 1]^T
            n = len(coords)
            ones = np.ones((n, 1), dtype=np.float64)
            xy1  = np.hstack([coords[:, :2].astype(np.float64), ones])
            transformed = (M @ xy1.T).T            # (n, 3)
            new_coords   = transformed[:, :2].astype(np.float32)

            meta = dict(roi.metadata) if roi.metadata else {}
            meta["image_id"]           = target_manager.image_id
            meta["pixel_size"]         = target_manager.pixel_size
            meta["transferred_from"]   = self.image_id
            meta["transferred_from_id"] = roi.object_id
            meta["transform_applied"]  = "affine"

            new_roi = target_manager.add_polygon(
                vertices=new_coords,
                annotation_type=roi.annotation_type,
                label=roi.label,
                locked=roi.locked,
            )
            new_roi.metadata.update(meta)
            created.append(new_roi)

        return created

    def to_geojson(self) -> Dict[str, Any]:
        """Export all ROIs as a GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [r.to_geojson_feature() for r in self._rois],
        }

    def save_geojson(self, path: Union[str, Path]) -> None:
        """Save all ROIs to a ``.geojson`` file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_geojson(), fh, indent=2)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rois)

    def __iter__(self):
        return iter(self._rois)

    def __repr__(self) -> str:
        from collections import Counter
        type_counts = Counter(r.annotation_type for r in self._rois)
        return (
            f"ROIManager(image_id={self.image_id!r}, "
            f"n_rois={len(self._rois)}, "
            f"types={dict(type_counts)})"
        )