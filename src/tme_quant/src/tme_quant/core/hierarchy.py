"""
Hierarchy management for TME objects.

TMEHierarchy wraps the root TMEObject and provides a manager-level API
for adding, retrieving, querying, and bulk-populating objects across the
full tree.  The underlying traversal logic lives on TMEObject itself;
this class is a thin convenience layer used by TMEProject.

New in this version
-------------------
  spatial_assign(objects, regions, fallback_parent, pixel_size)
      Point-in-polygon assignment: places each object under the region
      whose polygon contains its centroid.  Falls back to fallback_parent
      for objects that do not fall inside any region.

  attach_fiber_result(result, parent, image_id, pixel_size)
      Bulk-converts an ExtractionResult (list of FiberProperties) to
      FiberObject hierarchy nodes and attaches them under *parent*.

  attach_cell_result(result, parent, image_id)
      Bulk-converts a SegmentationResult (list of CellObject / CellProperties)
      and attaches the cells under *parent*.

  HierarchyIndex
      Maintained alongside the tree so that get_object() and
      get_objects_by_type() are O(1) rather than O(n) full-tree scans.
      Updated automatically on every add_object() / remove_object() call
      as well as after every spatial_assign / attach_* call.
      Direct tree mutations via TMEObject.add_child() / detach() bypass the
      index; call hierarchy.rebuild_index() afterward if needed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base_models import TMEObject, TMEType, ObjectType


# ─────────────────────────────────────────────────────────────────────────────
# Internal flat index
# ─────────────────────────────────────────────────────────────────────────────

class _HierarchyIndex:
    """
    Flat O(1) lookup structures maintained alongside the tree.

    Attributes
    ----------
    by_id : dict[str, TMEObject]
        Maps object_id → object.
    by_tme_type : dict[TMEType, list[TMEObject]]
        Maps TMEType → list of matching objects.
    by_object_type : dict[ObjectType, list[TMEObject]]
        Maps ObjectType → list of matching objects.
    """

    def __init__(self) -> None:
        self.by_id:          Dict[str, TMEObject]           = {}
        self.by_tme_type:    Dict[TMEType, List[TMEObject]] = defaultdict(list)
        self.by_object_type: Dict[ObjectType, List[TMEObject]] = defaultdict(list)

    def add(self, obj: TMEObject) -> None:
        self.by_id[obj.object_id] = obj
        self.by_tme_type[obj.tme_type].append(obj)
        self.by_object_type[obj.object_type].append(obj)

    def remove(self, obj: TMEObject) -> None:
        self.by_id.pop(obj.object_id, None)
        lst = self.by_tme_type.get(obj.tme_type, [])
        if obj in lst:
            lst.remove(obj)
        lst2 = self.by_object_type.get(obj.object_type, [])
        if obj in lst2:
            lst2.remove(obj)

    def rebuild(self, root: TMEObject) -> None:
        self.by_id.clear()
        self.by_tme_type.clear()
        self.by_object_type.clear()
        for obj in root.get_descendants(include_self=True):
            self.add(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper — point-in-polygon using ray casting
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(px: float, py: float, coords: np.ndarray) -> bool:
    """
    Return True if point (px, py) lies inside the polygon defined by *coords*
    (an (N, 2) float array of vertex coordinates).

    Uses the ray-casting algorithm; handles the case where shapely is absent.
    """
    n = len(coords)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = coords[i]
        xj, yj = coords[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _centroid_of_object(obj: TMEObject) -> Optional[Tuple[float, float]]:
    """
    Extract a 2-D centroid from a TMEObject regardless of subclass.

    Tries, in order:
      1. obj.centroid      (CellObject — tuple (x, y))
      2. obj.get_center_coordinates() (FiberObject — ndarray)
      3. obj.center_coordinates  (FiberProperties — property)
    Returns None if no centroid can be found.
    """
    centroid = getattr(obj, 'centroid', None)
    if centroid is not None and not isinstance(centroid, property):
        if hasattr(centroid, '__len__') and len(centroid) >= 2:
            return float(centroid[0]), float(centroid[1])

    if hasattr(obj, 'get_center_coordinates'):
        c = obj.get_center_coordinates()
        if c is not None and len(c) >= 2:
            return float(c[0]), float(c[1])

    c = getattr(obj, 'center_coordinates', None)
    if c is not None and hasattr(c, '__len__') and len(c) >= 2:
        return float(c[0]), float(c[1])

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TMEHierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TMEHierarchy:
    """
    Manager for a TMEObject hierarchy rooted at a single node.

    Core API (unchanged)
    --------------------
    get_object(object_id)          -> Optional[TMEObject]   O(1) via index
    add_object(obj, parent=...)    -> None
    get_objects_by_type(type)      -> List[TMEObject]        O(1) via index
    get_descendants(object_id)     -> List[TMEObject]
    get_children(object_id)        -> List[TMEObject]
    validate_hierarchy()           -> List[str]
    export_to_qupath()             -> Dict[str, Any]

    Convenience API (new)
    ----------------------
    spatial_assign(objects, regions, fallback_parent, pixel_size)
    attach_fiber_result(result, parent, image_id, pixel_size)
    attach_cell_result(result, parent, image_id)
    rebuild_index()
    """

    def __init__(self, root: Optional[TMEObject] = None) -> None:
        if root is None:
            root = TMEObject(object_id="root", name="root", tme_type=TMEType.PROJECT)
        self.root: TMEObject = root
        self._index = _HierarchyIndex()
        self._index.rebuild(self.root)

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """
        Rebuild the flat lookup index from the current tree state.

        Call this after performing direct tree mutations via
        TMEObject.add_child() / detach() that bypass the hierarchy manager.
        """
        self._index.rebuild(self.root)

    # ------------------------------------------------------------------
    # Retrieval — O(1) via index
    # ------------------------------------------------------------------

    def get_object(self, object_id: str) -> Optional[TMEObject]:
        """Find and return an object by id.  O(1) via index."""
        return self._index.by_id.get(object_id)

    def get_objects_by_type(
        self,
        object_type: Union[TMEType, ObjectType],
    ) -> List[TMEObject]:
        """
        Return all objects with the given type.  O(1) via index.

        Accepts either TMEType or ObjectType for backward compatibility.
        """
        if isinstance(object_type, TMEType):
            return list(self._index.by_tme_type.get(object_type, []))
        else:
            return list(self._index.by_object_type.get(object_type, []))

    def get_descendants(self, object_id: str) -> List[TMEObject]:
        """Return all descendants of the object with the given id."""
        node = self.get_object(object_id)
        return node.get_descendants() if node is not None else []

    def get_children(self, object_id: str) -> List[TMEObject]:
        """Return direct children of the object with the given id."""
        node = self.get_object(object_id)
        return list(node.children) if node is not None else []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_object(
        self,
        obj: TMEObject,
        parent: Optional[Union[TMEObject, str]] = None,
    ) -> None:
        """
        Add *obj* to the hierarchy and update the index.

        Parameters
        ----------
        obj:
            TMEObject to insert.
        parent:
            TMEObject instance, object_id string, or None (→ root).
        """
        if parent is None:
            self.root.add_child(obj)
        elif isinstance(parent, str):
            parent_node = self.get_object(parent)
            if parent_node is None:
                raise ValueError(
                    f"Parent object '{parent}' not found in hierarchy."
                )
            parent_node.add_child(obj)
        else:
            parent.add_child(obj)
        # Index the new node and all its descendants
        for node in obj.get_descendants(include_self=True):
            self._index.add(node)

    def remove_object(self, object_id: str) -> bool:
        """
        Remove an object (and its subtree) from the hierarchy.

        Returns True if found and removed, False otherwise.
        """
        node = self.get_object(object_id)
        if node is None:
            return False
        # Remove subtree from index before detaching
        for n in node.get_descendants(include_self=True):
            self._index.remove(n)
        node.detach()
        return True

    # ------------------------------------------------------------------
    # Convenience A — spatial assignment
    # ------------------------------------------------------------------

    def spatial_assign(
        self,
        objects: List[TMEObject],
        regions: List[TMEObject],
        fallback_parent: TMEObject,
        pixel_size: float = 1.0,
    ) -> Dict[str, int]:
        """
        Assign each object to the region whose polygon contains its centroid.

        Objects whose centroid falls outside every region are attached to
        *fallback_parent*.  All assigned objects are added to the hierarchy
        index automatically.

        The polygon geometry is read from ``region.geometry.coordinates``
        (a ``(N, 2)`` numpy array of vertex coordinates in pixel units).
        If a region has no geometry, it is skipped.

        Parameters
        ----------
        objects : list of TMEObject
            Cells, fibers, or any other objects to assign.
        regions : list of TMEObject
            Candidate parent regions, each with a ``.geometry`` attribute
            whose coordinates form a closed polygon.
        fallback_parent : TMEObject
            Where unassigned objects go (e.g. the ImageEntry node).
        pixel_size : float
            µm / pixel — used to convert centroid coordinates if the
            region coordinates are in µm but centroids are in pixels.
            Pass ``1.0`` (default) when both are in the same unit.

        Returns
        -------
        dict with keys:
            'assigned'   — number of objects placed under a matching region
            'fallback'   — number of objects placed under fallback_parent
            'no_centroid'— number of objects skipped (no centroid found)
        """
        # Pre-compute polygon coords for each region that has geometry
        region_polygons: List[Tuple[TMEObject, np.ndarray]] = []
        for region in regions:
            geom = getattr(region, 'geometry', None)
            if geom is None:
                continue
            coords = geom.coordinates
            if coords is None:
                continue
            arr = np.array(coords) if not isinstance(coords, np.ndarray) else coords
            if arr.ndim == 2 and arr.shape[1] >= 2:
                region_polygons.append((region, arr[:, :2]))

        counts = {'assigned': 0, 'fallback': 0, 'no_centroid': 0}

        for obj in objects:
            pt = _centroid_of_object(obj)
            if pt is None:
                fallback_parent.add_child(obj)
                self._index.add(obj)
                counts['no_centroid'] += 1
                continue

            px, py = pt[0] / pixel_size, pt[1] / pixel_size
            matched = False
            for region, poly in region_polygons:
                if _point_in_polygon(px, py, poly):
                    region.add_child(obj)
                    self._index.add(obj)
                    counts['assigned'] += 1
                    matched = True
                    break

            if not matched:
                fallback_parent.add_child(obj)
                self._index.add(obj)
                counts['fallback'] += 1

        return counts

    # ------------------------------------------------------------------
    # Convenience B — bulk attach from analysis results
    # ------------------------------------------------------------------

    def attach_fiber_result(
        self,
        result: Any,
        parent: TMEObject,
        image_id: str = "image",
        pixel_size: Optional[float] = None,
    ) -> List[TMEObject]:
        """
        Convert an ExtractionResult to FiberObject hierarchy nodes and
        attach them all under *parent* in one call.

        Parameters
        ----------
        result : ExtractionResult
            Returned by FiberAnalyzer.extract_fibers_2d / extract_fibers_3d.
            ``result.fibers`` may contain FiberProperties or FiberObject
            instances — both are handled.
        parent : TMEObject
            Node to attach fibers to (e.g. a TumorRegion or ImageEntry).
        image_id : str
            Prefix used when generating object_ids for new FiberObjects.
        pixel_size : float, optional
            Overrides ``result.pixel_size`` when provided.

        Returns
        -------
        list of FiberObject — the newly attached nodes.
        """
        from .tme_models.fiber_model import FiberObject
        from ..fiber_analysis.config.extraction_params import FiberProperties

        ps = pixel_size if pixel_size is not None else getattr(result, 'pixel_size', 1.0)
        attached: List[TMEObject] = []

        for fp in result.fibers:
            if isinstance(fp, FiberObject):
                fiber_obj = fp
            else:
                fid = getattr(fp, 'fiber_id', len(attached))
                fiber_obj = FiberObject.from_fiber_properties(
                    fp,
                    object_id=f"{image_id}_fiber_{fid}",
                )
            parent.add_child(fiber_obj)
            self._index.add(fiber_obj)
            attached.append(fiber_obj)

        return attached

    def attach_cell_result(
        self,
        result: Any,
        parent: TMEObject,
        image_id: str = "image",
    ) -> List[TMEObject]:
        """
        Convert a SegmentationResult to CellObject hierarchy nodes and
        attach them all under *parent* in one call.

        Parameters
        ----------
        result : SegmentationResult
            Returned by CellAnalyzer.segment_cells_2d / segment_cells_3d.
            ``result.cells`` may contain CellProperties or CellObject
            instances — both are handled.
        parent : TMEObject
            Node to attach cells to (e.g. a TumorRegion or ImageEntry).
        image_id : str
            Prefix used when generating object_ids for new CellObjects.

        Returns
        -------
        list of CellObject — the newly attached nodes.
        """
        from .tme_models.cell_model import CellObject, CellProperties

        attached: List[TMEObject] = []

        for cp in result.cells:
            if isinstance(cp, CellObject):
                cell_obj = cp
            else:
                cid = getattr(cp, 'cell_id', len(attached))
                cell_obj = CellObject.from_cell_properties(
                    cp,
                    object_id=f"{image_id}_cell_{cid}",
                )
            parent.add_child(cell_obj)
            self._index.add(cell_obj)
            attached.append(cell_obj)

        return attached

    # ------------------------------------------------------------------
    # Validation and export (unchanged)
    # ------------------------------------------------------------------

    def find_object(self, object_id: str) -> Optional[TMEObject]:
        """Alias for get_object — kept for backward compatibility."""
        return self.get_object(object_id)

    def validate_hierarchy(self) -> List[str]:
        """Return a list of consistency issues found in the hierarchy."""
        issues: List[str] = []
        self._validate_node(self.root, issues, set())
        return issues

    def _validate_node(
        self, node: TMEObject, issues: List[str], visited: set
    ) -> None:
        if node.object_id in visited:
            issues.append(f"Circular reference detected: {node.object_id}")
            return
        visited.add(node.object_id)
        for child in node.children:
            if child.parent is not node:
                issues.append(
                    f"Parent-child inconsistency: child={child.object_id}"
                )
            self._validate_node(child, issues, visited.copy())

    def get_spatial_hierarchy(self) -> Dict[str, Any]:
        """Return a nested dict representation of the full tree."""
        return self._build_spatial_tree(self.root)

    def _build_spatial_tree(self, node: TMEObject) -> Dict[str, Any]:
        geometry = getattr(node, "geometry", None)
        bounds = None
        if geometry is not None and getattr(geometry, 'bounds', None) is not None:
            raw = geometry.bounds
            bounds = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return {
            "id": node.object_id,
            "name": node.name,
            "type": node.tme_type.value,
            "bounds": bounds,
            "children": [self._build_spatial_tree(c) for c in node.children],
        }

    def export_to_qupath(self) -> Dict[str, Any]:
        """Export hierarchy to a QuPath-compatible nested dict."""
        return self._convert_to_qupath_object(self.root)

    def _convert_to_qupath_object(self, node: TMEObject) -> Dict[str, Any]:
        return {
            "id": node.object_id,
            "name": node.name,
            "type": node.tme_type.value,
            "properties": node.properties,
            "geometry": self._convert_geometry(node),
            "measurements": self._convert_measurements(node),
            "children": [
                self._convert_to_qupath_object(c) for c in node.children
            ],
        }

    def _convert_geometry(self, node: TMEObject) -> Optional[Dict[str, Any]]:
        geom = getattr(node, "geometry", None)
        if geom is None:
            return None
        coords = geom.coordinates
        return {
            "type": geom.type.value,
            "coordinates": (
                coords.tolist() if isinstance(coords, np.ndarray) else coords
            ),
            "bounds": (
                (list(geom.bounds) if not hasattr(geom.bounds, "tolist")
                 else geom.bounds.tolist()) if geom.bounds is not None else None
            ),
            "is_3d": geom.is_3d,
        }

    def _convert_measurements(self, node: TMEObject) -> List[Dict[str, Any]]:
        measurements = getattr(node, "measurements", None)
        if not measurements:
            return []
        return [m.to_dict() for m in measurements]

    def __repr__(self) -> str:
        total = len(self.root.get_descendants(include_self=True))
        return (
            f"TMEHierarchy(root={self.root.object_id!r}, "
            f"total_objects={total}, "
            f"index_size={len(self._index.by_id)})"
        )