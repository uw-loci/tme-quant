"""
Hierarchy management for TME objects.

TMEHierarchy wraps the root TMEObject and provides a manager-level API
for adding, retrieving, and querying objects across the full tree.
The underlying traversal logic lives on TMEObject itself (Issue 1);
this class is a thin convenience layer used by TMEProject.
"""
from typing import List, Dict, Optional, Any, Union
import numpy as np

from .base_models import TMEObject, TMEType, ObjectType


class TMEHierarchy:
    """
    Manager for a TMEObject hierarchy rooted at a single node.

    Provides the API expected by TMEProject:
      - get_object(object_id)        -> Optional[TMEObject]
      - add_object(obj, parent=...)  -> None
      - get_objects_by_type(...)     -> List[TMEObject]
      - get_descendants(object_id)   -> List[TMEObject]
      - get_children(object_id)      -> List[TMEObject]
    """

    def __init__(self, root: Optional[TMEObject] = None) -> None:
        # Allow construction without a root (project creates root later)
        if root is None:
            root = TMEObject(object_id="root", name="root", tme_type=TMEType.PROJECT)
        self.root: TMEObject = root

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_object(self, object_id: str) -> Optional[TMEObject]:
        """Find and return an object anywhere in the hierarchy by its id."""
        return self.root.find_by_id(object_id)

    def get_objects_by_type(
        self,
        object_type: Union[TMEType, ObjectType],
    ) -> List[TMEObject]:
        """
        Return all objects with the given type.

        Accepts either a TMEType or an ObjectType value so that existing
        project.py calls using ObjectType.FIBER continue to work.
        """
        if isinstance(object_type, TMEType):
            return self.root.filter_by_type(tme_type=object_type)
        else:
            return self.root.filter_by_type(object_type=object_type)

    def get_descendants(self, object_id: str) -> List[TMEObject]:
        """Return all descendants of the object with the given id."""
        node = self.get_object(object_id)
        if node is None:
            return []
        return node.get_descendants()

    def get_children(self, object_id: str) -> List[TMEObject]:
        """Return direct children of the object with the given id."""
        node = self.get_object(object_id)
        if node is None:
            return []
        return list(node.children)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_object(
        self,
        obj: TMEObject,
        parent: Optional[Union[TMEObject, str]] = None,
    ) -> None:
        """
        Add *obj* to the hierarchy.

        Parameters
        ----------
        obj:
            The TMEObject to insert.
        parent:
            Either a TMEObject instance or an object_id string.
            If None the object is attached to the root.
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

    def remove_object(self, object_id: str) -> bool:
        """
        Remove the object with the given id from its parent.

        Returns True if the object was found and removed, False otherwise.
        """
        node = self.get_object(object_id)
        if node is None:
            return False
        node.detach()
        return True

    # ------------------------------------------------------------------
    # Validation and export
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
        if geometry is not None and hasattr(geometry, "bounds") and geometry.bounds is not None:
            raw_bounds = geometry.bounds
            bounds = raw_bounds.tolist() if hasattr(raw_bounds, "tolist") else list(raw_bounds)
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
                geom.bounds.tolist() if geom.bounds is not None else None
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
        return f"TMEHierarchy(root={self.root.object_id!r}, total_objects={total})"