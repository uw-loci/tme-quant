"""
Hierarchy management for TME objects
"""
from typing import List, Dict, Optional, Any
from .base_models import TMEObject, TMEType

class TMEHierarchy:
    """Manages hierarchical organization of TME objects"""
    
    def __init__(self, root: TMEObject):
        self.root = root
    
    def find_object(self, object_id: str) -> Optional[TMEObject]:
        """Find object by ID in hierarchy"""
        return self.root.find_by_id(object_id)
    
    def get_objects_by_type(self, object_type: TMEType) -> List[TMEObject]:
        """Get all objects of specified type"""
        return self.root.filter_by_type(object_type)
    
    def get_spatial_hierarchy(self) -> Dict[str, Any]:
        """Get spatial hierarchy representation"""
        return self._build_spatial_tree(self.root)
    
    def _build_spatial_tree(self, node: TMEObject) -> Dict[str, Any]:
        """Recursively build spatial tree"""
        tree = {
            'id': node.id,
            'name': node.name,
            'type': node.type.value,
            'bounds': node.geometry.bounds.tolist() if hasattr(node, 'geometry') and node.geometry.bounds is not None else None,
            'children': [self._build_spatial_tree(child) for child in node.children]
        }
        return tree
    
    def validate_hierarchy(self) -> List[str]:
        """Validate hierarchy consistency"""
        issues = []
        self._validate_node(self.root, issues, set())
        return issues
    
    def _validate_node(self, node: TMEObject, issues: List[str], visited: set) -> None:
        """Validate node and children recursively"""
        if node.id in visited:
            issues.append(f"Circular reference detected: {node.id}")
            return
        
        visited.add(node.id)
        
        # Check parent-child consistency
        for child in node.children:
            if child.parent != node:
                issues.append(f"Parent-child inconsistency: {child.id}")
            
            self._validate_node(child, issues, visited.copy())
    
    def export_to_qupath(self) -> Dict[str, Any]:
        """Export hierarchy to QuPath-compatible format"""
        return self._convert_to_qupath_object(self.root)
    
    def _convert_to_qupath_object(self, node: TMEObject) -> Dict[str, Any]:
        """Convert TMEObject to QuPath-like object"""
        qupath_obj = {
            'id': node.id,
            'name': node.name,
            'type': node.type.value,
            'properties': node.properties,
            'geometry': self._convert_geometry(node) if hasattr(node, 'geometry') else None,
            'measurements': self._convert_measurements(node) if hasattr(node, 'measurements') else [],
            'children': [self._convert_to_qupath_object(child) for child in node.children]
        }
        return qupath_obj
    
    def _convert_geometry(self, node: TMEObject) -> Optional[Dict[str, Any]]:
        """Convert geometry to QuPath format"""
        if not hasattr(node, 'geometry'):
            return None
        
        geom = node.geometry
        return {
            'type': geom.type.value,
            'coordinates': geom.coordinates.tolist() if isinstance(geom.coordinates, np.ndarray) else geom.coordinates,
            'bounds': geom.bounds.tolist() if geom.bounds is not None else None,
            'is_3d': geom.is_3d
        }
    
    def _convert_measurements(self, node: TMEObject) -> List[Dict[str, Any]]:
        """Convert measurements to QuPath format"""
        if not hasattr(node, 'measurements'):
            return []
        
        return [measurement.to_dict() for measurement in node.measurements]