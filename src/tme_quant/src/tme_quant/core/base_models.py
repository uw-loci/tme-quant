""" 
Base models for TME objects with QuPath-like hierarchy support
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum, auto
import numpy as np
from datetime import datetime
import uuid

class TMEType(Enum):
    """Types of TME objects following QuPath hierarchy"""
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

class GeometryType(Enum):
    """Supported geometry types"""
    POINT = "point"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    LINE = "line"
    MULTIPOINT = "multipoint"
    PATH = "path"
    CUBOID = "cuboid"  # 3D
    MESH = "mesh"      # 3D

@dataclass
class TMEObject:
    """Base class for all TME objects with QuPath-like hierarchy"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: TMEType = TMEType.DETECTION
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    parent: Optional['TMEObject'] = None
    children: List['TMEObject'] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize hierarchy relationships"""
        if self.parent and self not in self.parent.children:
            self.parent.add_child(self)
    
    def add_child(self, child: 'TMEObject') -> None:
        """Add child object and set parent reference"""
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            self.modified = datetime.now()
    
    def remove_child(self, child: 'TMEObject') -> None:
        """Remove child object"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self.modified = datetime.now()
    
    def get_ancestors(self) -> List['TMEObject']:
        """Get all ancestors up to root"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self, include_self: bool = False) -> List['TMEObject']:
        """Get all descendants (recursive)"""
        descendants = []
        
        if include_self:
            descendants.append(self)
        
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        
        return descendants
    
    def find_by_id(self, object_id: str) -> Optional['TMEObject']:
        """Find object by ID in hierarchy"""
        if self.id == object_id:
            return self
        
        for child in self.children:
            result = child.find_by_id(object_id)
            if result:
                return result
        
        return None
    
    def filter_by_type(self, object_type: TMEType) -> List['TMEObject']:
        """Filter descendants by type"""
        return [obj for obj in self.get_descendants(include_self=True) 
                if obj.type == object_type]
    
    def to_hierarchy_dict(self) -> Dict[str, Any]:
        """Convert hierarchy to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'metadata': self.metadata,
            'properties': self.properties,
            'children': [child.to_hierarchy_dict() for child in self.children]
        }
    
    def update_properties(self, **kwargs) -> None:
        """Update object properties"""
        self.properties.update(kwargs)
        self.modified = datetime.now()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get property with optional default"""
        return self.properties.get(key, default)

@dataclass
class Geometry:
    """Geometric representation of TME objects"""
    type: GeometryType
    coordinates: Union[np.ndarray, List[Union[np.ndarray, float]]]
    bounds: Optional[np.ndarray] = None  # [x_min, y_min, z_min, x_max, y_max, z_max]
    is_3d: bool = False
    
    def __post_init__(self):
        """Initialize bounds from coordinates"""
        if self.bounds is None:
            self._calculate_bounds()
    
    def _calculate_bounds(self) -> None:
        """Calculate bounding box from coordinates"""
        if isinstance(self.coordinates, list):
            coords = np.array(self.coordinates)
        else:
            coords = self.coordinates
        
        if coords.ndim == 1:
            # Single point
            self.bounds = np.concatenate([coords, coords])
        elif coords.ndim == 2:
            # Multiple points
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            self.bounds = np.concatenate([min_coords, max_coords])
        
        # Check if 3D
        self.is_3d = coords.shape[-1] >= 3 if coords.ndim > 0 else False
    
    def area(self) -> float:
        """Calculate area/volume"""
        if self.bounds is None:
            return 0.0
        
        if not self.is_3d:
            # 2D area
            width = self.bounds[3] - self.bounds[0]
            height = self.bounds[4] - self.bounds[1]
            return width * height
        else:
            # 3D volume
            width = self.bounds[3] - self.bounds[0]
            height = self.bounds[4] - self.bounds[1]
            depth = self.bounds[5] - self.bounds[2]
            return width * height * depth
    
    def centroid(self) -> np.ndarray:
        """Calculate centroid"""
        if self.bounds is None:
            return np.array([0, 0, 0])
        
        if not self.is_3d:
            return np.array([
                (self.bounds[0] + self.bounds[3]) / 2,
                (self.bounds[1] + self.bounds[4]) / 2,
                0
            ])
        else:
            return np.array([
                (self.bounds[0] + self.bounds[3]) / 2,
                (self.bounds[1] + self.bounds[4]) / 2,
                (self.bounds[2] + self.bounds[5]) / 2
            ])

@dataclass
class Measurement:
    """Quantitative measurement for TME objects"""
    name: str
    value: Union[float, int, str]
    unit: str = ""
    description: str = ""
    method: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'description': self.description,
            'method': self.method,
            'confidence': self.confidence
        }

@dataclass
class Classification:
    """Object classification with hierarchy support"""
    name: str
    probability: float = 1.0
    parent_class: Optional['Classification'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        """Get full hierarchical classification name"""
        if self.parent_class:
            return f"{self.parent_class.full_name}:{self.name}"
        return self.name
    
    def is_subclass_of(self, classification: 'Classification') -> bool:
        """Check if this is a subclass of given classification"""
        current = self.parent_class
        while current:
            if current == classification:
                return True
            current = current.parent_class
        return False

@dataclass
class TMEMetadata:
    """Comprehensive metadata for TME analysis"""
    # Image metadata
    image_dimensions: Optional[tuple] = None  # (width, height, depth, channels)
    pixel_size: Optional[tuple] = None  # (x_resolution, y_resolution, z_resolution) in microns
    magnification: Optional[float] = None
    channels: List[str] = field(default_factory=list)
    
    # Sample metadata
    sample_id: str = ""
    patient_id: str = ""
    tissue_type: str = ""
    stain_type: str = ""
    diagnosis: str = ""
    
    # Processing metadata
    preprocessing_steps: List[str] = field(default_factory=list)
    analysis_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Clinical metadata
    clinical_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_dimensions': self.image_dimensions,
            'pixel_size': self.pixel_size,
            'magnification': self.magnification,
            'channels': self.channels,
            'sample_id': self.sample_id,
            'patient_id': self.patient_id,
            'tissue_type': self.tissue_type,
            'stain_type': self.stain_type,
            'diagnosis': self.diagnosis,
            'preprocessing_steps': self.preprocessing_steps,
            'analysis_parameters': self.analysis_parameters,
            'clinical_data': self.clinical_data
        }