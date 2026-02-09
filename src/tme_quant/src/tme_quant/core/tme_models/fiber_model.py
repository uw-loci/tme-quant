from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from shapely.geometry import LineString, Point

from .base_models import TMEObject
from ..geometry import BoundingBox


@dataclass
class FiberObject(TMEObject):
    """
    Individual fiber object in the TME hierarchy.
    
    Inherits from TMEObject to be part of the hierarchy.
    Contains both geometric and analysis properties.
    """
    # Inherited from TMEObject:
    # - object_id: str
    # - object_type: ObjectType (FIBER)
    # - parent_id: Optional[str]
    # - roi: Optional[ROI]
    # - metadata: Dict[str, Any]
    
    # Geometric properties (from fiber extraction)
    centerline: np.ndarray  # Nx2 or Nx3 coordinates
    length: float  # microns
    width: float  # microns
    
    # Orientation properties
    angle: float  # degrees (-90 to 90)
    mean_orientation: float  # degrees, same as angle for individual fiber
    
    # Shape properties
    straightness: float  # 0-1
    curvature: float  # 1/microns
    tortuosity: Optional[float] = None
    
    # Spatial relationships (computed after extraction)
    distance_to_tumor_boundary: Optional[float] = None
    angle_to_tumor_boundary: Optional[float] = None  # Alignment angle
    in_tumor_core: Optional[bool] = None
    in_tumor_boundary: Optional[bool] = None
    in_stroma: Optional[bool] = None
    
    # Derived properties
    aspect_ratio: Optional[float] = None
    
    # Analysis metadata
    extraction_mode: Optional[str] = None  # "ctfire", "ridge_detection", etc.
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Initialize derived properties."""
        if self.aspect_ratio is None and self.width > 0:
            self.aspect_ratio = self.length / self.width
        
        # Set object type
        self.object_type = ObjectType.FIBER
    
    @property
    def geometry(self) -> LineString:
        """Get fiber as Shapely LineString for spatial operations."""
        return LineString(self.centerline)
    
    @property
    def start_point(self) -> Point:
        """Get fiber start point."""
        return Point(self.centerline[0])
    
    @property
    def end_point(self) -> Point:
        """Get fiber end point."""
        return Point(self.centerline[-1])
    
    @property
    def midpoint(self) -> Point:
        """Get fiber midpoint."""
        mid_idx = len(self.centerline) // 2
        return Point(self.centerline[mid_idx])
    
    def get_bounding_box(self) -> BoundingBox:
        """Get bounding box of fiber."""
        min_coords = np.min(self.centerline, axis=0)
        max_coords = np.max(self.centerline, axis=0)
        return BoundingBox(
            min_x=min_coords[0],
            max_x=max_coords[0],
            min_y=min_coords[1],
            max_y=max_coords[1]
        )
    
    def compute_alignment_to_boundary(
        self,
        boundary: 'ROI'
    ) -> Dict[str, float]:
        """
        Compute fiber alignment relative to a tumor boundary.
        
        Returns:
            dict with:
                - distance_to_boundary: shortest distance
                - angle_to_boundary: alignment angle (0° = parallel, 90° = perpendicular)
                - alignment_score: 0-1 (1 = parallel, 0 = perpendicular)
        """
        from ..geometry import compute_fiber_to_boundary_alignment
        
        alignment = compute_fiber_to_boundary_alignment(
            self.centerline,
            boundary
        )
        
        # Store for later use
        self.distance_to_tumor_boundary = alignment['distance']
        self.angle_to_tumor_boundary = alignment['angle']
        
        return alignment


@dataclass
class RegionOrientationMap(TMEObject):
    """
    Region-level orientation analysis results.
    
    Stores orientation maps and statistics for a specific region
    (e.g., tumor boundary, tumor core, stroma).
    """
    # Inherited from TMEObject
    # - object_id: str
    # - parent_id: Optional[str]  (links to TumorRegion or StromaRegion)
    
    # Orientation analysis results
    orientation_result: 'OrientationResult'
    
    # Region context
    region_type: str  # "tumor_boundary", "tumor_core", "stroma", etc.
    roi: 'ROI'  # The region this orientation map corresponds to
    
    # Spatial binning (if computed)
    spatial_bins: Optional[Dict[str, 'OrientationResult']] = None
    
    def __post_init__(self):
        self.object_type = ObjectType.ORIENTATION_MAP
    
    def get_dominant_orientation(self) -> float:
        """Get the dominant orientation angle in this region."""
        return self.orientation_result.mean_orientation
    
    def get_alignment_score(self) -> float:
        """Get the fiber alignment score in this region."""
        return self.orientation_result.alignment_score


@dataclass
class FiberPopulation:
    """
    Collection of fibers with summary statistics.
    
    Used to group fibers by region or characteristics.
    """
    fiber_ids: List[str]
    region_id: str
    region_type: str  # "tumor_boundary", "tumor_core", "stroma"
    
    # Population statistics
    count: int
    mean_length: float
    mean_width: float
    mean_straightness: float
    mean_orientation: float
    alignment_score: float
    
    # Distribution statistics
    length_distribution: Optional[np.ndarray] = None
    orientation_distribution: Optional[np.ndarray] = None
    
    # TACS classification (if applicable)
    tacs_type: Optional[str] = None  # "TACS-1", "TACS-2", "TACS-3"
    tacs_score: Optional[float] = None


# Update ObjectType enum to include fiber-related types
class ObjectType(Enum):
    """Types of objects in TME hierarchy."""
    # Existing types
    IMAGE = "image"
    TISSUE = "tissue"
    TUMOR = "tumor"
    CELL = "cell"
    VESSEL = "vessel"
    STROMA = "stroma"
    
    # NEW: Fiber-related types
    FIBER = "fiber"
    ORIENTATION_MAP = "orientation_map"
    FIBER_POPULATION = "fiber_population"