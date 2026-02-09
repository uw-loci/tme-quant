"""
Complete fiber data models for TME analysis.

This module contains all fiber-related data models including:
- Enums for analysis modes
- Parameter classes for orientation and extraction
- Result classes for analysis outputs
- FiberObject for hierarchy integration
- Helper classes for region-level analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from shapely.geometry import LineString, Point, Polygon

# Import base models (assumes these exist in your project)
try:
    from .base_models import TMEObject, ObjectType
    from ..geometry import BoundingBox
except ImportError:
    # Fallback for standalone testing
    class TMEObject:
        pass
    class ObjectType(Enum):
        FIBER = "fiber"
        ORIENTATION_MAP = "orientation_map"
        FIBER_POPULATION = "fiber_population"
    class BoundingBox:
        pass


# ============================================================
# ENUMS FOR ANALYSIS MODES
# ============================================================

class OrientationMode(Enum):
    """Fiber orientation analysis modes."""
    CURVEALIGN = "curvealign"          # Curvelet-based (2D/3D)
    ORIENTATIONJ = "orientationj"      # OrientationJ plugin (2D only)
    PIXEL_WISE = "pixel_wise"          # Pixel-wise gradient (2D)
    VOXEL_WISE = "voxel_wise"          # Voxel-wise gradient (3D)
    STRUCTURE_TENSOR = "structure_tensor"  # Structure tensor method


class ExtractionMode(Enum):
    """Individual fiber extraction modes."""
    CTFIRE = "ctfire"                  # CT-FIRE curvelet-based
    RIDGE_DETECTION = "ridge_detection"  # Ridge Detection plugin (2D)
    FIBER_TRACING = "fiber_tracing"    # Fiber tracing algorithm
    SKELETON = "skeleton"              # Skeletonization-based


# ============================================================
# PARAMETER DATA CLASSES
# ============================================================

@dataclass
class OrientationParams:
    """Parameters for fiber orientation analysis."""
    # Common parameters
    mode: OrientationMode
    pixel_size: float = 1.0  # microns per pixel
    
    # CurveAlign-specific
    curvelet_levels: int = 5
    curvelet_angles: int = 16
    window_size: int = 128  # pixels
    overlap: float = 0.5    # fraction
    
    # OrientationJ-specific
    gradient_method: str = "cubic_spline"  # or "finite_difference"
    coherency_threshold: float = 0.1
    
    # Pixel/Voxel-wise
    smoothing_sigma: float = 2.0
    
    # Output options
    compute_coherency: bool = True
    compute_energy: bool = True
    compute_statistics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'pixel_size': self.pixel_size,
            'curvelet_levels': self.curvelet_levels,
            'curvelet_angles': self.curvelet_angles,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'gradient_method': self.gradient_method,
            'coherency_threshold': self.coherency_threshold,
            'smoothing_sigma': self.smoothing_sigma,
            'compute_coherency': self.compute_coherency,
            'compute_energy': self.compute_energy,
            'compute_statistics': self.compute_statistics,
        }


@dataclass
class ExtractionParams:
    """Parameters for individual fiber extraction."""
    # Common parameters
    mode: ExtractionMode
    pixel_size: float = 1.0  # microns per pixel
    min_fiber_length: float = 10.0  # microns
    
    # CT-FIRE specific
    ctfire_threshold: float = 0.1
    fiber_width_range: Tuple[float, float] = (1.0, 20.0)  # microns
    straightness_threshold: float = 0.8
    
    # Ridge Detection specific
    ridge_sigma: float = 2.0
    lower_threshold: float = 0.1
    upper_threshold: float = 0.3
    
    # Measurement options
    measure_length: bool = True
    measure_width: bool = True
    measure_straightness: bool = True
    measure_angle: bool = True
    measure_curvature: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'pixel_size': self.pixel_size,
            'min_fiber_length': self.min_fiber_length,
            'ctfire_threshold': self.ctfire_threshold,
            'fiber_width_range': self.fiber_width_range,
            'straightness_threshold': self.straightness_threshold,
            'ridge_sigma': self.ridge_sigma,
            'lower_threshold': self.lower_threshold,
            'upper_threshold': self.upper_threshold,
            'measure_length': self.measure_length,
            'measure_width': self.measure_width,
            'measure_straightness': self.measure_straightness,
            'measure_angle': self.measure_angle,
            'measure_curvature': self.measure_curvature,
        }


# ============================================================
# RESULT DATA CLASSES
# ============================================================

@dataclass
class OrientationResult:
    """Results from fiber orientation analysis."""
    mode: OrientationMode
    dimension: str  # "2D" or "3D"
    
    # Orientation maps
    orientation_map: np.ndarray  # angles in degrees
    coherency_map: Optional[np.ndarray] = None
    energy_map: Optional[np.ndarray] = None
    
    # Statistics
    mean_orientation: Optional[float] = None
    orientation_distribution: Optional[np.ndarray] = None
    alignment_score: Optional[float] = None  # 0-1
    
    # Metadata
    pixel_size: float = 1.0
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'mode': self.mode.value,
            'dimension': self.dimension,
            'mean_orientation': self.mean_orientation,
            'alignment_score': self.alignment_score,
            'pixel_size': self.pixel_size,
            'processing_time': self.processing_time,
            'parameters': self.parameters,
        }


@dataclass
class FiberProperties:
    """Properties of a single extracted fiber."""
    fiber_id: int
    
    # Geometric properties
    length: float  # microns
    width: float  # microns
    straightness: float  # 0-1
    angle: float  # degrees
    curvature: float  # 1/microns
    
    # Coordinates
    centerline: np.ndarray  # Nx2 or Nx3 array
    boundary: Optional[np.ndarray] = None
    
    # Derived properties
    aspect_ratio: Optional[float] = None
    tortuosity: Optional[float] = None
    
    # Metadata
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fiber_id': self.fiber_id,
            'length': self.length,
            'width': self.width,
            'straightness': self.straightness,
            'angle': self.angle,
            'curvature': self.curvature,
            'aspect_ratio': self.aspect_ratio,
            'tortuosity': self.tortuosity,
            'confidence': self.confidence,
            'centerline': self.centerline.tolist() if self.centerline is not None else None,
        }


@dataclass
class ExtractionResult:
    """Results from individual fiber extraction."""
    mode: ExtractionMode
    dimension: str  # "2D" or "3D"
    
    # Extracted fibers
    fibers: List[FiberProperties] = field(default_factory=list)
    
    # Summary statistics
    total_fiber_count: int = 0
    mean_fiber_length: float = 0.0
    mean_fiber_width: float = 0.0
    mean_straightness: float = 0.0
    
    # Visualization data
    fiber_mask: Optional[np.ndarray] = None
    labeled_fibers: Optional[np.ndarray] = None
    
    # Metadata
    pixel_size: float = 1.0
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'mode': self.mode.value,
            'dimension': self.dimension,
            'total_fiber_count': self.total_fiber_count,
            'mean_fiber_length': self.mean_fiber_length,
            'mean_fiber_width': self.mean_fiber_width,
            'mean_straightness': self.mean_straightness,
            'pixel_size': self.pixel_size,
            'processing_time': self.processing_time,
            'parameters': self.parameters,
            'fibers': [f.to_dict() for f in self.fibers],
        }


@dataclass
class FiberAnalysisResult:
    """Combined results from fiber analysis."""
    image_id: str
    
    # Analysis results
    orientation_result: Optional[OrientationResult] = None
    extraction_result: Optional[ExtractionResult] = None
    
    # Combined measurements
    measurements: Optional[Dict[str, Any]] = None
    
    # Export paths
    export_paths: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'image_id': self.image_id,
            'measurements': self.measurements,
            'export_paths': self.export_paths,
        }
        
        if self.orientation_result:
            result['orientation'] = self.orientation_result.to_dict()
        
        if self.extraction_result:
            result['extraction'] = self.extraction_result.to_dict()
        
        return result


# ============================================================
# FIBER OBJECT FOR HIERARCHY INTEGRATION
# ============================================================

@dataclass
class FiberObject(TMEObject):
    """
    Individual fiber object in the TME hierarchy.
    
    This is the main fiber class that integrates into the TME project hierarchy.
    It combines geometric properties from fiber extraction with spatial context
    and tumor boundary-relative metrics.
    """
    # Inherited from TMEObject:
    # - object_id: str
    # - object_type: ObjectType (will be set to FIBER)
    # - parent_id: Optional[str]
    # - roi: Optional[ROI]
    # - metadata: Dict[str, Any]
    
    # ============================================================
    # GEOMETRIC PROPERTIES (from fiber extraction)
    # ============================================================
    centerline: np.ndarray = field(default_factory=lambda: np.array([]))  # Nx2 or Nx3 coordinates
    length: float = 0.0  # microns
    width: float = 0.0  # microns
    
    # Orientation properties
    angle: float = 0.0  # degrees (-90 to 90)
    mean_orientation: float = 0.0  # degrees, same as angle for individual fiber
    orientation: Optional[float] = None  # Alternative field name (degrees)
    
    # Shape properties
    straightness: float = 0.0  # 0-1
    curvature: float = 0.0  # 1/microns
    tortuosity: Optional[float] = None
    aspect_ratio: Optional[float] = None
    
    # Fiber boundary (optional)
    boundary: Optional[np.ndarray] = None
    
    # ============================================================
    # TUMOR BOUNDARY-RELATIVE METRICS (for TACS classification)
    # ============================================================
    # Nearest-point method (NEW - more accurate)
    nearest_boundary_point: Optional[np.ndarray] = None  # [x, y] coordinates
    nearest_boundary_distance: Optional[float] = None  # microns
    nearest_boundary_normal_angle: Optional[float] = None  # degrees, normal to boundary
    
    # Relative orientation metrics (KEY for TACS)
    relative_angle_to_boundary_normal: Optional[float] = None  # 0° = perpendicular (TACS-3)
    relative_angle_to_boundary_tangent: Optional[float] = None  # 0° = parallel (TACS-2)
    
    # Global boundary metrics (OLD method - for comparison)
    distance_to_tumor_boundary: Optional[float] = None
    angle_to_tumor_boundary: Optional[float] = None  # Global alignment angle
    
    # ============================================================
    # SPATIAL CONTEXT
    # ============================================================
    in_tumor_core: Optional[bool] = None
    in_tumor_boundary: Optional[bool] = None
    in_stroma: Optional[bool] = None
    at_invasive_front: Optional[bool] = None
    
    # ============================================================
    # TACS CLASSIFICATION
    # ============================================================
    tacs_type: Optional[str] = None  # "TACS-1", "TACS-2", "TACS-3"
    tacs_score: Optional[float] = None  # 0-1
    
    # ============================================================
    # ANALYSIS METADATA
    # ============================================================
    extraction_mode: Optional[str] = None  # "ctfire", "ridge_detection", etc.
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Initialize derived properties and set object type."""
        # Set object type
        if hasattr(self, 'object_type'):
            self.object_type = ObjectType.FIBER
        
        # Compute aspect ratio if not set
        if self.aspect_ratio is None and self.width > 0:
            self.aspect_ratio = self.length / self.width
        
        # Sync orientation fields
        if self.orientation is None and self.angle != 0.0:
            self.orientation = self.angle
        elif self.angle == 0.0 and self.orientation is not None:
            self.angle = self.orientation
    
    # ============================================================
    # GEOMETRY PROPERTIES
    # ============================================================
    
    @property
    def geometry(self) -> LineString:
        """Get fiber as Shapely LineString for spatial operations."""
        if len(self.centerline) < 2:
            raise ValueError("Fiber must have at least 2 points")
        return LineString(self.centerline)
    
    @property
    def start_point(self) -> Point:
        """Get fiber start point."""
        if len(self.centerline) == 0:
            raise ValueError("Fiber has no points")
        return Point(self.centerline[0])
    
    @property
    def end_point(self) -> Point:
        """Get fiber end point."""
        if len(self.centerline) == 0:
            raise ValueError("Fiber has no points")
        return Point(self.centerline[-1])
    
    @property
    def midpoint(self) -> Point:
        """Get fiber midpoint."""
        if len(self.centerline) == 0:
            raise ValueError("Fiber has no points")
        mid_idx = len(self.centerline) // 2
        return Point(self.centerline[mid_idx])
    
    def get_bounding_box(self) -> BoundingBox:
        """Get bounding box of fiber."""
        if len(self.centerline) == 0:
            raise ValueError("Fiber has no points")
        
        min_coords = np.min(self.centerline, axis=0)
        max_coords = np.max(self.centerline, axis=0)
        
        return BoundingBox(
            min_x=min_coords[0],
            max_x=max_coords[0],
            min_y=min_coords[1],
            max_y=max_coords[1]
        )
    
    # ============================================================
    # BOUNDARY-RELATIVE ANALYSIS METHODS
    # ============================================================
    
    def compute_boundary_relative_metrics(
        self,
        tumor_boundary: 'ROI',
        pixel_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute all boundary-relative metrics for this fiber.
        
        This is the main method for calculating fiber orientation relative to
        the nearest point on the tumor boundary (NEW method).
        
        Args:
            tumor_boundary: Tumor boundary ROI
            pixel_size: Pixel size in microns
            
        Returns:
            Dictionary with all computed metrics
        """
        from ..utils.geometry_utils import (
            find_nearest_boundary_point,
            compute_boundary_normal,
            compute_relative_angles
        )
        
        # Get fiber midpoint (most representative point)
        mid_idx = len(self.centerline) // 2
        fiber_point = self.centerline[mid_idx]
        
        # Find nearest point on boundary
        nearest_point, distance = find_nearest_boundary_point(
            fiber_point,
            tumor_boundary,
            pixel_size=pixel_size
        )
        
        self.nearest_boundary_point = nearest_point
        self.nearest_boundary_distance = distance
        
        # Compute boundary normal at nearest point
        boundary_normal_angle = compute_boundary_normal(
            nearest_point,
            tumor_boundary
        )
        self.nearest_boundary_normal_angle = boundary_normal_angle
        
        # Compute fiber orientation
        if len(self.centerline) < 2:
            return {'error': 'Fiber must have at least 2 points'}
        
        fiber_vector = self.centerline[-1] - self.centerline[0]
        fiber_angle = np.degrees(np.arctan2(fiber_vector[1], fiber_vector[0]))
        
        # Compute relative angles
        relative_metrics = compute_relative_angles(
            fiber_angle,
            boundary_normal_angle
        )
        
        self.relative_angle_to_boundary_normal = relative_metrics['angle_to_normal']
        self.relative_angle_to_boundary_tangent = relative_metrics['angle_to_tangent']
        
        # Compute TACS classification
        tacs_result = self._classify_tacs_from_metrics()
        self.tacs_type = tacs_result['type']
        self.tacs_score = tacs_result['score']
        
        return {
            'nearest_point': nearest_point,
            'distance': distance,
            'normal_angle': boundary_normal_angle,
            'fiber_angle': fiber_angle,
            'angle_to_normal': self.relative_angle_to_boundary_normal,
            'angle_to_tangent': self.relative_angle_to_boundary_tangent,
            'tacs_type': self.tacs_type,
            'tacs_score': self.tacs_score
        }
    
    def compute_alignment_to_boundary(
        self,
        boundary: 'ROI'
    ) -> Dict[str, float]:
        """
        Compute fiber alignment relative to a tumor boundary (GLOBAL method).
        
        This is the older method for comparison. Use compute_boundary_relative_metrics
        for more accurate results.
        
        Returns:
            dict with:
                - distance_to_boundary: shortest distance
                - angle_to_boundary: alignment angle
                - alignment_score: 0-1 (1 = parallel, 0 = perpendicular)
        """
        from ..utils.geometry_utils import compute_fiber_to_boundary_alignment
        
        alignment = compute_fiber_to_boundary_alignment(
            self.centerline,
            boundary
        )
        
        # Store for later use
        self.distance_to_tumor_boundary = alignment['distance']
        self.angle_to_tumor_boundary = alignment.get('angle', 0.0)
        
        return alignment
    
    def _classify_tacs_from_metrics(self) -> Dict[str, Any]:
        """
        Classify TACS type based on fiber metrics.
        
        TACS-1: Random, curly fibers (high curvature, random orientation)
        TACS-2: Straightened, parallel fibers (low curvature, parallel to boundary)
        TACS-3: Perpendicular invasion fibers (low curvature, perpendicular to boundary)
        
        Returns:
            Dictionary with 'type' and 'score'
        """
        if self.relative_angle_to_boundary_normal is None:
            return {'type': None, 'score': 0.0}
        
        angle_to_normal = abs(self.relative_angle_to_boundary_normal)
        straightness = self.straightness if self.straightness is not None else 0.5
        
        # TACS-3: Perpendicular to boundary (angle to normal close to 0°)
        if angle_to_normal < 30 and straightness > 0.7:
            return {
                'type': 'TACS-3',
                'score': (1 - angle_to_normal / 30) * straightness
            }
        
        # TACS-2: Parallel to boundary (angle to normal close to 90°)
        elif angle_to_normal > 60 and straightness > 0.7:
            return {
                'type': 'TACS-2',
                'score': ((angle_to_normal - 60) / 30) * straightness
            }
        
        # TACS-1: Random orientation or curly
        else:
            curvature_score = 1 - straightness  # Higher for curly fibers
            randomness_score = 1 - abs(angle_to_normal - 45) / 45  # Closer to 45° = more random
            return {
                'type': 'TACS-1',
                'score': 0.6 * curvature_score + 0.4 * randomness_score
            }
    
    def is_perpendicular_to_boundary(self, threshold: float = 30.0) -> bool:
        """Check if fiber is perpendicular to boundary (TACS-3)."""
        if self.relative_angle_to_boundary_normal is None:
            return False
        return abs(self.relative_angle_to_boundary_normal) < threshold
    
    def is_parallel_to_boundary(self, threshold: float = 30.0) -> bool:
        """Check if fiber is parallel to boundary (TACS-2)."""
        if self.relative_angle_to_boundary_tangent is None:
            return False
        return abs(self.relative_angle_to_boundary_tangent) < threshold
    
    # ============================================================
    # CONVERSION METHODS
    # ============================================================
    
    def to_fiber_properties(self) -> FiberProperties:
        """Convert FiberObject to FiberProperties for export."""
        return FiberProperties(
            fiber_id=int(self.object_id.split('_')[-1]) if '_' in str(self.object_id) else 0,
            length=self.length,
            width=self.width,
            straightness=self.straightness,
            angle=self.angle,
            curvature=self.curvature,
            centerline=self.centerline,
            boundary=self.boundary,
            aspect_ratio=self.aspect_ratio,
            tortuosity=self.tortuosity,
            confidence=self.confidence
        )
    
    @classmethod
    def from_fiber_properties(
        cls,
        props: FiberProperties,
        object_id: str,
        parent_id: Optional[str] = None
    ) -> 'FiberObject':
        """Create FiberObject from FiberProperties."""
        return cls(
            object_id=object_id,
            parent_id=parent_id,
            centerline=props.centerline,
            length=props.length,
            width=props.width,
            angle=props.angle,
            straightness=props.straightness,
            curvature=props.curvature,
            boundary=props.boundary,
            aspect_ratio=props.aspect_ratio,
            tortuosity=props.tortuosity,
            confidence=props.confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'object_id': self.object_id,
            'parent_id': self.parent_id,
            'length': self.length,
            'width': self.width,
            'angle': self.angle,
            'straightness': self.straightness,
            'curvature': self.curvature,
            'aspect_ratio': self.aspect_ratio,
            'nearest_boundary_distance': self.nearest_boundary_distance,
            'relative_angle_to_boundary_normal': self.relative_angle_to_boundary_normal,
            'relative_angle_to_boundary_tangent': self.relative_angle_to_boundary_tangent,
            'tacs_type': self.tacs_type,
            'tacs_score': self.tacs_score,
            'in_tumor_boundary': self.in_tumor_boundary,
            'in_tumor_core': self.in_tumor_core,
            'centerline': self.centerline.tolist() if self.centerline is not None else None,
        }


# ============================================================
# REGION-LEVEL ORIENTATION MAP
# ============================================================

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
    orientation_result: OrientationResult = None
    
    # Region context
    region_type: str = ""  # "tumor_boundary", "tumor_core", "stroma", etc.
    roi: Optional['ROI'] = None  # The region this orientation map corresponds to
    
    # Spatial binning (if computed)
    spatial_bins: Optional[Dict[str, OrientationResult]] = None
    
    def __post_init__(self):
        """Set object type."""
        if hasattr(self, 'object_type'):
            self.object_type = ObjectType.ORIENTATION_MAP
    
    def get_dominant_orientation(self) -> float:
        """Get the dominant orientation angle in this region."""
        if self.orientation_result is None:
            return 0.0
        return self.orientation_result.mean_orientation or 0.0
    
    def get_alignment_score(self) -> float:
        """Get the fiber alignment score in this region."""
        if self.orientation_result is None:
            return 0.0
        return self.orientation_result.alignment_score or 0.0


# ============================================================
# FIBER POPULATION
# ============================================================

@dataclass
class FiberPopulation:
    """
    Collection of fibers with summary statistics.
    
    Used to group fibers by region or characteristics.
    """
    fiber_ids: List[str] = field(default_factory=list)
    region_id: str = ""
    region_type: str = ""  # "tumor_boundary", "tumor_core", "stroma"
    
    # Population statistics
    count: int = 0
    mean_length: float = 0.0
    mean_width: float = 0.0
    mean_straightness: float = 0.0
    mean_orientation: float = 0.0
    alignment_score: float = 0.0
    
    # Distribution statistics
    length_distribution: Optional[np.ndarray] = None
    orientation_distribution: Optional[np.ndarray] = None
    
    # TACS classification (if applicable)
    tacs_type: Optional[str] = None  # "TACS-1", "TACS-2", "TACS-3"
    tacs_score: Optional[float] = None
    tacs_type_distribution: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fiber_ids': self.fiber_ids,
            'region_id': self.region_id,
            'region_type': self.region_type,
            'count': self.count,
            'mean_length': self.mean_length,
            'mean_width': self.mean_width,
            'mean_straightness': self.mean_straightness,
            'mean_orientation': self.mean_orientation,
            'alignment_score': self.alignment_score,
            'tacs_type': self.tacs_type,
            'tacs_score': self.tacs_score,
            'tacs_type_distribution': self.tacs_type_distribution,
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_fiber_object_from_extraction(
    fiber_props: FiberProperties,
    parent_id: str,
    region_type: str = "unknown",
    pixel_size: float = 1.0
) -> FiberObject:
    """
    Create a FiberObject from FiberProperties extraction result.
    
    Args:
        fiber_props: Fiber properties from extraction
        parent_id: ID of parent region
        region_type: Type of region (tumor_boundary, tumor_core, stroma)
        pixel_size: Pixel size in microns
        
    Returns:
        FiberObject ready for hierarchy integration
    """
    object_id = f"{parent_id}_fiber_{fiber_props.fiber_id}"
    
    fiber_obj = FiberObject(
        object_id=object_id,
        parent_id=parent_id,
        centerline=fiber_props.centerline,
        length=fiber_props.length,
        width=fiber_props.width,
        angle=fiber_props.angle,
        mean_orientation=fiber_props.angle,
        straightness=fiber_props.straightness,
        curvature=fiber_props.curvature,
        boundary=fiber_props.boundary,
        aspect_ratio=fiber_props.aspect_ratio,
        tortuosity=fiber_props.tortuosity,
        confidence=fiber_props.confidence,
        metadata={'region_type': region_type}
    )
    
    return fiber_obj


def aggregate_fiber_population(
    fibers: List[FiberObject],
    region_id: str,
    region_type: str
) -> FiberPopulation:
    """
    Create FiberPopulation from a list of fibers.
    
    Args:
        fibers: List of FiberObject instances
        region_id: ID of the region
        region_type: Type of region
        
    Returns:
        FiberPopulation with aggregated statistics
    """
    if not fibers:
        return FiberPopulation(region_id=region_id, region_type=region_type)
    
    # Collect fiber IDs
    fiber_ids = [f.object_id for f in fibers]
    
    # Compute statistics
    lengths = [f.length for f in fibers]
    widths = [f.width for f in fibers]
    straightnesses = [f.straightness for f in fibers]
    orientations = [f.angle for f in fibers]
    
    # Compute alignment score (order parameter)
    angles_rad = np.deg2rad(orientations)
    mean_cos = np.mean(np.cos(2 * angles_rad))
    mean_sin = np.mean(np.sin(2 * angles_rad))
    alignment_score = np.sqrt(mean_cos**2 + mean_sin**2)
    mean_orientation = np.rad2deg(np.arctan2(mean_sin, mean_cos) / 2)
    
    # TACS distribution
    tacs_types = [f.tacs_type for f in fibers if f.tacs_type is not None]
    tacs_scores = [f.tacs_score for f in fibers if f.tacs_score is not None]
    
    tacs_type_dist = None
    dominant_tacs = None
    mean_tacs_score = None
    
    if tacs_types:
        from collections import Counter
        tacs_counts = Counter(tacs_types)
        tacs_type_dist = dict(tacs_counts)
        dominant_tacs = tacs_counts.most_common(1)[0][0]
    
    if tacs_scores:
        mean_tacs_score = float(np.mean(tacs_scores))
    
    return FiberPopulation(
        fiber_ids=fiber_ids,
        region_id=region_id,
        region_type=region_type,
        count=len(fibers),
        mean_length=float(np.mean(lengths)),
        mean_width=float(np.mean(widths)),
        mean_straightness=float(np.mean(straightnesses)),
        mean_orientation=float(mean_orientation),
        alignment_score=float(alignment_score),
        length_distribution=np.histogram(lengths, bins=20)[0],
        orientation_distribution=np.histogram(orientations, bins=36, range=(-90, 90))[0],
        tacs_type=dominant_tacs,
        tacs_score=mean_tacs_score,
        tacs_type_distribution=tacs_type_dist
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Enums
    'OrientationMode',
    'ExtractionMode',
    
    # Parameters
    'OrientationParams',
    'ExtractionParams',
    
    # Results
    'OrientationResult',
    'FiberProperties',
    'ExtractionResult',
    'FiberAnalysisResult',
    
    # Hierarchy objects
    'FiberObject',
    'RegionOrientationMap',
    'FiberPopulation',
    
    # Helper functions
    'create_fiber_object_from_extraction',
    'aggregate_fiber_population',
]