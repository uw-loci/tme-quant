"""
Complete cell data models for TME analysis.

This module contains all cell-related data models including:
- Enums for analysis modes
- Parameter classes for segmentation, classification, quantification
- Result classes for analysis outputs
- CellObject for hierarchy integration
- Helper classes for population analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from shapely.geometry import Point, Polygon

from .base_models import TMEObject, ObjectType
from ..geometry import BoundingBox, ROI


# ============================================================
# ENUMS FOR ANALYSIS MODES
# ============================================================

class SegmentationMode(Enum):
    """Cell segmentation modes."""
    STARDIST = "stardist"              # StarDist deep learning
    CELLPOSE = "cellpose"              # Cellpose deep learning
    THRESHOLDING = "thresholding"      # Threshold-based
    WATERSHED = "watershed"            # Watershed segmentation
    OTSU = "otsu"                      # Otsu thresholding
    ADAPTIVE_THRESHOLD = "adaptive"    # Adaptive thresholding
    CLASSICAL = "classical"            # Classical methods


class ImageModality(Enum):
    """Image modality types."""
    FLUORESCENCE = "fluorescence"      # Fluorescence microscopy
    BRIGHTFIELD = "brightfield"        # Bright-field (H&E)
    PHASE_CONTRAST = "phase_contrast"  # Phase contrast
    DIC = "dic"                        # Differential interference contrast
    CONFOCAL = "confocal"              # Confocal microscopy


class CellType(Enum):
    """Cell types in TME."""
    TUMOR = "tumor"                    # Tumor/cancer cells
    IMMUNE = "immune"                  # Immune cells (general)
    T_CELL = "t_cell"                  # T cells
    B_CELL = "b_cell"                  # B cells
    MACROPHAGE = "macrophage"          # Macrophages
    NEUTROPHIL = "neutrophil"          # Neutrophils
    NK_CELL = "nk_cell"                # Natural killer cells
    DENDRITIC = "dendritic"            # Dendritic cells
    FIBROBLAST = "fibroblast"          # Fibroblasts (CAFs)
    ENDOTHELIAL = "endothelial"        # Endothelial cells
    STROMAL = "stromal"                # Stromal cells (general)
    UNKNOWN = "unknown"                # Unclassified


class ClassificationMode(Enum):
    """Cell classification modes."""
    MORPHOLOGY = "morphology"          # Morphology-based
    MARKER = "marker"                  # Marker/IF-based
    DEEP_LEARNING = "deep_learning"    # Deep learning
    ENSEMBLE = "ensemble"              # Ensemble methods
    RULE_BASED = "rule_based"          # Rule-based classification


# ============================================================
# PARAMETER DATA CLASSES
# ============================================================

@dataclass
class SegmentationParams:
    """Parameters for cell segmentation."""
    # Common parameters
    mode: SegmentationMode
    image_modality: ImageModality = ImageModality.FLUORESCENCE
    pixel_size: float = 1.0  # microns per pixel
    target: str = "nucleus"  # "nucleus" or "whole_cell"
    
    # StarDist-specific
    stardist_model: str = "2D_versatile_fluo"  # or "2D_versatile_he"
    stardist_prob_thresh: float = 0.5
    stardist_nms_thresh: float = 0.4
    
    # Cellpose-specific
    cellpose_model: str = "nuclei"  # "nuclei", "cyto", "cyto2"
    cellpose_diameter: Optional[float] = None  # Auto if None
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    
    # Thresholding-specific
    threshold_method: str = "otsu"  # "otsu", "adaptive", "manual"
    threshold_value: Optional[float] = None  # For manual
    adaptive_block_size: int = 35
    
    # Watershed-specific
    watershed_markers: str = "distance"  # "distance", "peaks", "manual"
    watershed_min_distance: int = 10  # pixels
    
    # Post-processing
    min_cell_size: float = 20.0  # square microns
    max_cell_size: float = 500.0  # square microns
    remove_border_cells: bool = True
    fill_holes: bool = True
    
    # Output options
    return_probabilities: bool = False
    return_boundaries: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'image_modality': self.image_modality.value,
            'pixel_size': self.pixel_size,
            'target': self.target,
            'stardist_model': self.stardist_model,
            'cellpose_model': self.cellpose_model,
            'threshold_method': self.threshold_method,
            'min_cell_size': self.min_cell_size,
            'max_cell_size': self.max_cell_size,
        }


@dataclass
class ClassificationParams:
    """Parameters for cell classification."""
    # Common parameters
    mode: ClassificationMode
    cell_types: List[CellType] = field(default_factory=lambda: [
        CellType.TUMOR, CellType.IMMUNE, CellType.STROMAL
    ])
    
    # Morphology-based
    use_area: bool = True
    use_shape: bool = True
    use_texture: bool = True
    use_intensity: bool = True
    
    # Marker-based (for IF images)
    marker_channels: Dict[str, int] = field(default_factory=dict)  # e.g., {'CD3': 2, 'CD8': 3}
    marker_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Deep learning
    classification_model: Optional[str] = None
    
    # Rule-based
    classification_rules: Optional[Dict[str, Any]] = None
    
    # Confidence thresholds
    min_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'cell_types': [ct.value for ct in self.cell_types],
            'marker_channels': self.marker_channels,
            'min_confidence': self.min_confidence,
        }


@dataclass
class QuantificationParams:
    """Parameters for cell quantification."""
    # Morphological measurements
    measure_area: bool = True
    measure_perimeter: bool = True
    measure_circularity: bool = True
    measure_eccentricity: bool = True
    measure_solidity: bool = True
    measure_extent: bool = True
    measure_orientation: bool = True
    
    # Intensity measurements (per channel)
    measure_mean_intensity: bool = True
    measure_integrated_intensity: bool = True
    measure_std_intensity: bool = True
    measure_min_max_intensity: bool = True
    
    # Texture measurements
    measure_texture: bool = False
    texture_scales: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    # Spatial measurements
    measure_centroid: bool = True
    measure_distances: bool = True  # To neighbors
    measure_density: bool = True
    
    # Relationship measurements
    measure_cell_cell_distance: bool = True
    measure_cell_fiber_distance: bool = False
    measure_cell_tumor_distance: bool = False
    
    # Distance thresholds
    neighbor_distance_threshold: float = 50.0  # microns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'measure_area': self.measure_area,
            'measure_perimeter': self.measure_perimeter,
            'measure_texture': self.measure_texture,
            'measure_distances': self.measure_distances,
        }


# ============================================================
# RESULT DATA CLASSES
# ============================================================

@dataclass
class CellProperties:
    """Properties of a single segmented cell."""
    cell_id: int
    
    # Geometric properties
    area: float  # square microns
    perimeter: float  # microns
    centroid: Tuple[float, float]  # (x, y)
    
    # Shape properties
    circularity: float  # 0-1
    eccentricity: float  # 0-1
    solidity: float  # 0-1
    extent: float  # 0-1
    major_axis_length: float  # microns
    minor_axis_length: float  # microns
    orientation: float  # degrees
    
    # Coordinates
    boundary: np.ndarray  # Nx2 array
    mask: Optional[np.ndarray] = None  # Binary mask
    
    # Intensity properties (per channel)
    mean_intensity: Dict[str, float] = field(default_factory=dict)
    integrated_intensity: Dict[str, float] = field(default_factory=dict)
    std_intensity: Dict[str, float] = field(default_factory=dict)
    
    # Classification
    cell_type: Optional[CellType] = None
    cell_type_confidence: Optional[float] = None
    
    # Spatial context
    neighbor_indices: List[int] = field(default_factory=list)
    neighbor_distances: List[float] = field(default_factory=list)
    
    # Metadata
    confidence: Optional[float] = None  # Segmentation confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cell_id': self.cell_id,
            'area': self.area,
            'perimeter': self.perimeter,
            'centroid': self.centroid,
            'circularity': self.circularity,
            'eccentricity': self.eccentricity,
            'cell_type': self.cell_type.value if self.cell_type else None,
            'cell_type_confidence': self.cell_type_confidence,
        }


@dataclass
class SegmentationResult:
    """Results from cell segmentation."""
    mode: SegmentationMode
    dimension: str  # "2D" or "3D"
    image_modality: ImageModality
    
    # Segmented cells
    cells: List[CellProperties] = field(default_factory=list)
    
    # Summary statistics
    total_cell_count: int = 0
    mean_cell_area: float = 0.0
    mean_circularity: float = 0.0
    
    # Segmentation masks
    label_mask: Optional[np.ndarray] = None  # Labeled image
    boundary_mask: Optional[np.ndarray] = None  # Cell boundaries
    probability_map: Optional[np.ndarray] = None  # Probability map (if available)
    
    # Metadata
    pixel_size: float = 1.0
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'mode': self.mode.value,
            'dimension': self.dimension,
            'image_modality': self.image_modality.value,
            'total_cell_count': self.total_cell_count,
            'mean_cell_area': self.mean_cell_area,
            'mean_circularity': self.mean_circularity,
            'pixel_size': self.pixel_size,
            'processing_time': self.processing_time,
        }


@dataclass
class ClassificationResult:
    """Results from cell classification."""
    mode: ClassificationMode
    
    # Cell type assignments
    cell_types: Dict[int, CellType] = field(default_factory=dict)  # cell_id -> type
    confidences: Dict[int, float] = field(default_factory=dict)  # cell_id -> confidence
    
    # Type distribution
    type_counts: Dict[CellType, int] = field(default_factory=dict)
    type_ratios: Dict[CellType, float] = field(default_factory=dict)
    
    # Metadata
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'type_counts': {k.value: v for k, v in self.type_counts.items()},
            'type_ratios': {k.value: v for k, v in self.type_ratios.items()},
            'processing_time': self.processing_time,
        }


@dataclass
class QuantificationResult:
    """Results from cell quantification."""
    # Per-cell measurements
    measurements: Dict[int, Dict[str, float]] = field(default_factory=dict)  # cell_id -> features
    
    # Population statistics
    population_stats: Dict[str, float] = field(default_factory=dict)
    
    # Spatial statistics
    spatial_stats: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    parameters: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'population_stats': self.population_stats,
            'spatial_stats': self.spatial_stats,
            'processing_time': self.processing_time,
        }


@dataclass
class CellAnalysisResult:
    """Combined results from cell analysis."""
    image_id: str
    
    # Analysis results
    segmentation_result: Optional[SegmentationResult] = None
    classification_result: Optional[ClassificationResult] = None
    quantification_result: Optional[QuantificationResult] = None
    
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
        
        if self.segmentation_result:
            result['segmentation'] = self.segmentation_result.to_dict()
        
        if self.classification_result:
            result['classification'] = self.classification_result.to_dict()
        
        if self.quantification_result:
            result['quantification'] = self.quantification_result.to_dict()
        
        return result


# ============================================================
# CELL OBJECT FOR HIERARCHY INTEGRATION
# ============================================================

@dataclass
class CellObject(TMEObject):
    """
    Individual cell object in the TME hierarchy.
    
    Integrates segmentation, classification, and quantification results
    into the TME project hierarchy.
    """
    # Inherited from TMEObject:
    # - object_id: str
    # - object_type: ObjectType (will be set to CELL)
    # - parent_id: Optional[str]
    # - roi: Optional[ROI]
    # - metadata: Dict[str, Any]
    
    # ============================================================
    # GEOMETRIC PROPERTIES (from segmentation)
    # ============================================================
    centroid: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0  # square microns
    perimeter: float = 0.0  # microns
    boundary: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Shape properties
    circularity: float = 0.0  # 0-1
    eccentricity: float = 0.0  # 0-1
    solidity: float = 0.0  # 0-1
    extent: float = 0.0  # 0-1
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    orientation: float = 0.0  # degrees
    
    # ============================================================
    # CLASSIFICATION
    # ============================================================
    cell_type: Optional[CellType] = None
    cell_type_confidence: Optional[float] = None
    
    # Marker expression (for IF images)
    marker_expression: Dict[str, float] = field(default_factory=dict)
    
    # ============================================================
    # INTENSITY MEASUREMENTS
    # ============================================================
    mean_intensity: Dict[str, float] = field(default_factory=dict)  # channel -> value
    integrated_intensity: Dict[str, float] = field(default_factory=dict)
    std_intensity: Dict[str, float] = field(default_factory=dict)
    
    # ============================================================
    # SPATIAL CONTEXT
    # ============================================================
    # Tumor context
    in_tumor_region: Optional[bool] = None
    distance_to_tumor_boundary: Optional[float] = None
    in_invasive_margin: Optional[bool] = None
    
    # Neighbors
    neighbor_cell_ids: List[str] = field(default_factory=list)
    neighbor_distances: List[float] = field(default_factory=list)
    nearest_neighbor_distance: Optional[float] = None
    
    # Fiber interactions
    interacting_fiber_ids: List[str] = field(default_factory=list)
    fiber_distances: List[float] = field(default_factory=list)
    
    # ============================================================
    # ANALYSIS METADATA
    # ============================================================
    segmentation_mode: Optional[str] = None
    segmentation_confidence: Optional[float] = None
    
    def __post_init__(self):
        """Initialize derived properties and set object type."""
        # Set object type
        if hasattr(self, 'object_type'):
            self.object_type = ObjectType.CELL
    
    # ============================================================
    # GEOMETRY PROPERTIES
    # ============================================================
    
    @property
    def geometry(self) -> Polygon:
        """Get cell as Shapely Polygon for spatial operations."""
        if len(self.boundary) < 3:
            # Approximate as circle
            from shapely.geometry import Point
            return Point(self.centroid).buffer(np.sqrt(self.area / np.pi))
        return Polygon(self.boundary)
    
    @property
    def center_point(self) -> Point:
        """Get cell centroid as Point."""
        return Point(self.centroid)
    
    def get_bounding_box(self) -> BoundingBox:
        """Get bounding box of cell."""
        if len(self.boundary) == 0:
            # Use centroid and area
            radius = np.sqrt(self.area / np.pi)
            return BoundingBox(
                min_x=self.centroid[0] - radius,
                max_x=self.centroid[0] + radius,
                min_y=self.centroid[1] - radius,
                max_y=self.centroid[1] + radius
            )
        
        min_coords = np.min(self.boundary, axis=0)
        max_coords = np.max(self.boundary, axis=0)
        
        return BoundingBox(
            min_x=min_coords[0],
            max_x=max_coords[0],
            min_y=min_coords[1],
            max_y=max_coords[1]
        )
    
    # ============================================================
    # SPATIAL RELATIONSHIP METHODS
    # ============================================================
    
    def compute_neighbors(
        self,
        other_cells: List['CellObject'],
        max_distance: float = 50.0
    ) -> None:
        """
        Compute neighboring cells within distance threshold.
        
        Args:
            other_cells: List of other cells
            max_distance: Maximum distance for neighbors (microns)
        """
        self.neighbor_cell_ids = []
        self.neighbor_distances = []
        
        for other_cell in other_cells:
            if other_cell.object_id == self.object_id:
                continue
            
            # Compute distance
            distance = np.linalg.norm(
                np.array(self.centroid) - np.array(other_cell.centroid)
            )
            
            if distance <= max_distance:
                self.neighbor_cell_ids.append(other_cell.object_id)
                self.neighbor_distances.append(distance)
        
        # Find nearest neighbor
        if self.neighbor_distances:
            self.nearest_neighbor_distance = min(self.neighbor_distances)
    
    def is_immune_cell(self) -> bool:
        """Check if cell is an immune cell."""
        immune_types = [
            CellType.IMMUNE, CellType.T_CELL, CellType.B_CELL,
            CellType.MACROPHAGE, CellType.NEUTROPHIL, CellType.NK_CELL,
            CellType.DENDRITIC
        ]
        return self.cell_type in immune_types
    
    def is_tumor_cell(self) -> bool:
        """Check if cell is a tumor cell."""
        return self.cell_type == CellType.TUMOR
    
    def is_stromal_cell(self) -> bool:
        """Check if cell is a stromal cell."""
        stromal_types = [CellType.FIBROBLAST, CellType.ENDOTHELIAL, CellType.STROMAL]
        return self.cell_type in stromal_types
    
    # ============================================================
    # CONVERSION METHODS
    # ============================================================
    
    def to_cell_properties(self) -> CellProperties:
        """Convert CellObject to CellProperties for export."""
        return CellProperties(
            cell_id=int(self.object_id.split('_')[-1]) if '_' in str(self.object_id) else 0,
            area=self.area,
            perimeter=self.perimeter,
            centroid=self.centroid,
            circularity=self.circularity,
            eccentricity=self.eccentricity,
            solidity=self.solidity,
            extent=self.extent,
            major_axis_length=self.major_axis_length,
            minor_axis_length=self.minor_axis_length,
            orientation=self.orientation,
            boundary=self.boundary,
            mean_intensity=self.mean_intensity,
            integrated_intensity=self.integrated_intensity,
            std_intensity=self.std_intensity,
            cell_type=self.cell_type,
            cell_type_confidence=self.cell_type_confidence,
            confidence=self.segmentation_confidence
        )
    
    @classmethod
    def from_cell_properties(
        cls,
        props: CellProperties,
        object_id: str,
        parent_id: Optional[str] = None
    ) -> 'CellObject':
        """Create CellObject from CellProperties."""
        return cls(
            object_id=object_id,
            parent_id=parent_id,
            centroid=props.centroid,
            area=props.area,
            perimeter=props.perimeter,
            boundary=props.boundary,
            circularity=props.circularity,
            eccentricity=props.eccentricity,
            solidity=props.solidity,
            extent=props.extent,
            major_axis_length=props.major_axis_length,
            minor_axis_length=props.minor_axis_length,
            orientation=props.orientation,
            mean_intensity=props.mean_intensity,
            integrated_intensity=props.integrated_intensity,
            std_intensity=props.std_intensity,
            cell_type=props.cell_type,
            cell_type_confidence=props.cell_type_confidence,
            segmentation_confidence=props.confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'object_id': self.object_id,
            'parent_id': self.parent_id,
            'centroid': self.centroid,
            'area': self.area,
            'perimeter': self.perimeter,
            'circularity': self.circularity,
            'cell_type': self.cell_type.value if self.cell_type else None,
            'cell_type_confidence': self.cell_type_confidence,
            'in_tumor_region': self.in_tumor_region,
            'distance_to_tumor_boundary': self.distance_to_tumor_boundary,
            'nearest_neighbor_distance': self.nearest_neighbor_distance,
        }


# ============================================================
# CELL POPULATION
# ============================================================

@dataclass
class CellPopulation:
    """
    Collection of cells with summary statistics.
    
    Used to group cells by region or type.
    """
    cell_ids: List[str] = field(default_factory=list)
    region_id: str = ""
    region_type: str = ""  # "tumor", "stroma", "invasive_margin"
    
    # Population statistics
    count: int = 0
    mean_area: float = 0.0
    mean_circularity: float = 0.0
    density: float = 0.0  # cells per square mm
    
    # Type distribution
    type_distribution: Dict[CellType, int] = field(default_factory=dict)
    type_ratios: Dict[CellType, float] = field(default_factory=dict)
    
    # Spatial statistics
    mean_nearest_neighbor_distance: float = 0.0
    clustering_index: float = 0.0  # 0-1, higher = more clustered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cell_ids': self.cell_ids,
            'region_id': self.region_id,
            'region_type': self.region_type,
            'count': self.count,
            'mean_area': self.mean_area,
            'density': self.density,
            'type_distribution': {k.value: v for k, v in self.type_distribution.items()},
            'type_ratios': {k.value: v for k, v in self.type_ratios.items()},
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_cell_object_from_segmentation(
    cell_props: CellProperties,
    parent_id: str,
    region_type: str = "unknown",
    pixel_size: float = 1.0
) -> CellObject:
    """
    Create a CellObject from CellProperties segmentation result.
    
    Args:
        cell_props: Cell properties from segmentation
        parent_id: ID of parent region
        region_type: Type of region
        pixel_size: Pixel size in microns
        
    Returns:
        CellObject ready for hierarchy integration
    """
    object_id = f"{parent_id}_cell_{cell_props.cell_id}"
    
    cell_obj = CellObject.from_cell_properties(
        cell_props,
        object_id=object_id,
        parent_id=parent_id
    )
    
    cell_obj.metadata['region_type'] = region_type
    
    return cell_obj


def aggregate_cell_population(
    cells: List[CellObject],
    region_id: str,
    region_type: str,
    region_area: Optional[float] = None
) -> CellPopulation:
    """
    Create CellPopulation from a list of cells.
    
    Args:
        cells: List of CellObject instances
        region_id: ID of the region
        region_type: Type of region
        region_area: Area of region in square microns (for density)
        
    Returns:
        CellPopulation with aggregated statistics
    """
    if not cells:
        return CellPopulation(region_id=region_id, region_type=region_type)
    
    # Collect cell IDs
    cell_ids = [c.object_id for c in cells]
    
    # Compute statistics
    areas = [c.area for c in cells]
    circularities = [c.circularity for c in cells]
    
    # Type distribution
    type_counts = {}
    for cell in cells:
        if cell.cell_type:
            type_counts[cell.cell_type] = type_counts.get(cell.cell_type, 0) + 1
    
    type_ratios = {}
    if cells:
        for cell_type, count in type_counts.items():
            type_ratios[cell_type] = count / len(cells)
    
    # Spatial statistics
    nn_distances = [
        c.nearest_neighbor_distance for c in cells
        if c.nearest_neighbor_distance is not None
    ]
    mean_nn_dist = float(np.mean(nn_distances)) if nn_distances else 0.0
    
    # Compute density
    density = 0.0
    if region_area and region_area > 0:
        # Convert to cells per square mm
        density = len(cells) / (region_area / 1e6)
    
    return CellPopulation(
        cell_ids=cell_ids,
        region_id=region_id,
        region_type=region_type,
        count=len(cells),
        mean_area=float(np.mean(areas)),
        mean_circularity=float(np.mean(circularities)),
        density=density,
        type_distribution=type_counts,
        type_ratios=type_ratios,
        mean_nearest_neighbor_distance=mean_nn_dist
    )


# Export
__all__ = [
    # Enums
    'SegmentationMode',
    'ImageModality',
    'CellType',
    'ClassificationMode',
    
    # Parameters
    'SegmentationParams',
    'ClassificationParams',
    'QuantificationParams',
    
    # Results
    'CellProperties',
    'SegmentationResult',
    'ClassificationResult',
    'QuantificationResult',
    'CellAnalysisResult',
    
    # Hierarchy objects
    'CellObject',
    'CellPopulation',
    
    # Helper functions
    'create_cell_object_from_segmentation',
    'aggregate_cell_population',
]