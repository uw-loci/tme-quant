"""
Configuration parameters for TME analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class AnalysisMode(Enum):
    """TME analysis modes."""
    CELL_BASED = "cell_based"           # Cell → Fibers/Cells
    TUMOR_BASED = "tumor_based"         # Tumor boundary → Fibers/Cells
    FIBER_BASED = "fiber_based"         # Fiber → Cells/Fibers
    ROI_BASED = "roi_based"             # Custom ROI → All components


class InteractionStrategy(Enum):
    """How to find interaction pairs."""
    NEAREST = "nearest"                 # Nearest neighbor only
    RADIUS = "radius"                   # All within radius
    K_NEAREST = "k_nearest"             # K nearest neighbors
    CONTACT = "contact"                 # Physical contact/overlap


class TumorDetectionMethod(Enum):
    """Methods for tumor region detection."""
    CLUSTERING = "clustering"           # DBSCAN/hierarchical clustering
    DENSITY = "density"                 # Kernel density estimation
    DEEP_LEARNING = "deep_learning"     # U-Net, Mask R-CNN
    MANUAL = "manual"                   # User annotation
    CELL_TYPE = "cell_type"             # Based on cell classification


@dataclass
class TMEAnalysisParams:
    """Parameters for TME analysis."""
    
    # Analysis mode
    mode: AnalysisMode = AnalysisMode.TUMOR_BASED
    
    # Interaction detection
    interaction_strategy: InteractionStrategy = InteractionStrategy.RADIUS
    interaction_distance: float = 100.0  # microns
    k_neighbors: int = 5  # For K_NEAREST strategy
    
    # Distance thresholds for different analysis types
    cell_fiber_distance: float = 50.0    # Cell-fiber interaction threshold
    cell_cell_distance: float = 30.0     # Cell-cell interaction threshold
    fiber_fiber_distance: float = 20.0   # Fiber-fiber interaction threshold
    tumor_boundary_distance: float = 100.0  # TACS analysis boundary zone
    
    # Measurement flags
    compute_tacs: bool = True
    compute_morphology: bool = True
    compute_spatial: bool = True
    compute_orientation: bool = True
    compute_density: bool = True
    compute_prognostic: bool = True
    
    # TACS-specific parameters
    tacs_angle_threshold_perpendicular: float = 30.0  # TACS-3
    tacs_angle_threshold_parallel: float = 60.0       # TACS-2
    tacs_straightness_threshold: float = 0.7
    
    # Zone generation (for tumor-based analysis)
    generate_zones: bool = True
    invasive_margin_width: float = 50.0   # microns
    stroma_width: float = 200.0           # microns
    
    # Output options
    return_interaction_pairs: bool = True
    return_distance_maps: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'interaction_strategy': self.interaction_strategy.value,
            'interaction_distance': self.interaction_distance,
            'compute_tacs': self.compute_tacs,
            'compute_prognostic': self.compute_prognostic,
        }


@dataclass
class TumorDetectionParams:
    """Parameters for tumor region detection."""
    
    method: TumorDetectionMethod = TumorDetectionMethod.CLUSTERING
    
    # Clustering parameters
    clustering_algorithm: str = "dbscan"  # "dbscan", "hierarchical"
    dbscan_eps: float = 100.0             # microns
    dbscan_min_samples: int = 10
    
    # Density parameters
    density_bandwidth: float = 50.0       # KDE bandwidth
    density_threshold: float = 0.5        # Threshold for tumor vs stroma
    
    # Deep learning parameters
    dl_model_path: Optional[str] = None
    dl_confidence_threshold: float = 0.5
    
    # Cell type filtering (use classified tumor cells)
    use_cell_classification: bool = True
    tumor_cell_types: List[str] = field(default_factory=lambda: ['tumor'])
    
    # Post-processing
    min_tumor_area: float = 1000.0        # square microns
    smooth_boundary: bool = True
    smoothing_sigma: float = 10.0         # Gaussian smoothing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'clustering_algorithm': self.clustering_algorithm,
            'min_tumor_area': self.min_tumor_area,
        }


@dataclass
class InteractionPair:
    """Represents an interaction between two TME components."""
    
    # Component IDs
    source_id: str                        # Cell, fiber, or tumor ID
    target_id: str                        # Interacting component ID
    
    # Component types
    source_type: str                      # "cell", "fiber", "tumor"
    target_type: str
    
    # Spatial metrics
    distance: float                       # Distance between components (microns)
    contact: bool = False                 # Physical contact/overlap
    
    # Orientation metrics (if applicable)
    relative_angle: Optional[float] = None           # Degrees
    angle_to_boundary_normal: Optional[float] = None # For TACS
    angle_to_boundary_tangent: Optional[float] = None
    
    # Interaction region
    interaction_point: Optional[Tuple[float, float]] = None  # (x, y)
    nearest_boundary_point: Optional[Tuple[float, float]] = None
    
    # Classification
    interaction_type: Optional[str] = None  # "TACS-1", "TACS-2", "TACS-3", etc.
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'distance': self.distance,
            'contact': self.contact,
            'relative_angle': self.relative_angle,
            'interaction_type': self.interaction_type,
        }


@dataclass
class TMEAnalysisResult:
    """Results from TME analysis."""
    
    analysis_id: str
    mode: AnalysisMode
    
    # Interaction pairs
    interaction_pairs: List[InteractionPair] = field(default_factory=list)
    
    # Measurements by category
    tacs_features: Optional[Dict[str, Any]] = None
    morphological_features: Optional[Dict[str, Any]] = None
    spatial_features: Optional[Dict[str, Any]] = None
    orientation_features: Optional[Dict[str, Any]] = None
    density_features: Optional[Dict[str, Any]] = None
    
    # Prognostic features
    prognostic_scores: Optional[Dict[str, float]] = None
    
    # Distance maps
    distance_maps: Optional[Dict[str, Any]] = None
    
    # Region information
    roi_info: Optional[Dict[str, Any]] = None
    tumor_regions: List[str] = field(default_factory=list)
    zones: Optional[Dict[str, Any]] = None
    
    # Summary statistics
    summary: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'analysis_id': self.analysis_id,
            'mode': self.mode.value,
            'n_interactions': len(self.interaction_pairs),
            'tacs_features': self.tacs_features,
            'prognostic_scores': self.prognostic_scores,
            'summary': self.summary,
        }