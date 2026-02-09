"""
QuPath-like Hierarchical Object Model for Image Analysis

A standalone package that can be used independently or integrated
into larger bioimage analysis projects.
"""

__version__ = "0.1.0"
__author__ = "BioImage Team"

# Core exports
from .core.enums import (
    ObjectType, RoiType, ChannelType, MeasurementType
)
from .core.base_models import PathObject, BaseObject
from .core.specialized_models import (
    ImageObject,
    TissueRegion,
    TumorRegion,
    Cell,
    Nucleus,
    Cytoplasm,
    Fiber,
    Gland,
    Vessel,
)
from .core.geometry import ROI, GeometryOperations
from .core.image_models import ImageEntry, ImageChannel
from .core.project import QuPathLikeProject

# Measurement exports
from .measurement.calculator import MeasurementCalculator
from .measurement.intensity import IntensityCalculator
from .measurement.morphology import MorphologyCalculator
from .measurement.texture import TextureCalculator
from .measurement.spatial import SpatialCalculator

# IO exports
from .io.persistence import ProjectPersistence
from .io.qupath_io import QuPathIO
from .io.formats import export_to_dataframe

# Analysis exports
from .analysis.pipelines import (
    HierarchicalAnalysisPipeline,
    CellAnalysisPipeline,
    TissueAnalysisPipeline,
)
from .analysis.detection import (
    ObjectDetector,
    CellDetector,
    NucleusDetector,
    FiberDetector,
)
from .analysis.classification import ObjectClassifier

# Visualization exports
from .visualization.napari_bridge import NapariBridge
from .visualization.matplotlib_vis import MatplotlibVisualizer
from .visualization.colors import ColorManager

# Utils exports
from .utils.serialization import (
    to_json, from_json, to_pickle, from_pickle
)
from .utils.geometry_utils import (
    calculate_area, calculate_perimeter, calculate_centroid
)
from .utils.image_utils import (
    extract_roi_data, create_mask_from_roi
)

__all__ = [
    # Core
    "ObjectType", "RoiType", "ChannelType", "MeasurementType",
    "PathObject", "BaseObject",
    "ImageObject", "TissueRegion", "TumorRegion", "Cell",
    "Nucleus", "Cytoplasm", "Fiber", "Gland", "Vessel",
    "ROI", "GeometryOperations",
    "ImageEntry", "ImageChannel",
    "QuPathLikeProject",
    
    # Measurement
    "MeasurementCalculator",
    "IntensityCalculator",
    "MorphologyCalculator",
    "TextureCalculator",
    "SpatialCalculator",
    
    # IO
    "ProjectPersistence",
    "QuPathIO",
    "export_to_dataframe",
    
    # Analysis
    "HierarchicalAnalysisPipeline",
    "CellAnalysisPipeline",
    "TissueAnalysisPipeline",
    "ObjectDetector",
    "CellDetector",
    "NucleusDetector",
    "FiberDetector",
    "ObjectClassifier",
    
    # Visualization
    "NapariBridge",
    "MatplotlibVisualizer",
    "ColorManager",
    
    # Utils
    "to_json", "from_json", "to_pickle", "from_pickle",
    "calculate_area", "calculate_perimeter", "calculate_centroid",
    "extract_roi_data", "create_mask_from_roi",
]