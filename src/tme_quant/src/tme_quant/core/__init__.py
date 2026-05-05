"""
Core TME object model and hierarchy management.
"""

from .base_models import (
    TMEObject, TMEType, ObjectType,
    Geometry, GeometryType,
    Measurement, Classification, TMEMetadata,
)
from .geometry import (
    BoundingBox, ROI,
    compute_region_centroid,
    compute_region_area,
    point_in_polygon,
    compute_convex_hull,
    buffer_polygon,
)
from .hierarchy import TMEHierarchy
from .image_entry import ImageEntry
from .roi_manager import ROIManager, ROIObject, ANNOTATION_TYPES
from .io import save_project, load_project, export_project_summary
from .exceptions import (
    FiberAnalysisError,
    ROIProcessingError,
    BoundaryAnalysisError,
    FeatureExtractionError,
    ImageProcessingError,
)
from .tme_objects.cell_objects import CellType
from .tme_objects.fiber_objects import FiberPopulation
from .tme_objects.tumor_objects import TumorGrade

__all__ = [
    # Base model
    'TMEObject', 'TMEType', 'ObjectType',
    'Geometry', 'GeometryType',
    'Measurement', 'Classification', 'TMEMetadata',
    # Geometry
    'BoundingBox', 'ROI',
    'compute_region_centroid', 'compute_region_area',
    'point_in_polygon', 'compute_convex_hull', 'buffer_polygon',
    # Hierarchy
    'TMEHierarchy',
    # Image
    'ImageEntry',
    # ROI management
    'ROIManager', 'ROIObject', 'ANNOTATION_TYPES',
    # IO
    'save_project', 'load_project', 'export_project_summary',
    # Exceptions
    'FiberAnalysisError', 'ROIProcessingError', 'BoundaryAnalysisError',
    'FeatureExtractionError', 'ImageProcessingError',
    # Domain object types
    'CellType', 'FiberPopulation', 'TumorGrade',
]
