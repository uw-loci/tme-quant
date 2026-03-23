"""
Core TME object model and hierarchy management.
"""

from .base_models import (
    TMEObject, TMEType, ObjectType,
    Geometry, GeometryType,
    Measurement, Classification, TMEMetadata,
)
from .hierarchy import TMEHierarchy
from .image_entry import ImageEntry
from .roi_manager import ROIManager, ROIObject, ANNOTATION_TYPES

__all__ = [
    # Base model
    'TMEObject', 'TMEType', 'ObjectType',
    'Geometry', 'GeometryType',
    'Measurement', 'Classification', 'TMEMetadata',
    # Hierarchy
    'TMEHierarchy',
    # Image
    'ImageEntry',
    # ROI management
    'ROIManager', 'ROIObject', 'ANNOTATION_TYPES',
]