"""
tme_objects/base_objects.py — backward-compatibility shim.

All concrete object classes (FiberObject, CellObject, TumorRegion, ...)
now inherit directly from the unified TMEObject in core/base_models.py.

This file re-exports the symbols that existing code in this package
imported from the old local base_models, so no import changes are needed
in cell_objects.py, fiber_objects.py, or any other tme_objects module.
"""

from ..base_models import (   # noqa: F401  (re-export)
    TMEObject,
    ObjectType,
    TMEType,
    GeometryType,
    Geometry,
    Measurement,
    Classification,
    TMEMetadata,
)

__all__ = [
    "TMEObject",
    "ObjectType",
    "TMEType",
    "GeometryType",
    "Geometry",
    "Measurement",
    "Classification",
    "TMEMetadata",
]
