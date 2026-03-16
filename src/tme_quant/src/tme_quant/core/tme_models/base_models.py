"""
tme_models/base_models.py — backward-compatibility shim.

All concrete model classes (FiberObject, CellObject, TumorRegion, ...)
now inherit directly from the unified TMEObject in core/base_models.py.

This file re-exports the symbols that existing code in this package
imported from the old local base_models, so no import changes are needed
in cell_model.py, fiber_model.py, or any other tme_models module.
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