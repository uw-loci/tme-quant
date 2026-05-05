"""
External tool bridges — tools that tme_quant calls outward to.

Included
--------
fiji_bridge   — Fiji/ImageJ subprocess/pyimagej bridge, used by
                OrientationJMethod and RidgeDetectionMethod.
qupath_bridge — Bidirectional QuPath GeoJSON bridge:
                export_hierarchy_geojson, load_qupath_annotations,
                load_qupath_measurements.

Planned
-------
matlab_bridge — MATLAB Engine bridge (legacy curvelet support).

Not included
------------
napari — The dependency arrow is reversed: napari calls INTO tme_quant.
         Napari-specific code lives in the ``napari-tme-quant`` plugin
         package and must never be imported here.
"""

from .fiji_bridge import FijiBridge, FijiBackendMixin
from .qupath_bridge import (
    export_hierarchy_geojson,
    load_qupath_annotations,
    load_qupath_measurements,
)

__all__ = [
    "FijiBridge",
    "FijiBackendMixin",
    "export_hierarchy_geojson",
    "load_qupath_annotations",
    "load_qupath_measurements",
]
