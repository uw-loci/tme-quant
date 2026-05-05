"""
External tool bridges — tools that tme_quant calls outward to.

Included
--------
fiji_bridge  — Fiji/ImageJ subprocess/pyimagej bridge, used by
               OrientationJMethod and RidgeDetectionMethod.

Planned
-------
qupath_bridge — GeoJSON batch export for QuPath annotation import.
matlab_bridge — MATLAB Engine bridge (legacy curvelet support).

Not included
------------
napari — The dependency arrow is reversed: napari calls INTO tme_quant.
         Napari-specific code lives in the ``napari-tme-quant`` plugin
         package and must never be imported here.
"""

from .fiji_bridge import FijiBridge, FijiBackendMixin

__all__ = [
    "FijiBridge",
    "FijiBackendMixin",
]
