"""
ROI generation and annotation management.

Two complementary classes handle ROI work:

  RegionManager  (tme_analysis.core.region_manager)
      Automated tumor-region detection from cell populations:
      DBSCAN clustering, density estimation, cell-type filtering.
      Also provides generate_tumor_zones() and filter_cells/fibers_by_roi().
      Used internally by TMEAnalyzer.

  ROIManager  (core.roi_manager)
      Manual and imported annotation shape management:
      rectangle, circle, ellipse, polygon, freehand, line, point.
      QuPath GeoJSON import/export.
      First-class TMEHierarchy integration (each ROI is a TMEObject node).
"""

from ..core.region_manager import RegionManager
from ...core.roi_manager import ROIManager, ROIObject, ANNOTATION_TYPES

__all__ = [
    'RegionManager',
    'ROIManager',
    'ROIObject',
    'ANNOTATION_TYPES',
]
