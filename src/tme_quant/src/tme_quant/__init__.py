"""
TMEQuant — Tumor Microenvironment Quantification Platform

Quick start
-----------
>>> import tme_quant as tq
>>> fibers = tq.FiberExtractionAnalyzer().extract_2d(image, tq.ExtractionParams())
>>> tacs   = tq.classify_fiber_tacs(fibers, boundary_roi)
>>> cells  = tq.CellAnalyzer().segment(image, tq.SegmentationParams())
"""

__version__ = "0.2.0"
__author__ = "UW-LOCI / TMEQuant Development Team"

# ── Data model ────────────────────────────────────────────────────────────────
from .core import (
    TMEObject, TMEType, ObjectType,
    Geometry, GeometryType,
    Measurement, Classification, TMEMetadata,
    BoundingBox, ROI,
    compute_region_centroid, compute_region_area,
    point_in_polygon, compute_convex_hull, buffer_polygon,
    TMEHierarchy, ImageEntry,
    ROIManager, ROIObject, ANNOTATION_TYPES,
    FiberAnalysisError, ROIProcessingError, BoundaryAnalysisError,
    FeatureExtractionError, ImageProcessingError,
)
from .core.project import TMEProject
from .core.io import save_project, load_project, export_project_summary
from .core.tme_objects.fiber_objects import FiberObject
from .core.tme_objects.cell_objects import CellObject
from .core.tme_objects.tumor_objects import TumorRegion

# ── Fiber analysis ────────────────────────────────────────────────────────────
from .fiber_analysis import (
    FiberExtractionAnalyzer, BaseExtractionMethod,
    FiberOrientationAnalyzer, BaseOrientationMethod,
    CTFireExtraction, CurveAlignOrientation,
    classify_fiber_tacs, classify_fiber_segment_tacs_like, get_tacs_color,
    ExtractionParams, ExtractionResult,
    OrientationParams, OrientationResult,
    FiberAnalysisResult,
)

# ── Cell analysis ─────────────────────────────────────────────────────────────
from .cell_analysis import (
    CellAnalyzer,
    CellSegmentationAnalyzer, CellClassificationAnalyzer, CellQuantificationAnalyzer,
    CellAnalysisResult,
    SegmentationParams, ClassificationParams, QuantificationParams,
)

# ── TME / interaction analysis ────────────────────────────────────────────────
from .tme_analysis import (
    TMEAnalyzer, InteractionDetector, InteractionNetworkAnalyzer,
    RegionManager, MeasurementEngine,
    StandardTMEPipeline, InteractionAnalysisPipeline,
)

# ── Image registration ────────────────────────────────────────────────────────
from .image_registration import RegistrationManager

__all__ = [
    # Core
    "TMEObject", "TMEType", "ObjectType",
    "Geometry", "GeometryType", "Measurement", "Classification", "TMEMetadata",
    "BoundingBox", "ROI",
    "compute_region_centroid", "compute_region_area",
    "point_in_polygon", "compute_convex_hull", "buffer_polygon",
    "TMEHierarchy", "TMEProject", "ImageEntry",
    "ROIManager", "ROIObject", "ANNOTATION_TYPES",
    "FiberObject", "CellObject", "TumorRegion",
    # Fiber
    "FiberExtractionAnalyzer", "BaseExtractionMethod",
    "FiberOrientationAnalyzer", "BaseOrientationMethod",
    "CTFireExtraction", "CurveAlignOrientation",
    "classify_fiber_tacs", "classify_fiber_segment_tacs_like", "get_tacs_color",
    "ExtractionParams", "ExtractionResult",
    "OrientationParams", "OrientationResult", "FiberAnalysisResult",
    # Cell
    "CellAnalyzer",
    "CellSegmentationAnalyzer", "CellClassificationAnalyzer", "CellQuantificationAnalyzer",
    "CellAnalysisResult",
    "SegmentationParams", "ClassificationParams", "QuantificationParams",
    # TME
    "TMEAnalyzer", "InteractionDetector", "InteractionNetworkAnalyzer",
    "RegionManager", "MeasurementEngine",
    "StandardTMEPipeline", "InteractionAnalysisPipeline",
    # Registration
    "RegistrationManager",
    # Project IO
    "save_project", "load_project", "export_project_summary",
    # Exceptions
    "FiberAnalysisError", "ROIProcessingError", "BoundaryAnalysisError",
    "FeatureExtractionError", "ImageProcessingError",
]
