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
    CellType, FiberPopulation, TumorGrade,
)
from .core.project import TMEProject
from .core.io import save_project, load_project, export_project_summary
from .core.tme_objects.fiber_objects import FiberObject
from .core.tme_objects.cell_objects import CellObject
from .core.tme_objects.tumor_objects import TumorRegion

# ── External tool integrations ────────────────────────────────────────────────
from .integrations import FijiBridge, FijiBackendMixin

# ── Fiber analysis ────────────────────────────────────────────────────────────
from .fiber_analysis import (
    # Analyzers
    FiberExtractionAnalyzer, BaseExtractionMethod,
    FiberOrientationAnalyzer, BaseOrientationMethod,
    # Extraction methods
    CTFireExtraction, SkeletonExtractionMethod, RidgeDetectionMethod,
    # Orientation methods
    CurveAlignOrientation, GradientOrientationMethod,
    StructureTensorMethod, OrientationJMethod,
    # TACS
    classify_fiber_tacs, classify_fiber_segment_tacs_like, get_tacs_color,
    # Params (general)
    ExtractionParams, ExtractionResult,
    OrientationParams, OrientationResult,
    # Params (method-specific)
    CTFireParams, CurveAlignParams,
    SkeletonParams, RidgeDetectionParams,
    OrientationJParams, GradientParams, StructureTensorParams,
    # Fiber data
    FiberProperties, FiberAnalysisResult,
    # Boundary / ROI utilities
    extract_boundary_coords_from_mask,
    # Visualization
    generate_fiber_overlay, generate_fiber_heatmap,
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
    CurveAlignPipelineResult, curvealign_curvelets_mode_pipeline, analyze_tacs_zone,
    TMEAnalysisParams, TMEAnalysisResult, AnalysisMode,
    compute_fiber_alignment_to_roi, compute_orientation_relative_to_roi,
)

# ── Image registration ────────────────────────────────────────────────────────
from .image_registration import RegistrationManager

__all__ = [
    # ── Core: base model ──────────────────────────────────────────────────────
    "TMEObject", "TMEType", "ObjectType",
    "Geometry", "GeometryType", "Measurement", "Classification", "TMEMetadata",
    "BoundingBox", "ROI",
    "compute_region_centroid", "compute_region_area",
    "point_in_polygon", "compute_convex_hull", "buffer_polygon",
    "TMEHierarchy", "TMEProject", "ImageEntry",
    "ROIManager", "ROIObject", "ANNOTATION_TYPES",
    # ── Core: domain objects ──────────────────────────────────────────────────
    "FiberObject", "FiberPopulation",
    "CellObject", "CellType",
    "TumorRegion", "TumorGrade",
    # ── Core: IO + exceptions ─────────────────────────────────────────────────
    "save_project", "load_project", "export_project_summary",
    "FiberAnalysisError", "ROIProcessingError", "BoundaryAnalysisError",
    "FeatureExtractionError", "ImageProcessingError",
    # ── Integrations ──────────────────────────────────────────────────────────
    "FijiBridge", "FijiBackendMixin",
    # ── Fiber: analyzers ──────────────────────────────────────────────────────
    "FiberExtractionAnalyzer", "BaseExtractionMethod",
    "FiberOrientationAnalyzer", "BaseOrientationMethod",
    # ── Fiber: extraction methods ─────────────────────────────────────────────
    "CTFireExtraction", "SkeletonExtractionMethod", "RidgeDetectionMethod",
    # ── Fiber: orientation methods ────────────────────────────────────────────
    "CurveAlignOrientation", "GradientOrientationMethod",
    "StructureTensorMethod", "OrientationJMethod",
    # ── Fiber: TACS ───────────────────────────────────────────────────────────
    "classify_fiber_tacs", "classify_fiber_segment_tacs_like", "get_tacs_color",
    # ── Fiber: params ─────────────────────────────────────────────────────────
    "ExtractionParams", "ExtractionResult",
    "OrientationParams", "OrientationResult", "FiberAnalysisResult",
    "CTFireParams", "CurveAlignParams",
    "SkeletonParams", "RidgeDetectionParams",
    "OrientationJParams", "GradientParams", "StructureTensorParams",
    "FiberProperties",
    # ── Fiber: utilities + visualization ─────────────────────────────────────
    "extract_boundary_coords_from_mask",
    "generate_fiber_overlay", "generate_fiber_heatmap",
    # ── Cell ─────────────────────────────────────────────────────────────────
    "CellAnalyzer",
    "CellSegmentationAnalyzer", "CellClassificationAnalyzer", "CellQuantificationAnalyzer",
    "CellAnalysisResult",
    "SegmentationParams", "ClassificationParams", "QuantificationParams",
    # ── TME / pipelines ───────────────────────────────────────────────────────
    "TMEAnalyzer", "InteractionDetector", "InteractionNetworkAnalyzer",
    "RegionManager", "MeasurementEngine",
    "StandardTMEPipeline", "InteractionAnalysisPipeline",
    "CurveAlignPipelineResult", "curvealign_curvelets_mode_pipeline", "analyze_tacs_zone",
    "TMEAnalysisParams", "TMEAnalysisResult", "AnalysisMode",
    "compute_fiber_alignment_to_roi", "compute_orientation_relative_to_roi",
    # ── Registration ──────────────────────────────────────────────────────────
    "RegistrationManager",
]
