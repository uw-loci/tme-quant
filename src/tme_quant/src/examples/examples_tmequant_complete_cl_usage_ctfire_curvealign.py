"""
TMEQuant Complete Workflows — CurveAlign and CT-FIRE
=====================================================

Two end-to-end examples demonstrating the current TMEQuant fiber analysis API:

  Workflow 1  — CurveAlign (2-D)
      Fiber segment orientation analysis using the curvelet transform.
      Uses CurveAlignParams directly (subclass of OrientationParams).

  Workflow 2  — CT-FIRE (2-D)
      Individual fiber extraction using the two-stage CT-FIRE pipeline:
        Stage 1 (CT): multi-scale curvelet transform → fiber mask.
        Stage 2 (FIRE): Euclidean distance transform of the mask, then
          ridge-tracing along the distance-transform medial axis.
          Width is integral to tracing (not post-hoc), so touching fibers
          of different thickness are correctly separated.
      Uses CTFireParams directly (subclass of ExtractionParams).

  Workflow 3  — 3-D volumetric analysis
      CurveAlign on a 3-D volume via analyze_orientation_3d() (true
      volumetric curvelet transform with curvelops; warns if falling back).
      Skeleton extraction on a 3-D volume via extract_fibers_3d() with
      SkeletonParams (Lee algorithm, volumetric).
      CT-FIRE 3-D is noted as pending C++ FIRE extension.

Key parameter classes (today's API)
-------------------------------------
  CurveAlignParams    — window_size, overlap, curvelet_levels, curvelet_angles,
                        compute_coherency, compute_energy, use_matlab_backend,
                        return_fiber_segments, keep_values, pixel_size
  CTFireParams        — ctfire_threshold, ctfire_n_levels, ctfire_n_angles,
                        straightness_threshold, use_matlab_backend, z_spacing,
                        min/max_fiber_length, min/max_fiber_width, pixel_size
  SkeletonParams      — skeleton_method ('lee'/'zhang'), threshold_method,
                        manual_threshold, min_branch_length, smooth_skeleton

FIRE vs Skeleton distinction (important)
-----------------------------------------
  CT-FIRE / FIRE:
    - Takes the binary fiber mask as input.
    - Computes the distance transform; radius at each pixel = local half-width.
    - Traces fibers along the distance-transform ridge — width is integral.
    - Returned traces carry (row, col, radius_px) per centerline point.
    - Correctly separates touching fibers of different thickness.
  Skeleton:
    - Takes the binary fiber mask → 1-pixel-wide medial-axis skeleton.
    - Traces connected components of the skeleton.
    - Width estimated afterward from intensity cross-sections (FWHM).
    - Faster; appropriate for thin, well-separated fibers.

Backend availability
---------------------
  Curvelet transform:   curvelops (production) > MATLAB Engine > NumPy fallback
  FIRE algorithm:       CT-FIRE C++ extension > pure-Python approximation
  3-D CT-FIRE FIRE:     requires C++ extension (not yet compiled → NotImplementedError)
  3-D Skeleton:         fully available (Lee algorithm in skimage)
  3-D CurveAlign:       fully available (warns if no curvelops)

  Check at runtime:
    from tme_quant.fiber_analysis.utils import available_backends, ctfire_backend_status
    print(available_backends())        # {'curvelops': bool, 'matlab': bool, 'numpy': True}
    print(ctfire_backend_status())     # {'cpp_available': bool, '3d_supported': bool, ...}

TACS conventions (unchanged from prior version)
-------------------------------------------------
  All classify_fiber_tacs() calls use keyword angle_to_tangent=
  compute_angle_to_boundary_normal() returns angle to NORMAL;
  convert: angle_to_tangent = 90 - angle_to_normal
  TACS-3:  60–90° from boundary tangent (perpendicular, INVASIVE)
  TACS-2:   0–30° from boundary tangent (parallel)
  TACS-1:  30–60° or curly fibers
"""

import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy.spatial import cKDTree
from shapely.geometry import Point

# ── Image registration ────────────────────────────────────────────────────────
from tme_quant.image_registration.methods.intensity_based import HESHGRegistration
from tme_quant.image_registration.config import RegistrationParams, TransformType

# ── Fiber analysis — top-level coordinator ───────────────────────────────────
from tme_quant.fiber_analysis import FiberAnalyzer, FiberOrientationAnalyzer

# ── Fiber analysis — parameter and result classes ────────────────────────────
# Import the concrete subclasses directly; they carry all mode-specific fields.
# Using the base ExtractionParams / OrientationParams directly is supported but
# will not expose CT-FIRE-specific fields (ctfire_threshold, z_spacing, etc.).
from tme_quant.fiber_analysis.config import (
    # Orientation
    OrientationMode,
    OrientationParams,        # base — use subclasses for mode-specific fields
    CurveAlignParams,         # window_size, overlap, curvelet_levels/angles, …
    CurveAlignResult,
    # Extraction
    ExtractionMode,
    ExtractionParams,         # base — use subclasses for mode-specific fields
    CTFireParams,             # ctfire_threshold, ctfire_n_levels/angles, z_spacing, …
    CTFireResult,
    SkeletonParams,           # skeleton_method, threshold_method, min_branch_length, …
    SkeletonResult,
    FiberProperties,          # per-fiber geometry: length, width, straightness, angle, …
)

# ── Fiber analysis — backend status utilities ────────────────────────────────
from tme_quant.fiber_analysis.utils import (
    available_backends,       # {'curvelops': bool, 'matlab': bool, 'numpy': True}
    ctfire_backend_status,    # {'cpp_available': bool, '3d_supported': bool, …}
)

# ── Geometry utilities ────────────────────────────────────────────────────────
# compute_angle_to_boundary_normal returns angle relative to boundary NORMAL.
# For TACS classification convert: angle_to_tangent = 90 - angle_to_normal.
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,   # returns angle-to-NORMAL (0–90°)
    compute_relative_angles,            # returns dict with both angle_to_normal
                                        # and angle_to_tangent
)

# ── Cell analysis ─────────────────────────────────────────────────────────────
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config import SegmentationMode

try:
    from tme_quant.cell_analysis.config import SegmentationParams
except ImportError:
    from dataclasses import dataclass, field
    from typing import Any
    @dataclass
    class SegmentationParams:
        mode: Any = None
        stardist_model: str = '2D_versatile_he'
        pixel_size: float = 1.0
        min_cell_size: float = 20.0
        probability_threshold: float = 0.5

# ── TME analysis ──────────────────────────────────────────────────────────────
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams,
    AnalysisMode,
    TumorDetectionParams,
    TumorDetectionMethod,
    TMEAnalysisResult,
)

# ── TACS classification ───────────────────────────────────────────────────────
from tme_quant.fiber_analysis.tacs import (
    classify_fiber_tacs,               # full TACS with straightness (CT-FIRE)
    classify_fiber_segment_tacs_like,  # orientation-only (CurveAlign)
    get_tacs_color,
)

# ── Interaction detection & measurement ──────────────────────────────────────
from tme_quant.tme_analysis.interaction_detector import InteractionDetector
from tme_quant.tme_analysis.measurement_engine import MeasurementEngine
from tme_quant.tme_analysis.interaction_features import (
    annotate_interaction_pairs,
    compute_alignment_heterogeneity,
)
from tme_quant.tme_analysis.pipelines.interaction_analysis_pipeline import (
    InteractionAnalysisPipeline,
    PipelineConfig,
)
from tme_quant.tme_analysis.interaction_network import (
    InteractionNetworkAnalyzer,
)

# ── Distance utilities ────────────────────────────────────────────────────────
from tme_quant.tme_analysis.utils.distance_utils import compute_distance_to_boundary
from tme_quant.fiber_analysis.utils.geometry_utils import find_nearest_boundary_point

# ── Export ────────────────────────────────────────────────────────────────────
from tme_quant.tme_analysis.io import export_tme_analysis_results


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW 1: CURVEALIGN — FIBER SEGMENT ORIENTATION ANALYSIS (2-D)
# ─────────────────────────────────────────────────────────────────────────────

def workflow_1_curvealign_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    sample_id: str = "patient_001_curvealign",
) -> Dict:
    """
    Complete CurveAlign workflow for fiber segment orientation analysis.

    Each orientation point from CurveAlign is treated as a fiber segment.
    TACS-like classification is orientation-based only (no straightness).

    Parameters
    ----------
    he_image_path:  Path to H&E image.
    shg_image_path: Path to SHG collagen image.
    output_dir:     Directory to write all outputs.
    pixel_size:     µm per pixel.
    sample_id:      Identifier used for output file names.

    Returns
    -------
    Dict with keys: sample_id, registered_he, fiber_segments,
    segment_metrics, cells, tumor_regions, summary_stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WORKFLOW 1: CurveAlign Fiber Segment Analysis")
    print(f"Sample: {sample_id}")
    print("=" * 80)

    # ── Print backend status ─────────────────────────────────────────────────
    backends = available_backends()
    print(f"\n  Curvelet backends: {backends}")
    if not backends['curvelops']:
        print("  WARNING: curvelops not installed — NumPy fallback in use.")
        print("           Install with: pip install curvelops")

    # ── [1/10] Load images ───────────────────────────────────────────────────
    print("\n[1/10] Loading images...")
    he_image  = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    print(f"  ✓ H&E: {he_image.shape},  SHG: {shg_image.shape}")

    # ── [2/10] Register H&E to SHG ──────────────────────────────────────────
    print("\n[2/10] Registering H&E to SHG (Keikhosravi 2020)...")
    reg_params = RegistrationParams(
        transform_type=TransformType.AFFINE,
        use_multiresolution=True,
        pyramid_levels=3,
        num_iterations=200,
    )
    registration = HESHGRegistration()
    reg_result   = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    io.imsave(
        output_dir / f"{sample_id}_HE_registered.tif",
        (registered_he * 255).astype(np.uint8),
    )

    # ── [3/10] CurveAlign orientation analysis ───────────────────────────────
    print("\n[3/10] CurveAlign orientation analysis...")
    # Use CurveAlignParams directly — it carries all window/curvelet settings.
    # 'positions' is not a valid keep_values entry; valid options are:
    # 'angles', 'alignment', 'energy', 'all'.
    # compute_statistics (not compute_stats) controls summary statistics.
    # return_fiber_segments is a CurveAlignParams-specific flag.
    orientation_params = CurveAlignParams(
        pixel_size=pixel_size,
        window_size=64,
        overlap=0.5,
        curvelet_levels=4,
        curvelet_angles=8,
        compute_coherency=True,
        compute_energy=True,
        return_fiber_segments=True,
        keep_values=['angles', 'alignment', 'energy'],
        compute_statistics=True,
    )
    orientation_analyzer = FiberOrientationAnalyzer()
    orientation_result  = orientation_analyzer.analyze_2d(
        shg_image, orientation_params
    )
    print(f"  ✓ Mean orientation: {orientation_result.mean_orientation:.2f}°")
    print(f"  ✓ Mean alignment:   {orientation_result.mean_alignment:.4f}")
    if hasattr(orientation_result, 'n_windows_analyzed'):
        print(f"  ✓ Windows analyzed: {orientation_result.n_windows_analyzed}")

    # ── [4/10] Extract fiber segments ───────────────────────────────────────
    print("\n[4/10] Extracting fiber segments from orientation map...")
    fiber_segments = _extract_fiber_segments_from_curvealign(
        orientation_map=orientation_result.orientation_map,
        alignment_map=orientation_result.alignment_map,
        pixel_size=pixel_size,
        subsample=2,
    )
    print(f"  ✓ Extracted {len(fiber_segments)} fiber segments")

    # ── [5/10] Cell segmentation ─────────────────────────────────────────────
    print("\n[5/10] Segmenting cells (StarDist)...")
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0,
        probability_threshold=0.5,
    )
    cell_analyzer = CellAnalyzer()
    seg_result    = cell_analyzer.segment_cells_2d(
        registered_he, seg_params, image_id=sample_id
    )
    cells = seg_result.cells
    print(f"  ✓ Segmented {len(cells)} cells")

    # ── [6/10] Tumor boundary detection ─────────────────────────────────────
    print("\n[6/10] Detecting tumor boundaries (DBSCAN)...")
    tumor_params = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        clustering_algorithm='dbscan',
        dbscan_eps=100.0,
        dbscan_min_samples=10,
        min_tumor_area=1000.0,
        smooth_boundary=True,
    )
    tme_analyzer  = TMEAnalyzer()
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    print(f"  ✓ Detected {len(tumor_regions)} tumor region(s)")

    # ── [7/10] Compute segment metrics ──────────────────────────────────────
    print("\n[7/10] Computing fiber segment metrics...")
    segment_metrics = _compute_fiber_segment_metrics(
        fiber_segments=fiber_segments,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        pixel_size=pixel_size,
    )
    print(f"  ✓ Metrics computed for {len(segment_metrics)} segments")
    print(f"  Mean K-NN alignment: {segment_metrics['local_alignment'].mean():.3f}")
    print(f"  Mean density:        {segment_metrics['local_density'].mean():.1f} segs/mm²")
    tacs_counts = segment_metrics['tacs_type'].value_counts()
    print("  TACS-like distribution:")
    for tacs_type, count in tacs_counts.items():
        if tacs_type:
            print(f"    {tacs_type}: {count}  ({count / len(segment_metrics) * 100:.1f}%)")

    # ── [8/10] Generate heatmaps ─────────────────────────────────────────────
    print("\n[8/10] Generating heatmaps...")
    _generate_segment_heatmaps(
        shg_image=shg_image,
        segment_metrics=segment_metrics,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    print("  ✓ Orientation, alignment, density heatmaps written")

    # ── [9/10] Create visualizations ────────────────────────────────────────
    print("\n[9/10] Creating overlay visualizations...")
    _create_segment_visualizations(
        shg_image=shg_image,
        registered_he=registered_he,
        segment_metrics=segment_metrics,
        cells=cells,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    print("  ✓ Segment TACS overlay written")

    # ── [10/10] Export ───────────────────────────────────────────────────────
    print("\n[10/10] Exporting data...")
    segment_metrics.to_csv(
        output_dir / f"{sample_id}_fiber_segment_metrics.csv", index=False
    )

    summary_stats = {
        'sample_id':        sample_id,
        'n_segments':       len(segment_metrics),
        'n_cells':          len(cells),
        'n_tumors':         len(tumor_regions),
        'mean_orientation': float(segment_metrics['orientation'].mean()),
        'mean_alignment':   float(segment_metrics['local_alignment'].mean()),
        'mean_density':     float(segment_metrics['local_density'].mean()),
    }
    for tacs_type in ['TACS-1-like', 'TACS-2-like', 'TACS-3-like']:
        count = int((segment_metrics['tacs_type'] == tacs_type).sum())
        summary_stats[f'{tacs_type}_count'] = count
        summary_stats[f'{tacs_type}_ratio'] = count / max(len(segment_metrics), 1)

    pd.DataFrame([summary_stats]).to_csv(
        output_dir / f"{sample_id}_summary.csv", index=False
    )
    print(f"  ✓ Exported to {output_dir}")

    print("\n" + "=" * 80)
    print("CURVEALIGN WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nSample:          {sample_id}")
    print(f"Fiber segments:  {len(fiber_segments):,}")
    print(f"Cells:           {len(cells):,}")
    print(f"Tumors:          {len(tumor_regions)}")
    print(f"\nTACS-like classification (within 100 µm of boundary):")
    for tacs_type in ['TACS-1-like', 'TACS-2-like', 'TACS-3-like']:
        c = summary_stats[f'{tacs_type}_count']
        r = summary_stats[f'{tacs_type}_ratio']
        print(f"  {tacs_type}: {c}  ({r * 100:.1f}%)")
    print("=" * 80)

    return {
        'sample_id':       sample_id,
        'registered_he':   registered_he,
        'fiber_segments':  fiber_segments,
        'segment_metrics': segment_metrics,
        'cells':           cells,
        'tumor_regions':   tumor_regions,
        'summary_stats':   summary_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW 2: CT-FIRE — INDIVIDUAL FIBER EXTRACTION (2-D)
# ─────────────────────────────────────────────────────────────────────────────

def workflow_2_ctfire_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    sample_id: str = "patient_001_ctfire",
) -> Dict:
    """
    Complete CT-FIRE workflow for individual fiber analysis with full TACS.

    CT-FIRE pipeline (two stages):
      Stage 1 — CT (Curvelet Transform):
        Multi-scale curvelet decomposition → angular energy map → threshold →
        binary fiber mask.
      Stage 2 — FIRE (Fiber Extraction from mask):
        Euclidean distance transform of the mask gives per-pixel fiber radius.
        Fibers are traced along distance-transform ridges (medial axis).
        Width is integral to tracing — not post-hoc — so touching fibers of
        different thickness are correctly separated.
        Returns (row, col, radius_px) traces per fiber.

    Backend dispatch:
      - curvelops FDCT → MATLAB Engine → NumPy fallback for the CT stage.
      - CT-FIRE C++ extension → pure-Python approximation for the FIRE stage.
      Call ctfire_backend_status() to check at runtime.

    Parameters
    ----------
    he_image_path:  Path to H&E image.
    shg_image_path: Path to SHG collagen image.
    output_dir:     Directory to write all outputs.
    pixel_size:     µm per pixel.
    sample_id:      Identifier used for output file names.

    Returns
    -------
    Dict with keys: sample_id, registered_he, fibers, fiber_metrics,
    cells, tumor_regions, tme_result, pipeline_result,
    network_results, summary_stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WORKFLOW 2: CT-FIRE Individual Fiber Analysis")
    print(f"Sample: {sample_id}")
    print("=" * 80)

    # ── Print backend status ─────────────────────────────────────────────────
    fire_status = ctfire_backend_status()
    print(f"\n  CT-FIRE backend status: {fire_status}")
    if not fire_status['cpp_available']:
        print("  NOTE: C++ FIRE extension not compiled — Python fallback in use.")
        print("        See fiber_analysis/utils/ctfire_utils.py for build instructions.")
    if not fire_status['3d_supported']:
        print("  NOTE: 3-D CT-FIRE extraction is not yet available (requires C++).")
        print("        Use SkeletonParams with extract_fibers_3d() for 3-D analysis.")

    # ── [1/12] Load images ───────────────────────────────────────────────────
    print("\n[1/12] Loading images...")
    he_image  = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    print(f"  ✓ H&E: {he_image.shape},  SHG: {shg_image.shape}")

    # ── [2/12] Register H&E to SHG ──────────────────────────────────────────
    print("\n[2/12] Registering H&E to SHG...")
    reg_params = RegistrationParams(
        transform_type=TransformType.AFFINE,
        use_multiresolution=True,
        pyramid_levels=3,
        num_iterations=200,
    )
    registration  = HESHGRegistration()
    reg_result    = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    io.imsave(
        output_dir / f"{sample_id}_HE_registered.tif",
        (registered_he * 255).astype(np.uint8),
    )

    # ── [3/12] CT-FIRE fiber extraction ─────────────────────────────────────
    print("\n[3/12] Extracting individual fibers (CT-FIRE)...")
    # Use CTFireParams — it carries CT-FIRE-specific fields not in ExtractionParams:
    #   ctfire_threshold, ctfire_n_levels, ctfire_n_angles, straightness_threshold,
    #   use_matlab_backend, z_spacing (for 3-D anisotropic volumes).
    # The FIRE stage works on the fiber mask via the distance transform.
    extraction_params = CTFireParams(
        pixel_size=pixel_size,
        ctfire_threshold=0.1,
        ctfire_n_levels=5,
        ctfire_n_angles=16,
        straightness_threshold=0.0,     # keep all; stricter filter applied in TACS step
        use_matlab_backend=False,
        min_fiber_length=10.0,
        max_fiber_length=500.0,
        min_fiber_width=1.0,
        max_fiber_width=10.0,
        measure_length=True,
        measure_width=True,
        measure_straightness=True,
        measure_angle=True,
        measure_curvature=False,
        extract_centerlines=True,
    )
    fiber_analyzer = FiberAnalyzer()
    fiber_result   = fiber_analyzer.extract_2d(
        shg_image, extraction_params
    )
    # fiber_result is a CTFireResult; fiber_result.fibers is List[FiberProperties].
    # Each FiberProperties has: .length, .width (from distance-transform), .straightness,
    # .angle, .curvature, .centerline (N,2 row,col), .aspect_ratio.
    # Also: .fiber_mask, .curvelet_energy_map, .labeled_fibers, .n_candidates.
    fibers = fiber_result.fibers
    print(f"  ✓ Extracted {len(fibers)} individual fibers")
    if fibers:
        print(f"  Mean length:       {np.mean([f.length for f in fibers]):.2f} µm")
        print(f"  Mean width (DT):   {np.mean([f.width for f in fibers]):.2f} µm")
        print(f"  Mean straightness: {np.mean([f.straightness for f in fibers]):.3f}")
    print(f"  Candidates before filter: {fiber_result.n_candidates}")
    print(f"  Fiber mask coverage:      "
          f"{fiber_result.fiber_mask.mean() * 100:.1f}% of image")

    # ── [4/12] Cell segmentation ─────────────────────────────────────────────
    print("\n[4/12] Segmenting cells (StarDist)...")
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0,
    )
    cell_analyzer = CellAnalyzer()
    seg_result    = cell_analyzer.segment_cells_2d(
        registered_he, seg_params, image_id=sample_id
    )
    cells = seg_result.cells
    print(f"  ✓ Segmented {len(cells)} cells")

    # ── [5/12] Tumor boundary detection ─────────────────────────────────────
    print("\n[5/12] Detecting tumor boundaries...")
    tumor_params = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        clustering_algorithm='dbscan',
        dbscan_eps=100.0,
        dbscan_min_samples=10,
        min_tumor_area=1000.0,
        smooth_boundary=True,
    )
    tme_analyzer  = TMEAnalyzer()
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    print(f"  ✓ Detected {len(tumor_regions)} tumor region(s)")

    # ── [6/12] TME analysis (fiber-based) ───────────────────────────────────
    print("\n[6/12] TME analysis (fiber-tumor interactions)...")
    tme_params = TMEAnalysisParams(
        mode=AnalysisMode.FIBER_BASED,
        tumor_boundary_distance=100.0,
        compute_tacs=True,
        tacs_angle_threshold_perpendicular=30.0,
        tacs_angle_threshold_parallel=60.0,
        tacs_straightness_threshold=0.7,
        fiber_fiber_distance=50.0,
        compute_morphology=True,
        compute_spatial=True,
        compute_orientation=True,
        compute_density=True,
        compute_prognostic=True,
        return_interaction_pairs=True,
    )
    tme_result = tme_analyzer.analyze(
        cells=cells,
        fibers=fibers,
        tumor_regions=tumor_regions,
        params=tme_params,
        analysis_id=f"{sample_id}_tme",
    )
    print("  ✓ TME analysis complete")
    if tme_result.tacs_features:
        tf = tme_result.tacs_features
        print(f"  TACS-1: {tf['tacs1_ratio'] * 100:.1f}%")
        print(f"  TACS-2: {tf['tacs2_ratio'] * 100:.1f}%")
        print(f"  TACS-3: {tf['tacs3_ratio'] * 100:.1f}%  (INVASIVE)")
        print(f"  mean_angle_to_tangent: {tf.get('mean_angle_to_tangent', 'n/a')}")

    # ── [7/12] End-to-end pipeline ───────────────────────────────────────────
    print("\n[7/12] Running InteractionAnalysisPipeline...")
    pipeline_cfg = PipelineConfig(
        boundary_distance=100.0,
        tacs_zone_width=100.0,
        contact_threshold=5.0,
        compute_tacs=True,
        compute_mechanical=True,
        compute_contact_patterns=True,
        compute_prognostic=True,
        export_dir=output_dir,
    )
    pipeline = InteractionAnalysisPipeline(pipeline_cfg)
    pipeline_result = pipeline.run(
        cells=cells,
        fibers=fibers,
        tumors=tumor_regions,
        image_id=sample_id,
    )
    print(f"  ✓ Pipeline: {len(pipeline_result.interaction_pairs)} pairs detected")

    # ── [8/12] Compute individual fiber metrics ──────────────────────────────
    print("\n[8/12] Computing individual fiber metrics...")
    fiber_metrics = _compute_individual_fiber_metrics(
        fibers=fibers,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        straightness_threshold=0.7,
        pixel_size=pixel_size,
    )
    print(f"  ✓ Metrics computed for {len(fiber_metrics)} fibers")
    if len(fiber_metrics):
        print(f"  Mean K-NN alignment: {fiber_metrics['local_alignment'].mean():.3f}")
        print(f"  Mean density:        "
              f"{fiber_metrics['local_fiber_density'].mean():.1f} fibers/mm²")

    # ── [9/12] Network analysis ──────────────────────────────────────────────
    print("\n[9/12] Interaction network analysis...")
    network_analyzer = InteractionNetworkAnalyzer(
        weight_by='distance',
        community_method='greedy',
    )
    network_results = network_analyzer.analyze(
        pairs=pipeline_result.interaction_pairs,
        top_n_hubs=10,
    )
    nm = network_results['network_metrics']
    print(f"  ✓ Graph: {nm['n_nodes']} nodes, {nm['n_edges']} edges")
    print(f"  Density:    {nm['density']:.3f}")

    # ── [10/12] Generate heatmaps ────────────────────────────────────────────
    print("\n[10/12] Generating heatmaps...")
    _generate_fiber_heatmaps(
        shg_image=shg_image,
        fiber_metrics=fiber_metrics,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    print("  ✓ Heatmaps written")

    # ── [11/12] Create visualizations ───────────────────────────────────────
    print("\n[11/12] Creating visualizations...")
    _create_fiber_visualizations(
        shg_image=shg_image,
        registered_he=registered_he,
        fibers=fibers,
        fiber_metrics=fiber_metrics,
        cells=cells,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    print("  ✓ Overlays written")

    # ── [12/12] Export ───────────────────────────────────────────────────────
    print("\n[12/12] Exporting results...")
    export_tme_analysis_results(
        tme_result,
        output_dir=output_dir,
        formats=['csv', 'excel', 'json'],
        prefix=sample_id,
    )
    fiber_metrics.to_csv(output_dir / f"{sample_id}_fiber_metrics.csv", index=False)

    summary_stats: Dict = {
        'sample_id':          sample_id,
        'n_fibers':           len(fibers),
        'n_cells':            len(cells),
        'n_tumors':           len(tumor_regions),
        'mean_fiber_length':  float(fiber_metrics['length'].mean()) if len(fiber_metrics) else 0.0,
        'mean_fiber_width':   float(fiber_metrics['width'].mean())  if len(fiber_metrics) else 0.0,
        'mean_straightness':  float(fiber_metrics['straightness'].mean()) if len(fiber_metrics) else 0.0,
        'mean_alignment':     float(fiber_metrics['local_alignment'].mean()) if len(fiber_metrics) else 0.0,
        'mean_density':       float(fiber_metrics['local_fiber_density'].mean()) if len(fiber_metrics) else 0.0,
    }
    if tme_result.tacs_features:
        tf = tme_result.tacs_features
        summary_stats.update({
            'tacs1_count':           tf['tacs1_count'],
            'tacs2_count':           tf['tacs2_count'],
            'tacs3_count':           tf['tacs3_count'],
            'tacs1_ratio':           tf['tacs1_ratio'],
            'tacs2_ratio':           tf['tacs2_ratio'],
            'tacs3_ratio':           tf['tacs3_ratio'],
            'dominant_tacs':         tf['dominant_tacs_type'],
            'mean_angle_to_tangent': tf.get('mean_angle_to_tangent', None),
        })
    if tme_result.prognostic_scores:
        summary_stats['tme_risk_score'] = tme_result.prognostic_scores.get(
            'overall_tme_risk_score', None
        )
    summary_stats.update(pipeline_result.prognostic_scores)

    pd.DataFrame([summary_stats]).to_csv(
        output_dir / f"{sample_id}_summary.csv", index=False
    )
    print(f"  ✓ Exported to {output_dir}")

    print("\n" + "=" * 80)
    print("CT-FIRE WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nSample:           {sample_id}")
    print(f"Individual fibers: {len(fibers):,}")
    print(f"Cells:            {len(cells):,}")
    print(f"Tumors:           {len(tumor_regions)}")
    if fibers:
        print(f"\nFiber properties:")
        print(f"  Mean length:      {summary_stats['mean_fiber_length']:.2f} µm")
        print(f"  Mean width (DT):  {summary_stats['mean_fiber_width']:.2f} µm")
        print(f"  Mean straightness:{summary_stats['mean_straightness']:.3f}")
    if tme_result.tacs_features:
        print(f"\nTACS classification (within 100 µm, straightness ≥ 0.7):")
        print(f"  TACS-1 (random):               "
              f"{summary_stats['tacs1_count']}  "
              f"({summary_stats['tacs1_ratio'] * 100:.1f}%)")
        print(f"  TACS-2 (parallel):             "
              f"{summary_stats['tacs2_count']}  "
              f"({summary_stats['tacs2_ratio'] * 100:.1f}%)")
        print(f"  TACS-3 (perpendicular/INVASIVE): "
              f"{summary_stats['tacs3_count']}  "
              f"({summary_stats['tacs3_ratio'] * 100:.1f}%)")
        print(f"  Dominant: {summary_stats['dominant_tacs']}")
    print("=" * 80)

    return {
        'sample_id':        sample_id,
        'registered_he':    registered_he,
        'fibers':           fibers,
        'fiber_metrics':    fiber_metrics,
        'cells':            cells,
        'tumor_regions':    tumor_regions,
        'tme_result':       tme_result,
        'pipeline_result':  pipeline_result,
        'network_results':  network_results,
        'summary_stats':    summary_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW 3: 3-D VOLUMETRIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def workflow_3_volumetric(
    shg_volume_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    z_spacing: float = 1.0,
    sample_id: str = "patient_001_3d",
) -> Dict:
    """
    3-D volumetric fiber analysis workflow.

    Demonstrates:
      1. CurveAlign 3-D orientation — true volumetric curvelet transform
         (curvelops FDCT3D preferred; warns if falling back to slice-by-slice).
      2. Skeleton 3-D extraction — volumetric Lee skeletonization, fibers
         that span multiple z-planes, (z, row, col) centerlines.
      3. CT-FIRE 3-D — noted as pending (requires C++ FIRE extension).
         The CT stage (curvelet transform) is already implemented for 3-D;
         only the FIRE stage awaits the C++ extension compilation.

    3-D analysis means the full volume (Z, H, W) is treated as a single
    entity.  Slice-by-slice analysis of a 3-D stack is NOT the same thing
    and is deliberately not exposed through analyze_3d / extract_3d.

    Parameters
    ----------
    shg_volume_path : Path to a 3-D SHG volume (Z, H, W), e.g. a TIFF stack.
    output_dir      : Directory for outputs.
    pixel_size      : In-plane (XY) pixel size in µm.
    z_spacing       : Inter-slice spacing in µm.
    sample_id       : Sample identifier.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WORKFLOW 3: 3-D Volumetric Fiber Analysis")
    print(f"Sample: {sample_id}")
    print("=" * 80)

    # ── Backend status ───────────────────────────────────────────────────────
    backends    = available_backends()
    fire_status = ctfire_backend_status()
    print(f"\n  Curvelet backends: {backends}")
    print(f"  CT-FIRE status:    {fire_status}")
    if not backends['curvelops']:
        print("  WARNING: curvelops not installed — 3-D CurveAlign will use "
              "slice-by-slice fallback.  Install with: pip install curvelops")
    if not fire_status['3d_supported']:
        print("  NOTE: 3-D CT-FIRE not available (C++ extension pending).")
        print("        Using SkeletonParams for 3-D fiber extraction instead.")

    # ── Load volume ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading 3-D SHG volume...")
    volume = io.imread(shg_volume_path)   # expected shape (Z, H, W)
    if volume.ndim == 2:
        raise ValueError(
            f"Expected a 3-D volume (Z, H, W), got shape {volume.shape}. "
            "Pass a multi-page TIFF or reshape accordingly."
        )
    print(f"  ✓ Volume shape: {volume.shape}  (Z={volume.shape[0]}, "
          f"H={volume.shape[1]}, W={volume.shape[2]})")

    # ── 3-D CurveAlign orientation ───────────────────────────────────────────
    print("\n[2/4] 3-D CurveAlign orientation analysis...")
    # analyze_3d applies curvelet_transform_3d to the full volume, not slice-by-slice.
    # Result orientation_map has shape (Z, H, W).
    orient_params_3d = CurveAlignParams(
        pixel_size=pixel_size,
        window_size=32,          # smaller window for 3-D to keep memory manageable
        overlap=0.5,
        curvelet_levels=3,
        curvelet_angles=8,
        compute_coherency=True,
        compute_energy=False,
        keep_values=['angles', 'alignment'],
        compute_statistics=True,
    )
    orientation_analyzer = FiberOrientationAnalyzer()
    orient_result_3d    = orientation_analyzer.analyze_3d(
        volume, orient_params_3d
    )
    print(f"  ✓ Orientation volume shape: {orient_result_3d.orientation_map.shape}")
    print(f"  ✓ Mean orientation:         {orient_result_3d.mean_orientation:.2f}°")
    print(f"  ✓ Alignment score:          {orient_result_3d.alignment_score:.4f}")

    # ── 3-D Skeleton fiber extraction ────────────────────────────────────────
    print("\n[3/4] 3-D volumetric skeleton fiber extraction...")
    # SkeletonParams with skeleton_method='lee' (default) runs true 3-D
    # skeletonization on the full volume — NOT slice-by-slice.
    # Each FiberProperties.centerline is (N, 3) float32 in (z, row, col).
    # Zhang algorithm is 2-D only; using it with extract_3d raises ValueError.
    skel_params_3d = SkeletonParams(
        pixel_size=pixel_size,
        skeleton_method='lee',        # default; supports 3-D natively
        threshold_method='otsu',
        min_branch_length=5.0,
        smooth_skeleton=True,
        min_fiber_length=10.0,
        max_fiber_length=2000.0,
        extract_centerlines=True,
    )
    fiber_analyzer_3d = FiberAnalyzer()
    skel_result_3d    = fiber_analyzer_3d.extract_3d(
        volume, skel_params_3d
    )
    fibers_3d = skel_result_3d.fibers
    print(f"  ✓ Extracted {len(fibers_3d)} 3-D fibers")
    print(f"  ✓ Skeleton volume shape: {skel_result_3d.skeleton_mask.shape}")
    # Centerlines are (z, row, col) — show first fiber as example
    if fibers_3d and fibers_3d[0].centerline is not None:
        cl = fibers_3d[0].centerline
        print(f"  ✓ First fiber centerline: shape={cl.shape}, "
              f"z_range=[{cl[:,0].min():.0f}, {cl[:,0].max():.0f}]")

    # ── CT-FIRE 3-D status note ───────────────────────────────────────────────
    print("\n[4/4] CT-FIRE 3-D status check...")
    if fire_status['3d_supported']:
        # C++ extension available — run 3-D CT-FIRE
        ctfire_params_3d = CTFireParams(
            pixel_size=pixel_size,
            z_spacing=z_spacing,       # inter-slice spacing in µm
            ctfire_threshold=0.1,
            ctfire_n_levels=3,
            ctfire_n_angles=8,
            min_fiber_length=10.0,
            extract_centerlines=True,
        )
        ctfire_result_3d = fiber_analyzer_3d.extract_3d(
            volume, ctfire_params_3d
        )
        print(f"  ✓ CT-FIRE 3-D: extracted {len(ctfire_result_3d.fibers)} fibers")
    else:
        print("  CT-FIRE 3-D: requires the _ctfire_cpp extension (not yet compiled).")
        print("  The CT stage (curvelet transform) is fully 3-D-capable already.")
        print("  The FIRE stage (distance-transform ridge tracing) needs C++ for 3-D.")
        print("  Compilation instructions: see fiber_analysis/utils/ctfire_utils.py")
        print("  Using Skeleton 3-D result as the 3-D fiber extraction output.")
        ctfire_result_3d = None

    print("\n" + "=" * 80)
    print("3-D VOLUMETRIC WORKFLOW COMPLETE")
    print("=" * 80)

    return {
        'sample_id':         sample_id,
        'orient_result_3d':  orient_result_3d,
        'skel_result_3d':    skel_result_3d,
        'fibers_3d':         fibers_3d,
        'ctfire_result_3d':  ctfire_result_3d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS — CURVEALIGN
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fiber_segments_from_curvealign(
    orientation_map: np.ndarray,
    alignment_map: Optional[np.ndarray],
    pixel_size: float,
    subsample: int = 1,
) -> pd.DataFrame:
    """Extract fiber segments from a CurveAlign orientation map."""
    h, w = orientation_map.shape
    segments = []
    seg_id = 0
    for y in range(0, h, subsample):
        for x in range(0, w, subsample):
            orientation = orientation_map[y, x]
            alignment   = alignment_map[y, x] if alignment_map is not None else 0.0
            if not np.isnan(orientation) and orientation > 0:
                segments.append({
                    'segment_id':                f'seg_{seg_id:06d}',
                    'segment_index':             seg_id,
                    'position_x':                x,
                    'position_y':                y,
                    'orientation':               orientation,
                    'local_alignment_intrinsic': alignment,
                })
                seg_id += 1
    return pd.DataFrame(segments)


def _compute_fiber_segment_metrics(
    fiber_segments: pd.DataFrame,
    tumor_regions: List,
    k_neighbors: int,
    bbox_size: float,
    tacs_zone_width: float,
    pixel_size: float,
) -> pd.DataFrame:
    """
    Compute spatial, alignment, density, and TACS-like metrics for
    CurveAlign fiber segments.

    Angle convention:
      compute_angle_to_boundary_normal() → angle to NORMAL
      angle_to_tangent = 90 - angle_to_normal
      classify_fiber_segment_tacs_like(angle_to_tangent=...) → TACS-X-like
    """
    positions = fiber_segments[['position_x', 'position_y']].values
    tree      = cKDTree(positions)
    metrics   = []

    for _, row in fiber_segments.iterrows():
        seg_data    = row.to_dict()
        position    = (row['position_x'], row['position_y'])
        orientation = row['orientation']

        min_dist         = np.inf
        angle_to_tangent = None

        for tumor in tumor_regions:
            if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
                point    = Point(position)
                boundary = tumor.roi.polygon.boundary
                dist     = point.distance(boundary) * pixel_size

                if dist < min_dist:
                    min_dist = dist
                    nearest_pt = boundary.interpolate(boundary.project(point))
                    second_pt  = boundary.interpolate(
                        min(boundary.length, boundary.project(point) + 5.0)
                    )
                    angle_to_normal = compute_angle_to_boundary_normal(
                        fiber_orientation=orientation,
                        boundary_point1=(nearest_pt.x, nearest_pt.y),
                        boundary_point2=(second_pt.x,  second_pt.y),
                    )
                    if not np.isnan(angle_to_normal):
                        angle_to_tangent = 90.0 - angle_to_normal

        seg_data['distance_to_tumor'] = min_dist
        seg_data['angle_to_tangent']  = angle_to_tangent
        seg_data['angle_to_normal']   = (
            90.0 - angle_to_tangent if angle_to_tangent is not None else None
        )

        if min_dist <= tacs_zone_width and angle_to_tangent is not None:
            seg_data['tacs_type'] = classify_fiber_segment_tacs_like(
                angle_to_tangent=angle_to_tangent,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
            )
        else:
            seg_data['tacs_type'] = None

        if len(fiber_segments) > k_neighbors:
            _, indices = tree.query(position, k=k_neighbors + 1)
            neighbor_orientations = (
                fiber_segments.iloc[indices[1:]]['orientation'].values
            )
            diffs = np.abs(orientation - neighbor_orientations)
            diffs = np.minimum(diffs, 180.0 - diffs)
            seg_data['local_alignment'] = 1.0 - (np.mean(diffs) / 90.0)
            seg_data['mean_angle_diff'] = float(np.mean(diffs))
        else:
            seg_data['local_alignment'] = 0.0
            seg_data['mean_angle_diff'] = 0.0

        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby  = tree.query_ball_point(position, bbox_radius)
        count   = len(nearby) - 1
        area_mm2 = np.pi * ((bbox_size / 2.0) / 1000.0) ** 2
        seg_data['local_density'] = count / area_mm2 if area_mm2 > 0 else 0.0

        metrics.append(seg_data)

    return pd.DataFrame(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS — CT-FIRE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_individual_fiber_metrics(
    fibers: List[FiberProperties],
    tumor_regions: List,
    k_neighbors: int,
    bbox_size: float,
    tacs_zone_width: float,
    straightness_threshold: float,
    pixel_size: float,
) -> pd.DataFrame:
    """
    Compute spatial, alignment, density, and full TACS metrics for
    individual CT-FIRE fibers.

    FiberProperties fields used:
      .fiber_id, .length, .width (from distance-transform), .straightness,
      .angle, .centerline [(N,2) float32 in (row,col)], .aspect_ratio.

    Angle convention:
      compute_angle_to_boundary_normal() → angle to NORMAL
      angle_to_tangent = 90 - angle_to_normal
      classify_fiber_tacs(angle_to_tangent=...) → TACS-1/2/3
    """
    if not fibers:
        return pd.DataFrame()

    def _fiber_center(f) -> np.ndarray:
        """Return center coordinates from either FiberObject or FiberProperties."""
        if hasattr(f, 'get_center_coordinates'):
            c = f.get_center_coordinates()
        else:
            c = getattr(f, 'center_coordinates', np.array([]))
        return c if len(c) > 0 else np.array([0.0, 0.0])

    fiber_centers = np.array([_fiber_center(f) for f in fibers])
    tree    = cKDTree(fiber_centers)
    metrics = []

    for i, fiber in enumerate(fibers):
        center       = fiber_centers[i]
        angle        = fiber.angle
        straightness = fiber.straightness

        fiber_data = {
            'fiber_id':    getattr(fiber, 'object_id', getattr(fiber, 'fiber_id', i)),
            'fiber_index': i,
            'length':      fiber.length,
            'width':       fiber.width,    # distance-transform width (µm)
            'straightness': straightness,
            'orientation': angle,
            'midpoint_x':  float(center[0]),
            'midpoint_y':  float(center[1]),
        }

        min_dist         = np.inf
        angle_to_tangent = None

        for tumor in tumor_regions:
            if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
                point    = Point(center)
                boundary = tumor.roi.polygon.boundary
                dist     = point.distance(boundary) * pixel_size

                if dist < min_dist:
                    min_dist = dist
                    if angle is not None:
                        nearest_pt = boundary.interpolate(boundary.project(point))
                        second_pt  = boundary.interpolate(
                            min(boundary.length, boundary.project(point) + 5.0)
                        )
                        angle_to_normal = compute_angle_to_boundary_normal(
                            fiber_orientation=angle,
                            boundary_point1=(nearest_pt.x, nearest_pt.y),
                            boundary_point2=(second_pt.x,  second_pt.y),
                        )
                        if not np.isnan(angle_to_normal):
                            angle_to_tangent = 90.0 - angle_to_normal

        fiber_data['distance_to_tumor'] = min_dist
        fiber_data['angle_to_tangent']  = angle_to_tangent
        fiber_data['angle_to_normal']   = (
            90.0 - angle_to_tangent if angle_to_tangent is not None else None
        )

        if (
            min_dist <= tacs_zone_width
            and angle_to_tangent is not None
            and straightness is not None
        ):
            fiber_data['tacs_type'] = classify_fiber_tacs(
                angle_to_tangent=angle_to_tangent,
                straightness=straightness,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
                straightness_threshold=straightness_threshold,
            )
        else:
            fiber_data['tacs_type'] = None

        if len(fibers) > k_neighbors and angle is not None:
            _, indices = tree.query(center, k=k_neighbors + 1)
            neighbor_angles = [fibers[idx].angle for idx in indices[1:]]
            neighbor_angles = [a for a in neighbor_angles if a is not None]
            if neighbor_angles:
                diffs = np.array([abs((angle - na) % 180) for na in neighbor_angles])
                diffs = np.minimum(diffs, 180.0 - diffs)
                fiber_data['local_alignment'] = 1.0 - float(np.mean(diffs) / 90.0)
                fiber_data['mean_angle_diff'] = float(np.mean(diffs))
            else:
                fiber_data['local_alignment'] = 0.0
                fiber_data['mean_angle_diff'] = 0.0
        else:
            fiber_data['local_alignment'] = 0.0
            fiber_data['mean_angle_diff'] = 0.0

        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby = tree.query_ball_point(center, bbox_radius)
        count  = len(nearby) - 1
        area_mm2 = (bbox_size / 1000.0) ** 2
        fiber_data['local_fiber_density'] = count / area_mm2 if area_mm2 > 0 else 0.0

        metrics.append(fiber_data)

    return pd.DataFrame(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_segment_heatmaps(
    shg_image, segment_metrics, tumor_regions, output_dir, sample_id
):
    """Generate orientation, alignment and density heatmaps for CurveAlign."""
    from scipy.interpolate import griddata
    h, w = shg_image.shape[:2]
    resolution = 512
    gx, gy = np.meshgrid(
        np.linspace(0, w, resolution),
        np.linspace(0, h, resolution),
    )
    positions = segment_metrics[['position_x', 'position_y']].values
    for col, cmap, label in [
        ('orientation',    'hsv',     'Orientation (°)'),
        ('local_alignment','RdYlGn',  'K-NN Alignment'),
        ('local_density',  'viridis', 'Density (segs/mm²)'),
    ]:
        values = segment_metrics[col].fillna(0).values
        grid   = griddata(positions, values, (gx, gy), method='linear', fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(shg_image, cmap='gray', alpha=0.5)
        im = ax.imshow(grid, cmap=cmap, alpha=0.7)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f'{sample_id} — {label}')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_dir / f'{sample_id}_heatmap_{col}.png', dpi=150)
        plt.close(fig)


def _create_segment_visualizations(
    shg_image, registered_he, segment_metrics, cells, tumor_regions,
    output_dir, sample_id,
):
    """TACS-coloured segment overlay for CurveAlign."""
    import cv2
    shg_rgb = (
        cv2.cvtColor(
            (shg_image / shg_image.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        ) if shg_image.ndim == 2 else shg_image.copy()
    )
    he_rgb = (
        registered_he if registered_he.ndim == 3
        else cv2.cvtColor(
            (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
    )
    composite = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)
    overlay   = composite.copy()
    for _, row in segment_metrics.iterrows():
        tacs = row.get('tacs_type')
        if tacs:
            cv2.circle(
                overlay,
                (int(row['position_x']), int(row['position_y'])),
                3,
                get_tacs_color(tacs),
                -1,
            )
    for tumor in tumor_regions:
        if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
            coords = np.array(tumor.roi.polygon.exterior.coords).astype(np.int32)
            cv2.polylines(overlay, [coords], True, (255, 255, 0), 2)
    io.imsave(output_dir / f'{sample_id}_overlay_tacs.png', overlay)


def _generate_fiber_heatmaps(
    shg_image, fiber_metrics, tumor_regions, output_dir, sample_id
):
    """Generate orientation, alignment and density heatmaps for CT-FIRE."""
    from scipy.interpolate import griddata
    if len(fiber_metrics) == 0:
        return
    h, w = shg_image.shape[:2]
    resolution = 512
    gx, gy = np.meshgrid(
        np.linspace(0, w, resolution),
        np.linspace(0, h, resolution),
    )
    positions = fiber_metrics[['midpoint_x', 'midpoint_y']].values
    for col, cmap, label in [
        ('orientation',        'hsv',     'Orientation (°)'),
        ('local_alignment',    'RdYlGn',  'K-NN Alignment'),
        ('local_fiber_density','viridis', 'Density (fibers/mm²)'),
    ]:
        values = fiber_metrics[col].fillna(0).values
        grid   = griddata(positions, values, (gx, gy), method='linear', fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(shg_image, cmap='gray', alpha=0.5)
        im = ax.imshow(grid, cmap=cmap, alpha=0.7)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f'{sample_id} — {label}')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_dir / f'{sample_id}_heatmap_{col}.png', dpi=150)
        plt.close(fig)


def _create_fiber_visualizations(
    shg_image, registered_he, fibers, fiber_metrics, cells, tumor_regions,
    output_dir, sample_id,
):
    """TACS-coloured fiber overlay for CT-FIRE.

    fiber.centerline is (N, 2) float32 in (row, col) — i.e. [y, x] order.
    cv2.polylines expects [[x, y]] so we flip columns before drawing.
    """
    import cv2
    shg_rgb = (
        cv2.cvtColor(
            (shg_image / shg_image.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        ) if shg_image.ndim == 2 else shg_image.copy()
    )
    he_rgb = (
        registered_he if registered_he.ndim == 3
        else cv2.cvtColor(
            (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
    )
    composite = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)
    overlay   = composite.copy()

    for i, fiber in enumerate(fibers):
        tacs_type = (
            fiber_metrics.iloc[i].get('tacs_type')
            if i < len(fiber_metrics) else None
        )
        if tacs_type and fiber.centerline is not None and len(fiber.centerline) > 1:
            # centerline is (N, 2) in (row, col); cv2 needs (N, 1, 2) in (x, y)
            pts = fiber.centerline[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, get_tacs_color(tacs_type), 2)

    for tumor in tumor_regions:
        if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
            coords = np.array(tumor.roi.polygon.exterior.coords).astype(np.int32)
            cv2.polylines(overlay, [coords], True, (255, 255, 0), 3)

    io.imsave(output_dir / f'{sample_id}_overlay_tacs.png', overlay)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TMEQuant Complete Workflows — CurveAlign / CT-FIRE / 3-D")
    print("=" * 80)

    '''
    # ── Print runtime backend status ─────────────────────────────────────────
    print("\nRuntime backend status:")
    print(f"  Curvelet: {available_backends()}")
    print(f"  CT-FIRE:  {ctfire_backend_status()}")
    '''
   
    # # ── Workflow 1: CurveAlign ────────────────────────────────────────────────
    # print("\n\nWORKFLOW 1: CurveAlign Fiber Segments (2-D)")
    # print("=" * 80)
    # results_ca = workflow_1_curvealign_complete(
    #     he_image_path="data/patient_001_HE.tif",
    #     shg_image_path="data/patient_001_SHG.tif",
    #     output_dir="output/patient_001_curvealign",
    #     pixel_size=0.5,
    #     sample_id="patient_001_curvealign",
    # )
    
    
    # ── Workflow 2: CT-FIRE ──────────────────────────────────────────────────
    print("\n\nWORKFLOW 2: CT-FIRE Individual Fibers (2-D)")
    print("=" * 80)
    results_ct = workflow_2_ctfire_complete(
        he_image_path="data/patient_001_HE.tif",
        shg_image_path="data/patient_001_SHG.tif",
        output_dir="output/patient_001_ctfire",
        pixel_size=0.5,
        sample_id="patient_001_ctfire",
    )

    # ── Workflow 3: 3-D volumetric ────────────────────────────────────────────
    print("\n\nWORKFLOW 3: 3-D Volumetric Analysis")
    print("=" * 80)
    results_3d = workflow_3_volumetric(
        shg_volume_path="data/patient_001_SHG_3D.tif",
        output_dir="output/patient_001_3d",
        pixel_size=0.5,
        z_spacing=1.0,
        sample_id="patient_001_3d",
    )
    

    print("\n\n" + "=" * 80)
    print("ALL WORKFLOWS COMPLETE")
    print("=" * 80)
    print("\nParameter classes used:")
    print("  CurveAlignParams  — window_size, overlap, curvelet_levels/angles,")
    print("                      compute_coherency, compute_energy, keep_values,")
    print("                      return_fiber_segments, compute_statistics")
    print("  CTFireParams      — ctfire_threshold, ctfire_n_levels/angles,")
    print("                      straightness_threshold, z_spacing (3-D),")
    print("                      min/max_fiber_length, min/max_fiber_width")
    print("  SkeletonParams    — skeleton_method ('lee'/'zhang'), threshold_method,")
    print("                      min_branch_length, smooth_skeleton")
    print("\nFIRE vs Skeleton distinction:")
    print("  CT-FIRE FIRE:  mask → distance transform → ridge trace → (row,col,radius_px)")
    print("                 width is integral to tracing (not post-hoc)")
    print("  Skeleton:      mask → 1-px skeleton → component trace → (row,col)")
    print("                 width estimated afterward from intensity profiles")
    print("\nTACS angle convention (boundary tangent):")
    print("  TACS-3:  60–90° (perpendicular, INVASIVE)  — RED")
    print("  TACS-2:   0–30° (parallel)                 — GREEN")
    print("  TACS-1:  30–60° or curly                   — BLUE")
    print("\n3-D analysis note:")
    print("  CurveAlign 3-D:   analyze_orientation_3d() — volumetric, (Z,H,W) output")
    print("  Skeleton 3-D:     extract_fibers_3d() with SkeletonParams — volumetric Lee")
    print("  CT-FIRE 3-D:      extract_fibers_3d() with CTFireParams — needs C++ FIRE")
    print("  Slice-by-slice is NOT 3-D analysis and is not exposed through these APIs")
    print("=" * 80)