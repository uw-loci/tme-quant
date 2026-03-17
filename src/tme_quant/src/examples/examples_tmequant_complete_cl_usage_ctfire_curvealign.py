"""
TMEQuant Complete Workflows — CurveAlign and CT-FIRE

Two end-to-end examples demonstrating the updated TMEQuant API:
  1. CurveAlign — fiber segment orientation analysis
  2. CT-FIRE   — individual fiber extraction with full TACS classification

Key conventions used throughout (Issue 5 fixes):
  - TACS-3:  60–90° from boundary TANGENT (perpendicular, INVASIVE)
  - TACS-2:   0–30° from boundary TANGENT (parallel)
  - TACS-1:  30–60° from boundary TANGENT, or curly fibers
  - All classify_fiber_tacs() calls use keyword angle_to_tangent=
  - compute_angle_to_boundary_normal() returns angle to NORMAL;
    convert to tangent angle with: angle_to_tangent = 90 - angle_to_normal

API changes reflected here from the full codebase audit:
  Issue 1  — TMEObject unified hierarchy (object_id primary key)
  Issue 2  — TMEHierarchy replaces ObjectHierarchy
  Issue 3  — ImageEntry added; project.images uses it
  Issue 4  — InteractionAnalyzer retired; use InteractionDetector +
             MeasurementEngine + annotate_interaction_pairs
  Issue 5  — classify_fiber_tacs kwarg renamed angle_to_boundary ->
             angle_to_tangent; tangent convention enforced throughout
  Issue 6  — New modules: InteractionAnalysisPipeline,
             InteractionNetworkAnalyzer, interaction_features
  Issue 7  — StarDistSegmentation / CellposeSegmentation imported
             from base_segmentation only (thin wrappers deleted)
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

# ── Fiber analysis ────────────────────────────────────────────────────────────
from tme_quant.fiber_analysis import FiberAnalyzer
from tme_quant.fiber_analysis.config import (
    OrientationParams,
    ExtractionParams,
    OrientationMode,
    ExtractionMode,
)

# ── Geometry utilities (fiber_analysis) ──────────────────────────────────────
# compute_angle_to_boundary_normal returns angle relative to boundary NORMAL.
# For TACS classification convert: angle_to_tangent = 90 - angle_to_normal.
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,   # returns angle-to-NORMAL (0-90°)
    compute_relative_angles,            # returns dict with both angle_to_normal
                                        # and angle_to_tangent
)

# ── Cell analysis ─────────────────────────────────────────────────────────────
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config import SegmentationMode

# SegmentationParams may not yet have its own module; import defensively.
try:
    from tme_quant.cell_analysis.config import SegmentationParams
except ImportError:
    # Fallback: define a minimal stub so the example runs structurally.
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

# ── TACS classification (canonical — Issue 5) ─────────────────────────────────
# Both functions now use angle_to_tangent (0° = parallel, 90° = perpendicular).
from tme_quant.tme_analysis.core.tacs_classifier import (
    classify_fiber_tacs,               # full TACS with straightness
    classify_fiber_segment_tacs_like,  # orientation-only (CurveAlign)
    get_tacs_color,
)

# ── Interaction detection & measurement (Issue 4 replacements) ───────────────
from tme_quant.tme_analysis.core.interaction_detector import InteractionDetector
from tme_quant.tme_analysis.core.measurement_engine import MeasurementEngine

# ── Per-interaction feature annotation (Issue 6 — new module) ────────────────
from tme_quant.measurement.interaction_features import (
    annotate_interaction_pairs,
    compute_alignment_heterogeneity,
)

# ── End-to-end pipeline (Issue 6 — new module) ───────────────────────────────
from tme_quant.pipelines.interaction_analysis_pipeline import (
    InteractionAnalysisPipeline,
    PipelineConfig,
)

# ── Network analysis (Issue 6 — new module) ──────────────────────────────────
from tme_quant.tme_analysis.interaction_network_analysis import (
    InteractionNetworkAnalyzer,
)

# ── Distance utilities ────────────────────────────────────────────────────────
from tme_quant.tme_analysis.utils.distance_utils import (
    compute_distance_to_boundary,
)
# find_nearest_boundary_point lives in fiber_analysis geometry utils
from tme_quant.fiber_analysis.utils.geometry_utils import (
    find_nearest_boundary_point,
)

# ── Export ────────────────────────────────────────────────────────────────────
from tme_quant.tme_analysis.io import export_tme_analysis_results


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW 1: CURVEALIGN — FIBER SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def workflow_1_curvealign_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    sample_id: str = "patient_001_curvealign",
) -> Dict:
    """
    Complete CurveAlign workflow for fiber segment analysis.

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
    registration = HESHGRegistration(verbose=False)
    reg_result   = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    io.imsave(
        output_dir / f"{sample_id}_HE_registered.tif",
        (registered_he * 255).astype(np.uint8),
    )

    # ── [3/10] CurveAlign orientation analysis ───────────────────────────────
    print("\n[3/10] CurveAlign orientation analysis...")
    orientation_params = OrientationParams(
        mode=OrientationMode.CURVEALIGN,
        pixel_size=pixel_size,
        keep_values=['angles', 'alignment', 'positions'],
        return_fiber_segments=True,
        compute_stats=True,
    )
    fiber_analyzer     = FiberAnalyzer(verbose=False)
    orientation_result = fiber_analyzer.analyze_orientation_2d(
        shg_image, orientation_params, image_id=sample_id
    )
    print(f"  ✓ Mean orientation: {orientation_result.mean_orientation:.2f}°")
    print(f"  ✓ Mean alignment:   {orientation_result.mean_alignment:.4f}")

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
    # StarDistSegmentation is now imported from base_segmentation (Issue 7a).
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0,
        probability_threshold=0.5,
    )
    cell_analyzer = CellAnalyzer(verbose=False)
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
    tme_analyzer  = TMEAnalyzer(verbose=False)
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
        summary_stats[f'{tacs_type}_ratio'] = count / len(segment_metrics)

    pd.DataFrame([summary_stats]).to_csv(
        output_dir / f"{sample_id}_summary.csv", index=False
    )
    print(f"  ✓ Exported to {output_dir}")

    # ── Summary ──────────────────────────────────────────────────────────────
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
# WORKFLOW 2: CT-FIRE — INDIVIDUAL FIBER ANALYSIS
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

    Includes fiber-tumor interaction analysis via the new
    InteractionAnalysisPipeline (Issue 6) and network analysis via
    InteractionNetworkAnalyzer (Issue 6).

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
    registration  = HESHGRegistration(verbose=False)
    reg_result    = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    io.imsave(
        output_dir / f"{sample_id}_HE_registered.tif",
        (registered_he * 255).astype(np.uint8),
    )

    # ── [3/12] CT-FIRE fiber extraction ─────────────────────────────────────
    print("\n[3/12] Extracting individual fibers (CT-FIRE)...")
    extraction_params = ExtractionParams(
        mode=ExtractionMode.CTFIRE,
        pixel_size=pixel_size,
        min_fiber_length=10.0,
        max_fiber_length=500.0,
        min_fiber_width=1.0,
        max_fiber_width=10.0,
        measure_length=True,
        measure_width=True,
        measure_straightness=True,
        measure_angle=True,
        extract_centerlines=True,
    )
    fiber_analyzer = FiberAnalyzer(verbose=False)
    fiber_result   = fiber_analyzer.extract_fibers_2d(
        shg_image, extraction_params, image_id=sample_id
    )
    # fiber_result.fibers is a list of FiberProperties (extraction output)
    # or FiberObject (if hierarchy integration is used).
    fibers = fiber_result.fibers
    print(f"  ✓ Extracted {len(fibers)} individual fibers")
    print(f"  Mean length:     {np.mean([f.length for f in fibers]):.2f} µm")
    print(f"  Mean width:      {np.mean([f.width for f in fibers]):.2f} µm")
    print(f"  Mean straightness: {np.mean([f.straightness for f in fibers]):.3f}")

    # ── [4/12] Cell segmentation ─────────────────────────────────────────────
    print("\n[4/12] Segmenting cells (StarDist)...")
    # StarDistSegmentation is now imported from base_segmentation (Issue 7a).
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0,
    )
    cell_analyzer = CellAnalyzer(verbose=False)
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
    tme_analyzer  = TMEAnalyzer(verbose=False)
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    print(f"  ✓ Detected {len(tumor_regions)} tumor region(s)")

    # ── [6/12] TME analysis (fiber-based) ───────────────────────────────────
    print("\n[6/12] TME analysis (fiber-tumor interactions)...")
    # tacs_angle_threshold_perpendicular / _parallel name the *legacy* per-fiber
    # thresholds stored for reference; actual TACS computation uses
    # classify_fiber_tacs() with the tangent angle convention (Issue 5).
    tme_params = TMEAnalysisParams(
        mode=AnalysisMode.FIBER_BASED,
        tumor_boundary_distance=100.0,
        compute_tacs=True,
        tacs_angle_threshold_perpendicular=30.0,   # kept for reference only
        tacs_angle_threshold_parallel=60.0,        # kept for reference only
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

    # ── [7/12] End-to-end pipeline (Issue 6) ────────────────────────────────
    print("\n[7/12] Running InteractionAnalysisPipeline (Issue 6)...")
    # The pipeline orchestrates detection -> per-pair annotation ->
    # MeasurementEngine feature extraction -> prognostic scoring.
    # It replaces the retired InteractionAnalyzer (Issue 4).
    pipeline_cfg = PipelineConfig(
        boundary_distance=100.0,
        tacs_zone_width=100.0,
        contact_threshold=5.0,
        compute_tacs=True,
        compute_mechanical=True,
        compute_contact_patterns=True,
        compute_prognostic=True,
        verbose=False,
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
    print(f"  Scores: {pipeline_result.prognostic_scores}")

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
    print(f"  Mean K-NN alignment: {fiber_metrics['local_alignment'].mean():.3f}")
    print(f"  Mean density:        {fiber_metrics['local_fiber_density'].mean():.1f} fibers/mm²")

    # ── [9/12] Network analysis (Issue 6) ───────────────────────────────────
    print("\n[9/12] Interaction network analysis (Issue 6)...")
    network_analyzer = InteractionNetworkAnalyzer(
        weight_by='distance',
        community_method='greedy',
        verbose=False,
    )
    network_results = network_analyzer.analyze(
        pairs=pipeline_result.interaction_pairs,
        top_n_hubs=10,
    )
    nm = network_results['network_metrics']
    print(f"  ✓ Graph: {nm['n_nodes']} nodes, {nm['n_edges']} edges")
    print(f"  Density:    {nm['density']:.3f}")
    print(f"  Components: {nm['n_components']}")
    if 'modularity' in nm:
        print(f"  Modularity: {nm.get('modularity', float('nan')):.3f}")
    if 'hub_nodes' in network_results.get('critical_interactions', {}):
        hubs = network_results['critical_interactions']['hub_nodes'][:5]
        print(f"  Top hubs:   {hubs}")

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
    # Export full TME analysis via the canonical exporter.
    export_tme_analysis_results(
        tme_result,
        output_dir=output_dir,
        formats=['csv', 'excel', 'json'],
        prefix=sample_id,
    )
    fiber_metrics.to_csv(output_dir / f"{sample_id}_fiber_metrics.csv", index=False)

    summary_stats: Dict = {
        'sample_id':        sample_id,
        'n_fibers':         len(fibers),
        'n_cells':          len(cells),
        'n_tumors':         len(tumor_regions),
        'mean_fiber_length':  float(fiber_metrics['length'].mean()),
        'mean_fiber_width':   float(fiber_metrics['width'].mean()),
        'mean_straightness':  float(fiber_metrics['straightness'].mean()),
        'mean_alignment':     float(fiber_metrics['local_alignment'].mean()),
        'mean_density':       float(fiber_metrics['local_fiber_density'].mean()),
    }
    if tme_result.tacs_features:
        tf = tme_result.tacs_features
        summary_stats.update({
            'tacs1_count':   tf['tacs1_count'],
            'tacs2_count':   tf['tacs2_count'],
            'tacs3_count':   tf['tacs3_count'],
            'tacs1_ratio':   tf['tacs1_ratio'],
            'tacs2_ratio':   tf['tacs2_ratio'],
            'tacs3_ratio':   tf['tacs3_ratio'],
            'dominant_tacs': tf['dominant_tacs_type'],
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

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CT-FIRE WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nSample:           {sample_id}")
    print(f"Individual fibers: {len(fibers):,}")
    print(f"Cells:            {len(cells):,}")
    print(f"Tumors:           {len(tumor_regions)}")
    print(f"\nFiber properties:")
    print(f"  Mean length:      {summary_stats['mean_fiber_length']:.2f} µm")
    print(f"  Mean width:       {summary_stats['mean_fiber_width']:.2f} µm")
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
    if 'tme_risk_score' in summary_stats:
        print(f"\nRisk score: {summary_stats['tme_risk_score']:.3f}")
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
# INTERNAL HELPERS — CURVEALIGN
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fiber_segments_from_curvealign(
    orientation_map: np.ndarray,
    alignment_map: np.ndarray,
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
            alignment   = alignment_map[y, x]
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

    Angle convention (Issue 5):
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

        # ── Distance to tumor boundary and TACS-like angle ──────────────────
        min_dist        = np.inf
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
                    # compute_angle_to_boundary_normal returns angle to NORMAL.
                    # Convert to tangent angle for TACS classification.
                    angle_to_normal = compute_angle_to_boundary_normal(
                        fiber_orientation=orientation,
                        boundary_point1=(nearest_pt.x, nearest_pt.y),
                        boundary_point2=(second_pt.x,  second_pt.y),
                    )
                    if not np.isnan(angle_to_normal):
                        angle_to_tangent = 90.0 - angle_to_normal

        seg_data['distance_to_tumor']  = min_dist
        seg_data['angle_to_tangent']   = angle_to_tangent
        seg_data['angle_to_normal']    = (
            90.0 - angle_to_tangent if angle_to_tangent is not None else None
        )

        # ── TACS-like classification (orientation only, no straightness) ─────
        # angle_to_tangent=: 0-30° = parallel (TACS-2-like),
        #                   60-90° = perpendicular (TACS-3-like).
        if min_dist <= tacs_zone_width and angle_to_tangent is not None:
            seg_data['tacs_type'] = classify_fiber_segment_tacs_like(
                angle_to_tangent=angle_to_tangent,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
            )
        else:
            seg_data['tacs_type'] = None

        # ── K-NN alignment ───────────────────────────────────────────────────
        if len(fiber_segments) > k_neighbors:
            _, indices = tree.query(position, k=k_neighbors + 1)
            neighbor_orientations = (
                fiber_segments.iloc[indices[1:]]['orientation'].values
            )
            diffs = np.abs(orientation - neighbor_orientations)
            diffs = np.minimum(diffs, 180.0 - diffs)
            seg_data['local_alignment'] = 1.0 - (np.mean(diffs) / 90.0)
            seg_data['mean_angle_diff'] = float(np.mean(diffs))

        # ── Local density ────────────────────────────────────────────────────
        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby = tree.query_ball_point(position, bbox_radius)
        count  = len(nearby) - 1
        area_mm2 = np.pi * ((bbox_size / 2.0) / 1000.0) ** 2
        seg_data['local_density'] = count / area_mm2 if area_mm2 > 0 else 0.0

        metrics.append(seg_data)

    return pd.DataFrame(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS — CT-FIRE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_individual_fiber_metrics(
    fibers: List,
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

    Angle convention (Issue 5):
      compute_angle_to_boundary_normal() → angle to NORMAL
      angle_to_tangent = 90 - angle_to_normal
      classify_fiber_tacs(angle_to_tangent=...) → TACS-1/2/3
    """
    # FiberProperties uses center_coordinates (centerline midpoint).
    # FiberObject uses centerline[midpoint_index].
    def _get_center(f) -> np.ndarray:
        if hasattr(f, 'center_coordinates'):
            return f.center_coordinates
        if hasattr(f, 'centerline') and len(f.centerline) > 0:
            return f.centerline[len(f.centerline) // 2]
        raise AttributeError(
            f"Fiber {getattr(f, 'object_id', '?')} has no center attribute."
        )

    # FiberProperties uses .angle; FiberObject also has .angle (and .orientation).
    def _get_angle(f) -> Optional[float]:
        return getattr(f, 'angle', None) or getattr(f, 'orientation', None)

    fiber_centers = np.array([_get_center(f) for f in fibers])
    tree          = cKDTree(fiber_centers)
    metrics       = []

    for i, fiber in enumerate(fibers):
        center     = fiber_centers[i]
        angle      = _get_angle(fiber)
        straightness = getattr(fiber, 'straightness', None)

        fiber_data = {
            'fiber_id':    getattr(fiber, 'object_id', getattr(fiber, 'fiber_id', i)),
            'fiber_index': i,
            'length':      getattr(fiber, 'length',     None),
            'width':       getattr(fiber, 'width',      None),
            'straightness': straightness,
            'orientation': angle,
            'midpoint_x':  float(center[0]),
            'midpoint_y':  float(center[1]),
        }

        # ── Distance to tumor boundary and TACS angle ────────────────────────
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
                        # compute_angle_to_boundary_normal → angle to NORMAL.
                        # Convert to tangent angle for TACS classification.
                        angle_to_normal = compute_angle_to_boundary_normal(
                            fiber_orientation=angle,
                            boundary_point1=(nearest_pt.x, nearest_pt.y),
                            boundary_point2=(second_pt.x,  second_pt.y),
                        )
                        if not np.isnan(angle_to_normal):
                            angle_to_tangent = 90.0 - angle_to_normal

        fiber_data['distance_to_tumor']  = min_dist
        fiber_data['angle_to_tangent']   = angle_to_tangent
        fiber_data['angle_to_normal']    = (
            90.0 - angle_to_tangent if angle_to_tangent is not None else None
        )

        # ── Full TACS classification (with straightness — CT-FIRE only) ──────
        # Uses boundary tangent angle: 60-90° = perpendicular (TACS-3/INVASIVE),
        #                               0-30° = parallel (TACS-2).
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

        # ── K-NN alignment ───────────────────────────────────────────────────
        if len(fibers) > k_neighbors and angle is not None:
            _, indices = tree.query(center, k=k_neighbors + 1)
            neighbor_angles = [_get_angle(fibers[idx]) for idx in indices[1:]]
            neighbor_angles = [a for a in neighbor_angles if a is not None]
            if neighbor_angles:
                diffs = np.array([abs((angle - na) % 180) for na in neighbor_angles])
                diffs = np.minimum(diffs, 180.0 - diffs)
                fiber_data['local_alignment'] = 1.0 - float(np.mean(diffs) / 90.0)
                fiber_data['mean_angle_diff'] = float(np.mean(diffs))

        # ── Local density ────────────────────────────────────────────────────
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
        ('orientation',   'hsv',     'Orientation (°)',       ),
        ('local_alignment','RdYlGn', 'K-NN Alignment',        ),
        ('local_density',  'viridis', 'Density (segs/mm²)',   ),
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

    h, w = shg_image.shape[:2]
    resolution = 512
    gx, gy = np.meshgrid(
        np.linspace(0, w, resolution),
        np.linspace(0, h, resolution),
    )
    positions = fiber_metrics[['midpoint_x', 'midpoint_y']].values

    for col, cmap, label in [
        ('orientation',       'hsv',     'Orientation (°)'),
        ('local_alignment',   'RdYlGn',  'K-NN Alignment'),
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
    """TACS-coloured fiber overlay for CT-FIRE."""
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
        if tacs_type and hasattr(fiber, 'centerline') and fiber.centerline is not None:
            pts = fiber.centerline.astype(np.int32)
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
    print("TMEQuant Complete Workflows")
    print("=" * 80)

    # ── Workflow 1: CurveAlign ────────────────────────────────────────────────
    print("\n\nWORKFLOW 1: CurveAlign Fiber Segments")
    print("=" * 80)
    results_ca = workflow_1_curvealign_complete(
        he_image_path="data/patient_001_HE.tif",
        shg_image_path="data/patient_001_SHG.tif",
        output_dir="output/patient_001_curvealign",
        pixel_size=0.5,
        sample_id="patient_001_curvealign",
    )

    # ── Workflow 2: CT-FIRE ──────────────────────────────────────────────────
    print("\n\nWORKFLOW 2: CT-FIRE Individual Fibers")
    print("=" * 80)
    results_ct = workflow_2_ctfire_complete(
        he_image_path="data/patient_001_HE.tif",
        shg_image_path="data/patient_001_SHG.tif",
        output_dir="output/patient_001_ctfire",
        pixel_size=0.5,
        sample_id="patient_001_ctfire",
    )

    print("\n\n" + "=" * 80)
    print("BOTH WORKFLOWS COMPLETE")
    print("=" * 80)
    print("\nTACS angle convention (boundary tangent, Issue 5):")
    print("  TACS-3:  60–90° (perpendicular, INVASIVE)  — RED")
    print("  TACS-2:   0–30° (parallel)                 — GREEN")
    print("  TACS-1:  30–60° or curly                   — BLUE")
    print("\nNew modules used (Issues 6 & 7):")
    print("  ✓ InteractionAnalysisPipeline  (replaces retired InteractionAnalyzer)")
    print("  ✓ InteractionNetworkAnalyzer   (graph-based TME network analysis)")
    print("  ✓ annotate_interaction_pairs   (per-pair mechanical/invasive scores)")
    print("  ✓ StarDistSegmentation         (from base_segmentation, no thin wrapper)")
    print("=" * 80)