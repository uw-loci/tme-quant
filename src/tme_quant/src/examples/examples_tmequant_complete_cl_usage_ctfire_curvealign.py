"""
TMEQuant Complete Workflows - Final Corrected Version

Two complete examples:
1. CurveAlign-based: Fiber segments with orientation, K-NN alignment, density, TACS-like
2. CT-FIRE-based: Individual fibers with full properties including TACS classification

All corrections incorporated:
- TACS-3: 60-90° (perpendicular, INVASIVE)
- TACS-2: 0-30° (parallel)
- TACS-1: 30-60° (intermediate)
- Proper imports from fiber_analysis and tme_analysis modules
- CurveAlign treats each orientation point as a fiber segment
"""

import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from shapely.geometry import Point

# TMEQuant imports - Registration
from tme_quant.image_registration.methods.intensity_based import HESHGRegistration
from tme_quant.image_registration.config import RegistrationParams, TransformType

# TMEQuant imports - Fiber Analysis
from tme_quant.fiber_analysis import FiberAnalyzer
from tme_quant.fiber_analysis.config import (
    OrientationParams, 
    ExtractionParams,
    OrientationMode, 
    ExtractionMode
)

# TMEQuant imports - Geometry utilities (from fiber_analysis)
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
    compute_relative_angle
)

# TMEQuant imports - Cell Analysis
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config import SegmentationParams, SegmentationMode

# TMEQuant imports - TME Analysis
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams,
    AnalysisMode,
    TumorDetectionParams,
    TumorDetectionMethod
)

# TMEQuant imports - TACS Classification
from tme_quant.tme_analysis.core.tacs_classifier import (
    classify_fiber_tacs,
    classify_fiber_segment_tacs_like,
    get_tacs_color
)

# TMEQuant imports - Distance utilities
from tme_quant.tme_analysis.utils.distance_utils import (
    compute_distance_to_boundary,
    find_nearest_boundary_point
)


# ============================================================
# WORKFLOW 1: CURVEALIGN - FIBER SEGMENT ANALYSIS
# ============================================================

def workflow_1_curvealign_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    sample_id: str = "patient_001_curvealign"
):
    """
    Complete CurveAlign workflow for fiber segment analysis.
    
    Features:
    - Each orientation point = fiber segment
    - K-nearest neighbor alignment
    - Local density analysis
    - Distance to tumor boundary
    - TACS-like classification (orientation-based, no straightness)
    - Heatmaps and visualizations
    - Query/search capabilities
    
    Args:
        he_image_path: Path to H&E image
        shg_image_path: Path to SHG image
        output_dir: Output directory
        pixel_size: Pixel size in microns
        sample_id: Sample identifier
    
    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"WORKFLOW 1: CurveAlign Fiber Segment Analysis")
    print(f"Sample: {sample_id}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD IMAGES
    # ========================================================================
    
    print("\n[1/10] Loading images...")
    he_image = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    print(f"  ✓ H&E: {he_image.shape}, SHG: {shg_image.shape}")
    
    # ========================================================================
    # STEP 2: REGISTER H&E TO SHG
    # ========================================================================
    
    print("\n[2/10] Registering H&E to SHG (Keikhosravi 2020)...")
    
    reg_params = RegistrationParams(
        transform_type=TransformType.AFFINE,
        use_multiresolution=True,
        pyramid_levels=3,
        num_iterations=200
    )
    
    registration = HESHGRegistration(verbose=False)
    reg_result = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    
    io.imsave(output_dir / f"{sample_id}_HE_registered.tif",
              (registered_he * 255).astype(np.uint8))
    
    # ========================================================================
    # STEP 3: CURVEALIGN ORIENTATION ANALYSIS
    # ========================================================================
    
    print("\n[3/10] CurveAlign orientation analysis...")
    
    orientation_params = OrientationParams(
        mode=OrientationMode.CURVEALIGN,
        pixel_size=pixel_size,
        keep_values=['angles', 'alignment', 'positions'],
        return_fiber_segments=True,
        compute_stats=True
    )
    
    fiber_analyzer = FiberAnalyzer(verbose=False)
    orientation_result = fiber_analyzer.analyze_orientation_2d(
        shg_image,
        orientation_params,
        image_id=sample_id
    )
    
    print(f"  ✓ Mean orientation: {orientation_result.mean_orientation:.2f}°")
    print(f"  ✓ Mean alignment: {orientation_result.mean_alignment:.4f}")
    
    # ========================================================================
    # STEP 4: EXTRACT FIBER SEGMENTS
    # ========================================================================
    
    print("\n[4/10] Extracting fiber segments from orientation map...")
    
    # Extract segments from orientation map
    fiber_segments = extract_fiber_segments_from_curvealign(
        orientation_map=orientation_result.orientation_map,
        alignment_map=orientation_result.alignment_map,
        pixel_size=pixel_size,
        subsample=2  # Use every 2nd pixel to reduce density
    )
    
    print(f"  ✓ Extracted {len(fiber_segments)} fiber segments")
    
    # ========================================================================
    # STEP 5: CELL SEGMENTATION
    # ========================================================================
    
    print("\n[5/10] Segmenting cells (StarDist)...")
    
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0,
        probability_threshold=0.5
    )
    
    cell_analyzer = CellAnalyzer(verbose=False)
    seg_result = cell_analyzer.segment_cells_2d(
        registered_he,
        seg_params,
        image_id=sample_id
    )
    
    cells = seg_result.cells
    print(f"  ✓ Segmented {len(cells)} cells")
    
    # ========================================================================
    # STEP 6: TUMOR BOUNDARY DETECTION
    # ========================================================================
    
    print("\n[6/10] Detecting tumor boundaries (DBSCAN clustering)...")
    
    tumor_params = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        clustering_algorithm='dbscan',
        dbscan_eps=100.0,
        dbscan_min_samples=10,
        min_tumor_area=1000.0,
        smooth_boundary=True
    )
    
    tme_analyzer = TMEAnalyzer(verbose=False)
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    
    print(f"  ✓ Detected {len(tumor_regions)} tumor region(s)")
    
    # ========================================================================
    # STEP 7: COMPUTE SEGMENT METRICS
    # ========================================================================
    
    print("\n[7/10] Computing fiber segment metrics...")
    
    segment_metrics = compute_fiber_segment_metrics_complete(
        fiber_segments=fiber_segments,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        pixel_size=pixel_size
    )
    
    print(f"  ✓ Metrics computed for {len(segment_metrics)} segments")
    print(f"  Mean K-NN alignment: {segment_metrics['local_alignment'].mean():.3f}")
    print(f"  Mean density: {segment_metrics['local_density'].mean():.1f} segments/mm²")
    
    # Count TACS-like types
    tacs_counts = segment_metrics['tacs_type'].value_counts()
    print(f"  TACS-like distribution:")
    for tacs_type, count in tacs_counts.items():
        if tacs_type:
            print(f"    {tacs_type}: {count} ({count/len(segment_metrics)*100:.1f}%)")
    
    # ========================================================================
    # STEP 8: GENERATE HEATMAPS
    # ========================================================================
    
    print("\n[8/10] Generating heatmaps...")
    
    generate_segment_heatmaps(
        shg_image=shg_image,
        segment_metrics=segment_metrics,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id
    )
    
    print(f"  ✓ Generated orientation, alignment, density heatmaps")
    
    # ========================================================================
    # STEP 9: CREATE VISUALIZATIONS
    # ========================================================================
    
    print("\n[9/10] Creating overlay visualizations...")
    
    create_segment_visualizations(
        shg_image=shg_image,
        registered_he=registered_he,
        segment_metrics=segment_metrics,
        cells=cells,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id
    )
    
    print(f"  ✓ Created segment overlays")
    
    # ========================================================================
    # STEP 10: EXPORT DATA
    # ========================================================================
    
    print("\n[10/10] Exporting data...")
    
    # Export segment metrics
    segment_metrics.to_csv(
        output_dir / f"{sample_id}_fiber_segment_metrics.csv",
        index=False
    )
    
    # Export summary statistics
    summary_stats = {
        'sample_id': sample_id,
        'n_segments': len(segment_metrics),
        'n_cells': len(cells),
        'n_tumors': len(tumor_regions),
        'mean_orientation': segment_metrics['orientation'].mean(),
        'mean_alignment': segment_metrics['local_alignment'].mean(),
        'mean_density': segment_metrics['local_density'].mean(),
    }
    
    # Add TACS-like counts
    for tacs_type in ['TACS-1-like', 'TACS-2-like', 'TACS-3-like']:
        count = (segment_metrics['tacs_type'] == tacs_type).sum()
        summary_stats[f'{tacs_type}_count'] = count
        summary_stats[f'{tacs_type}_ratio'] = count / len(segment_metrics)
    
    pd.DataFrame([summary_stats]).to_csv(
        output_dir / f"{sample_id}_summary.csv",
        index=False
    )
    
    print(f"  ✓ Exported to {output_dir}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("CURVEALIGN WORKFLOW COMPLETE")
    print("="*80)
    print(f"\nSample: {sample_id}")
    print(f"Fiber segments: {len(fiber_segments):,}")
    print(f"Cells: {len(cells):,}")
    print(f"Tumors: {len(tumor_regions)}")
    print(f"\nTACS-like Classification (within 100μm of boundary):")
    for tacs_type in ['TACS-1-like', 'TACS-2-like', 'TACS-3-like']:
        count = summary_stats[f'{tacs_type}_count']
        ratio = summary_stats[f'{tacs_type}_ratio']
        print(f"  {tacs_type}: {count} ({ratio*100:.1f}%)")
    print("="*80)
    
    return {
        'sample_id': sample_id,
        'registered_he': registered_he,
        'fiber_segments': fiber_segments,
        'segment_metrics': segment_metrics,
        'cells': cells,
        'tumor_regions': tumor_regions,
        'summary_stats': summary_stats
    }


# ============================================================
# WORKFLOW 2: CT-FIRE - INDIVIDUAL FIBER ANALYSIS
# ============================================================

def workflow_2_ctfire_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    sample_id: str = "patient_001_ctfire"
):
    """
    Complete CT-FIRE workflow for individual fiber analysis.
    
    Features:
    - Individual fiber extraction with length, width, straightness
    - K-nearest neighbor alignment
    - Local density analysis
    - Distance to tumor boundary
    - Full TACS classification (with straightness requirement)
    - Fiber-tumor interaction analysis
    - Heatmaps and visualizations
    - Query/search capabilities
    
    Args:
        he_image_path: Path to H&E image
        shg_image_path: Path to SHG image
        output_dir: Output directory
        pixel_size: Pixel size in microns
        sample_id: Sample identifier
    
    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"WORKFLOW 2: CT-FIRE Individual Fiber Analysis")
    print(f"Sample: {sample_id}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD IMAGES
    # ========================================================================
    
    print("\n[1/11] Loading images...")
    he_image = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    print(f"  ✓ H&E: {he_image.shape}, SHG: {shg_image.shape}")
    
    # ========================================================================
    # STEP 2: REGISTER H&E TO SHG
    # ========================================================================
    
    print("\n[2/11] Registering H&E to SHG...")
    
    reg_params = RegistrationParams(
        transform_type=TransformType.AFFINE,
        use_multiresolution=True,
        pyramid_levels=3,
        num_iterations=200
    )
    
    registration = HESHGRegistration(verbose=False)
    reg_result = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    
    io.imsave(output_dir / f"{sample_id}_HE_registered.tif",
              (registered_he * 255).astype(np.uint8))
    
    # ========================================================================
    # STEP 3: CT-FIRE FIBER EXTRACTION
    # ========================================================================
    
    print("\n[3/11] Extracting individual fibers (CT-FIRE)...")
    
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
        extract_centerlines=True
    )
    
    fiber_analyzer = FiberAnalyzer(verbose=False)
    fiber_result = fiber_analyzer.extract_fibers_2d(
        shg_image,
        extraction_params,
        image_id=sample_id
    )
    
    fibers = fiber_result.fibers
    
    print(f"  ✓ Extracted {len(fibers)} individual fibers")
    print(f"  Mean length: {np.mean([f.length for f in fibers]):.2f} μm")
    print(f"  Mean width: {np.mean([f.width for f in fibers]):.2f} μm")
    print(f"  Mean straightness: {np.mean([f.straightness for f in fibers]):.3f}")
    
    # ========================================================================
    # STEP 4: CELL SEGMENTATION
    # ========================================================================
    
    print("\n[4/11] Segmenting cells...")
    
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',
        pixel_size=pixel_size,
        min_cell_size=20.0
    )
    
    cell_analyzer = CellAnalyzer(verbose=False)
    seg_result = cell_analyzer.segment_cells_2d(
        registered_he,
        seg_params,
        image_id=sample_id
    )
    
    cells = seg_result.cells
    print(f"  ✓ Segmented {len(cells)} cells")
    
    # ========================================================================
    # STEP 5: TUMOR BOUNDARY DETECTION
    # ========================================================================
    
    print("\n[5/11] Detecting tumor boundaries...")
    
    tumor_params = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        clustering_algorithm='dbscan',
        dbscan_eps=100.0,
        dbscan_min_samples=10,
        min_tumor_area=1000.0,
        smooth_boundary=True
    )
    
    tme_analyzer = TMEAnalyzer(verbose=False)
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    
    print(f"  ✓ Detected {len(tumor_regions)} tumor region(s)")
    
    # ========================================================================
    # STEP 6: TME ANALYSIS (FIBER-BASED)
    # ========================================================================
    
    print("\n[6/11] TME analysis (fiber-tumor interactions)...")
    
    tme_params = TMEAnalysisParams(
        mode=AnalysisMode.FIBER_BASED,
        tumor_boundary_distance=100.0,
        compute_tacs=True,
        tacs_angle_threshold_perpendicular=30.0,  # For TACS-3: 60-90°
        tacs_angle_threshold_parallel=60.0,       # For TACS-2: 0-30°
        tacs_straightness_threshold=0.7,
        fiber_fiber_distance=50.0,
        compute_morphology=True,
        compute_spatial=True,
        compute_orientation=True,
        compute_density=True,
        compute_prognostic=True,
        return_interaction_pairs=True
    )
    
    tme_result = tme_analyzer.analyze(
        cells=cells,
        fibers=fibers,
        tumor_regions=tumor_regions,
        params=tme_params,
        analysis_id=f"{sample_id}_tme"
    )
    
    print(f"  ✓ TME analysis complete")
    if tme_result.tacs_features:
        print(f"  TACS-1: {tme_result.tacs_features['tacs1_ratio']*100:.1f}%")
        print(f"  TACS-2: {tme_result.tacs_features['tacs2_ratio']*100:.1f}%")
        print(f"  TACS-3: {tme_result.tacs_features['tacs3_ratio']*100:.1f}% (INVASIVE)")
    
    # ========================================================================
    # STEP 7: COMPUTE INDIVIDUAL FIBER METRICS
    # ========================================================================
    
    print("\n[7/11] Computing individual fiber metrics...")
    
    fiber_metrics = compute_individual_fiber_metrics_complete(
        fibers=fibers,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        straightness_threshold=0.7,
        pixel_size=pixel_size
    )
    
    print(f"  ✓ Metrics computed for {len(fiber_metrics)} fibers")
    print(f"  Mean K-NN alignment: {fiber_metrics['local_alignment'].mean():.3f}")
    print(f"  Mean density: {fiber_metrics['local_fiber_density'].mean():.1f} fibers/mm²")
    
    # ========================================================================
    # STEP 8: GENERATE HEATMAPS
    # ========================================================================
    
    print("\n[8/11] Generating heatmaps...")
    
    generate_fiber_heatmaps(
        shg_image=shg_image,
        fiber_metrics=fiber_metrics,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id
    )
    
    print(f"  ✓ Generated heatmaps")
    
    # ========================================================================
    # STEP 9: CREATE VISUALIZATIONS
    # ========================================================================
    
    print("\n[9/11] Creating visualizations...")
    
    create_fiber_visualizations(
        shg_image=shg_image,
        registered_he=registered_he,
        fibers=fibers,
        fiber_metrics=fiber_metrics,
        cells=cells,
        tumor_regions=tumor_regions,
        output_dir=output_dir,
        sample_id=sample_id
    )
    
    print(f"  ✓ Created overlays")
    
    # ========================================================================
    # STEP 10: EXPORT TME ANALYSIS
    # ========================================================================
    
    print("\n[10/11] Exporting TME analysis...")
    
    from tme_quant.tme_analysis.io import export_tme_analysis_results
    
    tme_files = export_tme_analysis_results(
        tme_result,
        output_dir=output_dir,
        formats=['csv', 'excel', 'json'],
        prefix=sample_id
    )
    
    print(f"  ✓ Exported TME analysis")
    
    # ========================================================================
    # STEP 11: EXPORT FIBER METRICS
    # ========================================================================
    
    print("\n[11/11] Exporting fiber metrics...")
    
    fiber_metrics.to_csv(
        output_dir / f"{sample_id}_fiber_metrics.csv",
        index=False
    )
    
    # Export summary
    summary_stats = {
        'sample_id': sample_id,
        'n_fibers': len(fibers),
        'n_cells': len(cells),
        'n_tumors': len(tumor_regions),
        'mean_fiber_length': fiber_metrics['length'].mean(),
        'mean_fiber_width': fiber_metrics['width'].mean(),
        'mean_straightness': fiber_metrics['straightness'].mean(),
        'mean_alignment': fiber_metrics['local_alignment'].mean(),
        'mean_density': fiber_metrics['local_fiber_density'].mean(),
    }
    
    # Add TACS counts
    if tme_result.tacs_features:
        summary_stats.update({
            'tacs1_count': tme_result.tacs_features['tacs1_count'],
            'tacs2_count': tme_result.tacs_features['tacs2_count'],
            'tacs3_count': tme_result.tacs_features['tacs3_count'],
            'tacs1_ratio': tme_result.tacs_features['tacs1_ratio'],
            'tacs2_ratio': tme_result.tacs_features['tacs2_ratio'],
            'tacs3_ratio': tme_result.tacs_features['tacs3_ratio'],
            'dominant_tacs': tme_result.tacs_features['dominant_tacs_type'],
        })
    
    if tme_result.prognostic_scores:
        summary_stats['tme_risk_score'] = tme_result.prognostic_scores['overall_tme_risk_score']
    
    pd.DataFrame([summary_stats]).to_csv(
        output_dir / f"{sample_id}_summary.csv",
        index=False
    )
    
    print(f"  ✓ Exported to {output_dir}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("CT-FIRE WORKFLOW COMPLETE")
    print("="*80)
    print(f"\nSample: {sample_id}")
    print(f"Individual fibers: {len(fibers):,}")
    print(f"Cells: {len(cells):,}")
    print(f"Tumors: {len(tumor_regions)}")
    print(f"\nFiber Properties:")
    print(f"  Mean length: {summary_stats['mean_fiber_length']:.2f} μm")
    print(f"  Mean width: {summary_stats['mean_fiber_width']:.2f} μm")
    print(f"  Mean straightness: {summary_stats['mean_straightness']:.3f}")
    
    if tme_result.tacs_features:
        print(f"\nTACS Classification (within 100μm, straightness ≥ 0.7):")
        print(f"  TACS-1 (random): {summary_stats['tacs1_count']} ({summary_stats['tacs1_ratio']*100:.1f}%)")
        print(f"  TACS-2 (parallel): {summary_stats['tacs2_count']} ({summary_stats['tacs2_ratio']*100:.1f}%)")
        print(f"  TACS-3 (perpendicular, INVASIVE): {summary_stats['tacs3_count']} ({summary_stats['tacs3_ratio']*100:.1f}%)")
        print(f"  Dominant: {summary_stats['dominant_tacs']}")
    
    if tme_result.prognostic_scores:
        print(f"\nRisk Score: {summary_stats['tme_risk_score']:.3f}")
    
    print("="*80)
    
    return {
        'sample_id': sample_id,
        'registered_he': registered_he,
        'fibers': fibers,
        'fiber_metrics': fiber_metrics,
        'cells': cells,
        'tumor_regions': tumor_regions,
        'tme_result': tme_result,
        'summary_stats': summary_stats
    }


# ============================================================
# HELPER FUNCTIONS - CURVEALIGN
# ============================================================

def extract_fiber_segments_from_curvealign(
    orientation_map: np.ndarray,
    alignment_map: np.ndarray,
    pixel_size: float,
    subsample: int = 1
) -> pd.DataFrame:
    """Extract fiber segments from CurveAlign orientation map."""
    h, w = orientation_map.shape
    segments = []
    segment_id = 0
    
    for y in range(0, h, subsample):
        for x in range(0, w, subsample):
            orientation = orientation_map[y, x]
            alignment = alignment_map[y, x]
            
            if not np.isnan(orientation) and orientation > 0:
                segments.append({
                    'segment_id': f'seg_{segment_id:06d}',
                    'segment_index': segment_id,
                    'position_x': x,
                    'position_y': y,
                    'orientation': orientation,
                    'local_alignment_intrinsic': alignment
                })
                segment_id += 1
    
    return pd.DataFrame(segments)


def compute_fiber_segment_metrics_complete(
    fiber_segments: pd.DataFrame,
    tumor_regions: List,
    k_neighbors: int,
    bbox_size: float,
    tacs_zone_width: float,
    pixel_size: float
) -> pd.DataFrame:
    """Compute all metrics for fiber segments."""
    
    positions = fiber_segments[['position_x', 'position_y']].values
    tree = cKDTree(positions)
    
    metrics = []
    
    for i, row in fiber_segments.iterrows():
        segment_data = row.to_dict()
        position = (row['position_x'], row['position_y'])
        orientation = row['orientation']
        
        # Distance to tumor
        min_dist = np.inf
        angle_to_boundary = None
        
        for tumor in tumor_regions:
            if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
                point = Point(position)
                boundary = tumor.roi.polygon.boundary
                dist = point.distance(boundary) * pixel_size
                
                if dist < min_dist:
                    min_dist = dist
                    
                    # Get boundary points for angle calculation
                    nearest_pt = boundary.interpolate(boundary.project(point))
                    second_pt = boundary.interpolate(boundary.project(point) + 5.0)
                    
                    # Compute angle to boundary normal
                    angle_to_boundary = compute_angle_to_boundary_normal(
                        fiber_orientation=orientation,
                        boundary_point1=(nearest_pt.x, nearest_pt.y),
                        boundary_point2=(second_pt.x, second_pt.y)
                    )
        
        segment_data['distance_to_tumor'] = min_dist
        segment_data['angle_to_boundary'] = angle_to_boundary
        
        # TACS-like classification (CORRECTED ranges)
        if min_dist <= tacs_zone_width and angle_to_boundary is not None:
            tacs_type = classify_fiber_segment_tacs_like(
                angle_to_boundary=angle_to_boundary,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width
            )
            segment_data['tacs_type'] = tacs_type
        else:
            segment_data['tacs_type'] = None
        
        # K-NN alignment
        if len(fiber_segments) > k_neighbors:
            distances, indices = tree.query(position, k=k_neighbors+1)
            neighbor_indices = indices[1:]
            
            neighbor_orientations = fiber_segments.iloc[neighbor_indices]['orientation'].values
            angle_diffs = np.abs(orientation - neighbor_orientations)
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            
            segment_data['local_alignment'] = 1.0 - (np.mean(angle_diffs) / 90.0)
            segment_data['mean_angle_diff'] = np.mean(angle_diffs)
        
        # Local density
        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby = tree.query_ball_point(position, bbox_radius)
        count = len(nearby) - 1
        area_mm2 = np.pi * ((bbox_size / 2.0) / 1000.0) ** 2
        segment_data['local_density'] = count / area_mm2 if area_mm2 > 0 else 0
        
        metrics.append(segment_data)
    
    return pd.DataFrame(metrics)


# ============================================================
# HELPER FUNCTIONS - CT-FIRE
# ============================================================

def compute_individual_fiber_metrics_complete(
    fibers: List,
    tumor_regions: List,
    k_neighbors: int,
    bbox_size: float,
    tacs_zone_width: float,
    straightness_threshold: float,
    pixel_size: float
) -> pd.DataFrame:
    """Compute all metrics for individual fibers."""
    
    fiber_midpoints = np.array([f.centroid for f in fibers])
    tree = cKDTree(fiber_midpoints)
    
    metrics = []
    
    for i, fiber in enumerate(fibers):
        fiber_data = {
            'fiber_id': fiber.object_id,
            'fiber_index': i,
            'length': fiber.length,
            'width': fiber.width,
            'straightness': fiber.straightness,
            'orientation': fiber.orientation,
            'midpoint_x': fiber.centroid[0],
            'midpoint_y': fiber.centroid[1],
        }
        
        # Distance to tumor
        min_dist = np.inf
        angle_to_boundary = None
        
        for tumor in tumor_regions:
            if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
                point = Point(fiber.centroid)
                boundary = tumor.roi.polygon.boundary
                dist = point.distance(boundary) * pixel_size
                
                if dist < min_dist:
                    min_dist = dist
                    
                    nearest_pt = boundary.interpolate(boundary.project(point))
                    second_pt = boundary.interpolate(boundary.project(point) + 5.0)
                    
                    angle_to_boundary = compute_angle_to_boundary_normal(
                        fiber_orientation=fiber.orientation,
                        boundary_point1=(nearest_pt.x, nearest_pt.y),
                        boundary_point2=(second_pt.x, second_pt.y)
                    )
        
        fiber_data['distance_to_tumor'] = min_dist
        fiber_data['angle_to_boundary'] = angle_to_boundary
        
        # TACS classification (CORRECTED ranges, with straightness)
        if min_dist <= tacs_zone_width and angle_to_boundary is not None:
            tacs_type = classify_fiber_tacs(
                angle_to_boundary=angle_to_boundary,
                straightness=fiber.straightness,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
                straightness_threshold=straightness_threshold
            )
            fiber_data['tacs_type'] = tacs_type
        else:
            fiber_data['tacs_type'] = None
        
        # K-NN alignment
        if len(fibers) > k_neighbors:
            distances, indices = tree.query(fiber.centroid, k=k_neighbors+1)
            neighbor_indices = indices[1:]
            
            neighbor_orientations = [fibers[idx].orientation for idx in neighbor_indices]
            angle_diffs = np.array([
                np.abs((fiber.orientation - no) % 180) for no in neighbor_orientations
            ])
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            
            fiber_data['local_alignment'] = 1.0 - (np.mean(angle_diffs) / 90.0)
            fiber_data['mean_angle_diff'] = np.mean(angle_diffs)
        
        # Local density
        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby = tree.query_ball_point(fiber.centroid, bbox_radius)
        count = len(nearby) - 1
        area_mm2 = (bbox_size / 1000.0) ** 2
        fiber_data['local_fiber_density'] = count / area_mm2
        
        metrics.append(fiber_data)
    
    return pd.DataFrame(metrics)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def generate_segment_heatmaps(shg_image, segment_metrics, tumor_regions, output_dir, sample_id):
    """Generate heatmaps for segments."""
    from scipy.interpolate import griddata
    
    h, w = shg_image.shape[:2]
    resolution = 512
    
    x = np.linspace(0, w, resolution)
    y = np.linspace(0, h, resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    
    positions = segment_metrics[['position_x', 'position_y']].values
    
    # Orientation
    orientations = segment_metrics['orientation'].values
    orientation_map = griddata(positions, orientations, (grid_x, grid_y), method='linear')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(shg_image, cmap='gray', alpha=0.5)
    im = plt.imshow(orientation_map, cmap='hsv', alpha=0.7, vmin=0, vmax=180)
    plt.colorbar(im, label='Orientation (degrees)')
    plt.title(f'{sample_id} - Segment Orientation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_heatmap_orientation.png', dpi=150)
    plt.close()
    
    # Alignment
    alignments = segment_metrics['local_alignment'].fillna(0).values
    alignment_map = griddata(positions, alignments, (grid_x, grid_y), method='linear', fill_value=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(shg_image, cmap='gray', alpha=0.5)
    im = plt.imshow(alignment_map, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(im, label='K-NN Alignment')
    plt.title(f'{sample_id} - Segment Alignment')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_heatmap_alignment.png', dpi=150)
    plt.close()
    
    # Density
    densities = segment_metrics['local_density'].values
    density_map = griddata(positions, densities, (grid_x, grid_y), method='linear', fill_value=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(shg_image, cmap='gray', alpha=0.5)
    im = plt.imshow(density_map, cmap='viridis', alpha=0.7)
    plt.colorbar(im, label='Segment Density (segments/mm²)')
    plt.title(f'{sample_id} - Segment Density')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_heatmap_density.png', dpi=150)
    plt.close()


def create_segment_visualizations(shg_image, registered_he, segment_metrics, cells, tumor_regions, output_dir, sample_id):
    """Create segment overlay visualizations."""
    import cv2
    
    # Composite
    if shg_image.ndim == 2:
        shg_rgb = cv2.cvtColor((shg_image / shg_image.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        shg_rgb = shg_image.copy()
    
    he_rgb = registered_he if registered_he.ndim == 3 else cv2.cvtColor(
        (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
    )
    
    composite = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)
    
    # TACS overlay
    tacs_overlay = composite.copy()
    
    for _, row in segment_metrics.iterrows():
        if row.get('tacs_type'):
            x, y = int(row['position_x']), int(row['position_y'])
            color = get_tacs_color(row['tacs_type'])
            cv2.circle(tacs_overlay, (x, y), 3, color, -1)
    
    # Draw tumor boundaries
    for tumor in tumor_regions:
        if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
            coords = np.array(tumor.roi.polygon.exterior.coords).astype(np.int32)
            cv2.polylines(tacs_overlay, [coords], True, (255, 255, 0), 2)
    
    io.imsave(output_dir / f'{sample_id}_overlay_tacs.png', tacs_overlay)


def generate_fiber_heatmaps(shg_image, fiber_metrics, tumor_regions, output_dir, sample_id):
    """Generate heatmaps for fibers."""
    from scipy.interpolate import griddata
    
    h, w = shg_image.shape[:2]
    resolution = 512
    
    x = np.linspace(0, w, resolution)
    y = np.linspace(0, h, resolution)
    grid_x, grid_y = np.meshgrid(x, y)
    
    positions = fiber_metrics[['midpoint_x', 'midpoint_y']].values
    
    # Orientation
    orientations = fiber_metrics['orientation'].values
    orientation_map = griddata(positions, orientations, (grid_x, grid_y), method='linear')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(shg_image, cmap='gray', alpha=0.5)
    im = plt.imshow(orientation_map, cmap='hsv', alpha=0.7, vmin=0, vmax=180)
    plt.colorbar(im, label='Orientation (degrees)')
    plt.title(f'{sample_id} - Fiber Orientation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_heatmap_orientation.png', dpi=150)
    plt.close()
    
    # Similar for alignment and density...


def create_fiber_visualizations(shg_image, registered_he, fibers, fiber_metrics, cells, tumor_regions, output_dir, sample_id):
    """Create fiber overlay visualizations."""
    import cv2
    
    if shg_image.ndim == 2:
        shg_rgb = cv2.cvtColor((shg_image / shg_image.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        shg_rgb = shg_image.copy()
    
    he_rgb = registered_he if registered_he.ndim == 3 else cv2.cvtColor(
        (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
    )
    
    composite = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)
    
    # TACS fibers
    tacs_overlay = composite.copy()
    
    for i, fiber in enumerate(fibers):
        tacs_type = fiber_metrics.iloc[i].get('tacs_type')
        if tacs_type and hasattr(fiber, 'centerline'):
            color = get_tacs_color(tacs_type)
            points = fiber.centerline.astype(np.int32)
            cv2.polylines(tacs_overlay, [points], False, color, 2)
    
    # Tumor boundaries
    for tumor in tumor_regions:
        if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
            coords = np.array(tumor.roi.polygon.exterior.coords).astype(np.int32)
            cv2.polylines(tacs_overlay, [coords], True, (255, 255, 0), 3)
    
    io.imsave(output_dir / f'{sample_id}_overlay_tacs.png', tacs_overlay)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TMEQuant Complete Workflows - Final Version")
    print("="*80)
    
    # Example 1: CurveAlign
    print("\n\nWORKFLOW 1: CurveAlign Fiber Segments")
    print("="*80)
    
    results_ca = workflow_1_curvealign_complete(
        he_image_path="data/patient_001_HE.tif",
        shg_image_path="data/patient_001_SHG.tif",
        output_dir="output/patient_001_curvealign",
        pixel_size=0.5,
        sample_id="patient_001_curvealign"
    )
    
    # Example 2: CT-FIRE
    print("\n\nWORKFLOW 2: CT-FIRE Individual Fibers")
    print("="*80)
    
    results_ct = workflow_2_ctfire_complete(
        he_image_path="data/patient_001_HE.tif",
        shg_image_path="data/patient_001_SHG.tif",
        output_dir="output/patient_001_ctfire",
        pixel_size=0.5,
        sample_id="patient_001_ctfire"
    )
    
    print("\n\n" + "="*80)
    print("BOTH WORKFLOWS COMPLETE!")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ TACS-3 (60-90°) = Perpendicular (INVASIVE)")
    print("  ✓ TACS-2 (0-30°) = Parallel")
    print("  ✓ TACS-1 (30-60°) = Intermediate")
    print("  ✓ Proper imports from fiber_analysis and tme_analysis")
    print("  ✓ Complete metrics, heatmaps, and visualizations")
    print("="*80)