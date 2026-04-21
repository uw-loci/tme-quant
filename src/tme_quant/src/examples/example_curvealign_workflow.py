"""
TMEQuant Example — Workflow 1: CurveAlign Fiber Segment Orientation (2-D)
=========================================================================

End-to-end demonstration of CurveAlign-based TACS fiber segment analysis:

  1.  Load H&E and SHG images.
  2.  Register H&E → SHG (affine, Keikhosravi 2020).
  3.  CurveAlign orientation analysis (curvelet transform, 2-D).
  4.  Extract fiber segments from the orientation map.
  5.  Segment cells from the registered H&E (StarDist).
  6.  Detect tumor boundaries (DBSCAN clustering).
  7.  Compute segment metrics: spatial, K-NN alignment, density, TACS-like.
  8.  Generate heatmaps (orientation, alignment, density) saved to disk.
  9.  Create TACS-coloured overlay saved to disk.
  10. Print measurement tables (summary stats + per-segment sample).
  11. Display a 2×2 figure panel: SHG | orientation heatmap |
                                  alignment heatmap | TACS overlay.

TACS-like angle convention (boundary tangent)
----------------------------------------------
  compute_angle_to_boundary_normal() → angle to NORMAL (0–90°).
  angle_to_tangent = 90° − angle_to_normal.
  classify_fiber_segment_tacs_like(angle_to_tangent=...) → TACS-X-like:
    TACS-3-like:  60–90° (perpendicular, INVASIVE)  — RED
    TACS-2-like:   0–30° (parallel)                 — GREEN
    TACS-1-like:  30–60° or curly                   — BLUE

Usage
-----
  python example_curvealign_workflow.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from shapely.geometry import Point

# ── Image registration ────────────────────────────────────────────────────────
from tme_quant.image_registration.methods.intensity_based import HESHGRegistration
from tme_quant.image_registration.config import RegistrationParams, TransformType

# ── Fiber analysis ────────────────────────────────────────────────────────────
from tme_quant.fiber_analysis import FiberOrientationAnalyzer
from tme_quant.fiber_analysis.config import CurveAlignParams
from tme_quant.fiber_analysis.utils import available_backends
from tme_quant.fiber_analysis.utils.geometry_utils import compute_angle_to_boundary_normal
from tme_quant.fiber_analysis.tacs import classify_fiber_segment_tacs_like, get_tacs_color

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
    TumorDetectionParams,
    TumorDetectionMethod,
)

# ── TACS zone analysis ────────────────────────────────────────────────────────
from tme_quant.core.base_models import Geometry, GeometryType
from tme_quant.core.roi_manager import ROIObject
from tme_quant.tme_analysis.pipelines import analyze_tacs_zone


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _tumor_to_roi(tumor, pixel_size: float, idx: int = 0) -> ROIObject:
    """Convert a TumorRegion to an ROIObject with pixel-space coordinates.

    TumorRegion.geometry.coordinates are stored in µm; divide by pixel_size
    to convert back to the pixel grid that orientation maps live on.
    """
    coords_px = np.asarray(tumor.geometry.coordinates) / pixel_size
    geom  = Geometry(type=GeometryType.POLYGON, coordinates=coords_px)
    label = getattr(tumor, 'object_id', None) or f'tumor_{idx}'
    return ROIObject(object_id=label, label=label, geometry=geom)


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
    Compute spatial, K-NN alignment, density, and TACS-like metrics for
    CurveAlign fiber segments.
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
                    min_dist   = dist
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

        seg_data['tacs_type'] = (
            classify_fiber_segment_tacs_like(
                angle_to_tangent=angle_to_tangent,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
            )
            if min_dist <= tacs_zone_width and angle_to_tangent is not None
            else None
        )

        if len(fiber_segments) > k_neighbors:
            _, indices = tree.query(position, k=k_neighbors + 1)
            neighbor_orientations = (
                fiber_segments.iloc[indices[1:]]['orientation'].values
            )
            diffs = np.abs(orientation - neighbor_orientations)
            diffs = np.minimum(diffs, 180.0 - diffs)
            seg_data['local_alignment'] = 1.0 - float(np.mean(diffs) / 90.0)
            seg_data['mean_angle_diff'] = float(np.mean(diffs))
        else:
            seg_data['local_alignment'] = 0.0
            seg_data['mean_angle_diff'] = 0.0

        bbox_radius = (bbox_size / 2.0) / pixel_size
        nearby   = tree.query_ball_point(position, bbox_radius)
        count    = len(nearby) - 1
        area_mm2 = np.pi * ((bbox_size / 2.0) / 1000.0) ** 2
        seg_data['local_density'] = count / area_mm2 if area_mm2 > 0 else 0.0

        metrics.append(seg_data)

    return pd.DataFrame(metrics)


def _generate_segment_heatmaps(
    shg_image: np.ndarray,
    segment_metrics: pd.DataFrame,
    tumor_regions: List,
    output_dir: Path,
    sample_id: str,
) -> Dict[str, Path]:
    """
    Generate orientation, alignment, and density heatmaps.
    Saves PNGs to *output_dir* and returns a dict of saved paths.
    """
    h, w = shg_image.shape[:2]
    resolution = 512
    gx, gy = np.meshgrid(
        np.linspace(0, w, resolution),
        np.linspace(0, h, resolution),
    )
    positions  = segment_metrics[['position_x', 'position_y']].values
    saved: Dict[str, Path] = {}

    for col, cmap, label in [
        ('orientation',    'hsv',    'Orientation (°)'),
        ('local_alignment','RdYlGn', 'K-NN Alignment'),
        ('local_density',  'viridis','Density (segs/mm²)'),
    ]:
        values = segment_metrics[col].fillna(0).values
        grid   = griddata(positions, values, (gx, gy), method='linear', fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(shg_image, cmap='gray', alpha=0.5)
        im = ax.imshow(grid, cmap=cmap, alpha=0.7)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f'{sample_id} — {label}')
        ax.axis('off')
        fig.tight_layout()
        out_path = output_dir / f'{sample_id}_heatmap_{col}.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved[col] = out_path

    return saved


def _create_segment_overlay(
    shg_image: np.ndarray,
    registered_he: np.ndarray,
    segment_metrics: pd.DataFrame,
    tumor_regions: List,
    output_dir: Path,
    sample_id: str,
) -> Path:
    """
    Paint TACS-coloured dots on an SHG+H&E composite and save as PNG.
    Returns the saved path.
    """
    import cv2
    shg_rgb = (
        cv2.cvtColor(
            (shg_image / shg_image.max() * 255).astype(np.uint8),
            cv2.COLOR_GRAY2RGB,
        )
        if shg_image.ndim == 2
        else shg_image.copy()
    )
    he_rgb  = (
        registered_he
        if registered_he.ndim == 3
        else cv2.cvtColor(
            (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
    )
    overlay = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)

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
            coords = np.array(
                tumor.roi.polygon.exterior.coords
            ).astype(np.int32)
            cv2.polylines(overlay, [coords], True, (255, 255, 0), 2)

    out_path = output_dir / f'{sample_id}_overlay_tacs.png'
    io.imsave(out_path, overlay)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL SIZE UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _infer_pixel_size(image_path: str) -> Optional[float]:
    """
    Try to read pixel size (µm/px) from TIFF image metadata.

    Checks in order:
      1. TIFF XResolution / ResolutionUnit tags  (most scanners)
      2. OME-TIFF XML ``PhysicalSizeX`` in ImageDescription
      3. MicroManager ``Summary.PixelSize_um`` JSON in ImageDescription

    Returns ``None`` if no calibration metadata is found.
    """
    try:
        import tifffile
        with tifffile.TiffFile(image_path) as tif:
            page   = tif.pages[0]
            tags   = {t.name: t.value for t in page.tags.values()}

            # ── Standard TIFF resolution tags ───────────────────────────────
            if 'XResolution' in tags:
                xres = tags['XResolution']
                if isinstance(xres, tuple) and xres[1] != 0:
                    xres = xres[0] / xres[1]      # rational → float
                elif isinstance(xres, (int, float)):
                    xres = float(xres)
                else:
                    xres = None

                if xres is not None and xres > 0:
                    res_unit = tags.get('ResolutionUnit', 2)  # 2=inch,3=cm
                    if res_unit == 3:       # cm  → µm/px
                        ps = 10_000.0 / xres
                    elif res_unit == 2:     # inch → µm/px
                        ps = 25_400.0 / xres
                    else:
                        ps = None
                    if ps is not None and 0.01 < ps < 100.0:
                        return round(ps, 6)

            # ── OME-TIFF XML in ImageDescription ─────────────────────────────
            img_desc = tags.get('ImageDescription', '')
            if isinstance(img_desc, bytes):
                img_desc = img_desc.decode('utf-8', errors='ignore')

            if 'PhysicalSizeX' in img_desc:
                import re
                m = re.search(r'PhysicalSizeX="([\d.]+)"', img_desc)
                if m:
                    ps = float(m.group(1))
                    if 0.01 < ps < 100.0:
                        return round(ps, 6)

            # ── MicroManager JSON metadata ────────────────────────────────────
            if 'PixelSize_um' in img_desc:
                import json, re
                m = re.search(r'\{.*\}', img_desc, re.DOTALL)
                if m:
                    try:
                        meta = json.loads(m.group())
                        ps   = float(
                            meta.get('Summary', meta).get('PixelSize_um', 0)
                        )
                        if 0.01 < ps < 100.0:
                            return round(ps, 6)
                    except Exception:
                        pass

    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def workflow_curvealign_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: Optional[float] = None,
    sample_id: str = 'patient_001_curvealign',
    skip_registration: bool = False,
    skip_tumor_detection: bool = False,
) -> Dict:
    """
    Complete CurveAlign workflow for fiber segment orientation analysis.

    Parameters
    ----------
    pixel_size : float or None
        Image resolution in µm/pixel.  When ``None`` (default), the value
        is read automatically from the SHG TIFF metadata (XResolution tag,
        OME-TIFF PhysicalSizeX, or MicroManager PixelSize_um).  If no
        calibration metadata is found a fallback of 1.0 px/µm is used and
        a warning is printed.  Correct pixel size is important for:
          • StarDist size filter (min/max cell area in µm²)
          • TACS zone width (100 µm → pixel distance)
          • Fiber density (segments/mm²)
    skip_registration : bool, optional
        If True, step 2 (H&E → SHG registration) is skipped and the
        H&E image is used as-is (assumed already registered).  Default False.
    skip_tumor_detection : bool, optional
        If True, skip DBSCAN tumor-boundary detection and replace it with a
        single synthetic ellipse ROI centred on the image with area ≈ 1/5
        of the total image area.  This is useful for demos or when the
        clustering result is unreliable (too few cells, noisy tissue).
        Downstream steps that require real ``tumor_regions`` (segment metrics,
        heatmaps, overlay) receive an empty list.  The TACS zone step always
        receives the ellipse ROI.  Default False.

    Returns
    -------
    Dict with keys: sample_id, registered_he, shg_image, fiber_segments,
    segment_metrics, cells, tumor_regions, summary_stats, saved_figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('  Workflow: CurveAlign Fiber Segment Analysis')
    print(f'  Sample  : {sample_id}')
    print('=' * 72)

    backends = available_backends()
    print(f'\n  Curvelet backends: {backends}')
    if not backends['curvelops']:
        print('  WARNING: curvelops not installed — NumPy fallback in use.')
        print('           Install with: pip install curvelops')

    # ── [1/9] Load images ────────────────────────────────────────────────────
    print('\n[1/9] Loading images...')
    he_image  = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    print(f'  H&E: {he_image.shape},  SHG: {shg_image.shape}')

    # ── Resolve pixel size ───────────────────────────────────────────────────────────
    _px_source = 'user-supplied'
    if pixel_size is None:
        pixel_size = _infer_pixel_size(shg_image_path)
        if pixel_size is not None:
            _px_source = 'auto (SHG metadata)'
        else:
            pixel_size = _infer_pixel_size(he_image_path)
            if pixel_size is not None:
                _px_source = 'auto (H&E metadata)'
            else:
                pixel_size = 1.0
                _px_source = 'fallback (no metadata found)'
                print(
                    '  WARNING: pixel size not found in image metadata.\n'
                    '           Falling back to 1.0 µm/px.\n'
                    '           Size filters and TACS zone widths may be incorrect.\n'
                    '           Pass pixel_size=<value> explicitly if known.'
                )
    print(f'  Pixel size: {pixel_size} µm/px  [{_px_source}]')
    # When pixel_size=1.0 (fallback) the size filter defaults work in pixel².
    # StarDist 2D_versatile_he was trained at ~0.5 µm/px; at different
    # resolutions the model still runs but nucleus size filtering matters.
    _size_scale = pixel_size ** 2          # px² per µm²
    _min_cell_px2 = max(20.0 / _size_scale, 20.0)   # ≥20 px² regardless
    _max_cell_px2 = max(500.0 / _size_scale, 500.0) # limit in µm² OR px²

    # ── [2/9] Register H&E → SHG ─────────────────────────────────────────────
    if skip_registration:
        print('\n[2/9] Registration skipped — H&E assumed already registered.')
        registered_he = he_image
    else:
        print('\n[2/9] Registering H&E → SHG (affine)...')
        reg_params    = RegistrationParams(
            transform_type=TransformType.AFFINE,
            use_multiresolution=True,
            pyramid_levels=3,
            num_iterations=200,
        )
        registration  = HESHGRegistration()
        reg_result    = registration.register(shg_image, he_image, reg_params)
        registered_he = reg_result.registered_image
        print(f'  MI score: {reg_result.mutual_information:.4f}')
        io.imsave(
            out / f'{sample_id}_HE_registered.tif',
            (registered_he * 255).astype(np.uint8),
        )

    # ── [3/9] CurveAlign orientation analysis ───────────────────────────────
    print('\n[3/9] CurveAlign orientation analysis...')
    orientation_params = CurveAlignParams(
        pixel_size=pixel_size,
        window_size=64,
        overlap=0.5,
        curvelet_levels=4,
        curvelet_angles=8,
        compute_coherency=True,
        compute_energy=True,
        return_fiber_segments=True,
        candidate_keep=0.05,
        candidate_scale=1,
        candidate_radius=4.0,
        keep_values=['angles', 'alignment', 'energy'],
        compute_statistics=True,
    )
    orientation_analyzer = FiberOrientationAnalyzer()
    orientation_result   = orientation_analyzer.analyze_2d(
        shg_image, orientation_params
    )
    print(f'  Mean orientation: {orientation_result.mean_orientation:.2f}°')
    print(f'  Mean alignment  : {orientation_result.mean_alignment:.4f}')

    # ── [4/9] Extract fiber segments ─────────────────────────────────────────
    print('\n[4/9] Extracting fiber segments...')
    if (orientation_result.fiber_structure is not None
            and not orientation_result.fiber_structure.empty):
        # Use curvelet fiber candidates produced by analyze_2d (requires curvelops)
        _fs = orientation_result.fiber_structure
        _align_map = orientation_result.alignment_map
        if _align_map is not None:
            r_idx = _fs['center_row'].astype(int).clip(0, _align_map.shape[0] - 1)
            c_idx = _fs['center_col'].astype(int).clip(0, _align_map.shape[1] - 1)
            local_align = _align_map[r_idx.values, c_idx.values]
        else:
            local_align = np.zeros(len(_fs))
        fiber_segments = pd.DataFrame({
            'segment_id':                [f'seg_{i:06d}' for i in range(len(_fs))],
            'segment_index':             np.arange(len(_fs)),
            'position_x':                _fs['center_col'].values,
            'position_y':                _fs['center_row'].values,
            'orientation':               _fs['angle'].values,
            'local_alignment_intrinsic': local_align,
        })
        print(f'  Extracted {len(fiber_segments):,} curvelet fiber candidates')
    else:
        # Fallback: subsample orientation map (curvelops not installed)
        fiber_segments = _extract_fiber_segments_from_curvealign(
            orientation_map=orientation_result.orientation_map,
            alignment_map=orientation_result.alignment_map,
            pixel_size=pixel_size,
            subsample=2,
        )
        print(f'  Extracted {len(fiber_segments):,} fiber segments (subsample fallback)')

    # ── [5/9] Cell segmentation ──────────────────────────────────────────────
    print('\n[5/9] Segmenting cells (StarDist)...')
    seg_params    = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',   # H&E model; use '2D_versatile_fluo' for IF
        pixel_size=pixel_size,
        # Detection thresholds
        stardist_prob_thresh=0.5,           # lower → more detections, more false positives
        stardist_nms_thresh=0.4,            # non-max suppression overlap tolerance
        # Size filter in µm² (scaled from pixel² to µm² using pixel_size)
        min_cell_size=_min_cell_px2 * _size_scale,
        max_cell_size=_max_cell_px2 * _size_scale,
    )
    print(f'  Size filter: {seg_params.min_cell_size:.1f}–{seg_params.max_cell_size:.1f} µm²'
          f'  ({_min_cell_px2:.0f}–{_max_cell_px2:.0f} px²)')
    cell_analyzer = CellAnalyzer()
    seg_result    = cell_analyzer.segment_cells_2d(
        registered_he, seg_params, image_id=sample_id
    )
    cells = seg_result.cells
    print(f'  Segmented {len(cells):,} cells')

    # ── [6/9] Tumor boundary detection ──────────────────────────────────────
    tme_analyzer = TMEAnalyzer()
    if skip_tumor_detection:
        print('\n[6/9] Tumor boundary detection SKIPPED → using synthetic ellipse ROI')
        tumor_regions = []
        # Build a centred ellipse with area ≈ 1/5 of the image area.
        _H, _W = shg_image.shape[:2]
        _area_target = _H * _W / 5.0
        # area = π × rx × ry; keep aspect ratio rx/ry = W/H
        # → rx = W/sqrt(5π), ry = H/sqrt(5π)
        _denom = np.sqrt(5.0 * np.pi)
        _rx    = _W / _denom
        _ry    = _H / _denom
        _cx, _cy = _W / 2.0, _H / 2.0
        _theta = np.linspace(0, 2 * np.pi, 65)[:-1]
        _ell_coords = np.column_stack([
            _cx + _rx * np.cos(_theta),
            _cy + _ry * np.sin(_theta),
        ]).astype(np.float32)
        _ell_geom = Geometry(type=GeometryType.POLYGON, coordinates=_ell_coords)
        _tacs_rois = [ROIObject(
            object_id='synthetic_ellipse_roi',
            label='Synthetic Ellipse (demo)',
            geometry=_ell_geom,
        )]
        print(f'  Synthetic ellipse: centre ({_cx:.0f}, {_cy:.0f}) px, '
              f'rx={_rx:.0f} px, ry={_ry:.0f} px, '
              f'area≈{np.pi*_rx*_ry:.0f} px²  (target {_area_target:.0f} px²)')
    else:
        print('\n[6/9] Detecting tumor boundaries (DBSCAN)...')
        tumor_params  = TumorDetectionParams(
            method=TumorDetectionMethod.CLUSTERING,
            clustering_algorithm='dbscan',
            dbscan_eps=100.0,
            dbscan_min_samples=10,
            min_tumor_area=1000.0,
            smooth_boundary=True,
        )
        tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
        print(f'  Detected {len(tumor_regions)} tumor region(s)')
        _tacs_rois = [_tumor_to_roi(t, pixel_size, i) for i, t in enumerate(tumor_regions)
                      if t.geometry is not None and len(np.asarray(t.geometry.coordinates)) >= 3]

    # ── [7/9] Compute segment metrics ───────────────────────────────────────
    print('\n[7/9] Computing fiber segment metrics...')
    segment_metrics = _compute_fiber_segment_metrics(
        fiber_segments=fiber_segments,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        pixel_size=pixel_size,
    )
    print(f'  Metrics computed for {len(segment_metrics):,} segments')
    print(f'  Mean K-NN alignment : {segment_metrics["local_alignment"].mean():.3f}')
    print(f'  Mean density        : {segment_metrics["local_density"].mean():.1f} segs/mm²')
    tacs_counts = segment_metrics['tacs_type'].value_counts()
    print('  TACS-like distribution:')
    for tacs_type, count in tacs_counts.items():
        if tacs_type:
            print(f'    {tacs_type}: {count}  ({count / len(segment_metrics) * 100:.1f}%)')
    # ── TACS zone analysis (CurveAlign orientation map + tumor boundaries) ──
    # Runs analyze_tacs_zone for every detected tumor region.  The per-fiber
    # path is skipped (fiber_objects=None) — the orientation map from the
    # CurveAlign step provides all pixel-level TACS information.
    # Each TumorRegion geometry (µm) is converted to pixel-space before being
    # wrapped as an ROIObject so coordinates align with the orientation map.
    print('\n[TACS] TACS zone analysis (CurveAlign pixel-map, per-tumor ROI)...')
    tacs_zone_results: List[Dict] = []
    for _ti, _roi in enumerate(_tacs_rois):
        _tz  = analyze_tacs_zone(
            orientation_map = orientation_result.orientation_map,
            alignment_map   = orientation_result.alignment_map,
            roi             = _roi,
            fiber_objects   = None,
            pixel_size      = pixel_size,
            tacs_zone_width = 100.0,
            subsample       = 2,
            dense_boundary  = True,
            image_size      = shg_image.shape[:2],
        )
        tacs_zone_results.append({'tumor_idx': _ti, 'roi': _roi, 'result': _tz})
        _pr = _tz['pixel_result']
        if _pr.get('n_points_in_zone', 0) > 0:
            print(f'  ROI {_ti}: {_pr["n_points_in_zone"]} px in zone, '
                  f'mean angle {_pr["mean_angle_to_tangent"]:.1f}°  '
                  f'(combined: {_tz["combined_mean_angle_to_tangent"]:.1f}°)')
        else:
            print(f'  ROI {_ti}: 0 px in zone')
    print(f'  TACS zone analysis done for {len(tacs_zone_results)} ROI(s)')
    # ── [8/9] Generate heatmaps + overlay ───────────────────────────────────
    print('\n[8/9] Generating heatmaps and overlay...')
    heatmap_paths = _generate_segment_heatmaps(
        shg_image=shg_image,
        segment_metrics=segment_metrics,
        tumor_regions=tumor_regions,
        output_dir=out,
        sample_id=sample_id,
    )
    overlay_path = _create_segment_overlay(
        shg_image=shg_image,
        registered_he=registered_he,
        segment_metrics=segment_metrics,
        tumor_regions=tumor_regions,
        output_dir=out,
        sample_id=sample_id,
    )
    print(f'  Heatmaps + overlay written to {out}')

    # ── [9/9] Export ─────────────────────────────────────────────────────────
    print('\n[9/9] Exporting metrics...')
    segment_metrics.to_csv(
        out / f'{sample_id}_fiber_segment_metrics.csv', index=False
    )

    summary_stats: Dict = {
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
        out / f'{sample_id}_summary.csv', index=False
    )
    print(f'  Exported to {out}')

    print('\n' + '=' * 72)
    print('  CURVEALIGN WORKFLOW COMPLETE')
    print('=' * 72)

    return {
        'sample_id':          sample_id,
        'pixel_size':         pixel_size,
        'shg_image':          shg_image,
        'registered_he':      registered_he,
        'fiber_segments':     fiber_segments,
        'segment_metrics':    segment_metrics,
        'cells':              cells,
        'tumor_regions':      tumor_regions,
        'tacs_zone_results':  tacs_zone_results,
        'summary_stats':      summary_stats,
        'output_dir':         out,
        'saved_figures': {
            'heatmap_orientation': heatmap_paths.get('orientation'),
            'heatmap_alignment':   heatmap_paths.get('local_alignment'),
            'heatmap_density':     heatmap_paths.get('local_density'),
            'overlay_tacs':        overlay_path,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(summary_stats: Dict) -> None:
    """Print a single-row summary table of the workflow results."""
    rows = [
        ('Segments',           summary_stats['n_segments']),
        ('Cells',              summary_stats['n_cells']),
        ('Tumor regions',      summary_stats['n_tumors']),
        ('Mean orientation',   f"{summary_stats['mean_orientation']:.2f}°"),
        ('Mean K-NN alignment',f"{summary_stats['mean_alignment']:.4f}"),
        ('Mean density',       f"{summary_stats['mean_density']:.1f} segs/mm²"),
        ('TACS-1-like count',  summary_stats.get('TACS-1-like_count', 'n/a')),
        ('TACS-2-like count',  summary_stats.get('TACS-2-like_count', 'n/a')),
        ('TACS-3-like count',  summary_stats.get('TACS-3-like_count', 'n/a')),
        ('TACS-1-like ratio',  f"{summary_stats.get('TACS-1-like_ratio', 0)*100:.1f}%"),
        ('TACS-2-like ratio',  f"{summary_stats.get('TACS-2-like_ratio', 0)*100:.1f}%"),
        ('TACS-3-like ratio',  f"{summary_stats.get('TACS-3-like_ratio', 0)*100:.1f}%"),
    ]
    width = 42
    print()
    print('┌' + '─' * width + '┐')
    print('│{:^{w}}│'.format(' CurveAlign Summary Statistics ', w=width))
    print('├' + '─' * 24 + '┬' + '─' * (width - 25) + '┤')
    for label, value in rows:
        print('│ {:<22} │ {:<{w}} │'.format(label, str(value), w=width - 27))
    print('└' + '─' * 24 + '┴' + '─' * (width - 25) + '┘')


def _print_segments_table(
    segment_metrics: pd.DataFrame,
    max_rows: int = 25,
) -> None:
    """Print a sample of per-segment measurements as a formatted table."""
    cols = [
        'segment_id', 'position_x', 'position_y',
        'orientation', 'local_alignment', 'local_density',
        'distance_to_tumor', 'angle_to_tangent', 'tacs_type',
    ]
    # Show segments that have a TACS classification first, then fill with others
    classified = segment_metrics[segment_metrics['tacs_type'].notna()].copy()
    unclassified = segment_metrics[segment_metrics['tacs_type'].isna()].head(5)
    sample_df = (
        pd.concat([classified, unclassified])
        .head(max_rows)
        [[c for c in cols if c in segment_metrics.columns]]
    )

    # Round floats for display
    fmt = {
        'position_x':       '{:.0f}',
        'position_y':       '{:.0f}',
        'orientation':      '{:.1f}',
        'local_alignment':  '{:.3f}',
        'local_density':    '{:.1f}',
        'distance_to_tumor':'{:.1f}',
        'angle_to_tangent': '{:.1f}',
    }
    display_df = sample_df.copy()
    for col, f in fmt.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v: f.format(v) if pd.notna(v) else 'n/a'
            )

    print()
    print(f'  Fiber segment measurements  '
          f'(showing {len(display_df)} of {len(segment_metrics):,} segments)')
    print('  ' + '-' * 100)
    with pd.option_context(
        'display.max_columns', None,
        'display.width',       120,
        'display.max_colwidth', 16,
    ):
        print(display_df.to_string(index=False))
    print()


def print_measurement_tables(results: Dict) -> None:
    """Print summary and per-segment measurement tables to stdout."""
    _print_summary_table(results['summary_stats'])
    _print_segments_table(results['segment_metrics'])


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: FIGURE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _display_cell_segmentation_figure(results: Dict) -> None:
    """
    Show cell segmentation outlines (coloured by cell type) and tumor
    boundary polygons overlaid on the registered H&E image.

    ``cell.boundary`` stores ALL interior pixel coordinates in µm
    (``region.coords[:, ::-1] * pixel_size``).  A ConvexHull is computed
    from that cloud to extract the actual outline polygon.
    Coordinates are divided by ``pixel_size`` before plotting so they
    align with the imshow pixel grid.
    Tumor boundaries undergo the same µm → pixel conversion.
    """
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import ConvexHull as _ConvexHull

    registered_he = results['registered_he']
    cells         = results['cells']
    tumor_regions = results['tumor_regions']
    pixel_size    = results.get('pixel_size', 1.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(registered_he)
    ax.axis('off')

    # ── Cell outlines, grouped by cell type ─────────────────────────────────
    # cell.boundary = region.coords[:, ::-1] * pixel_size  →  (N,2) (x,y) µm
    # ConvexHull on the interior cloud gives the outline vertices.
    _PALETTE = [
        '#00BFFF', '#FF6347', '#7CFC00', '#FFD700',
        '#DA70D6', '#FF8C00', '#00FA9A', '#F08080',
    ]
    type_to_segs:  Dict[str, list] = {}
    type_to_count: Dict[str, int]  = {}
    for cell in cells:
        label = cell.cell_type.value if cell.cell_type else 'Unknown'
        type_to_count[label] = type_to_count.get(label, 0) + 1
        bnd = cell.boundary
        if bnd is None or len(bnd) < 4:
            continue
        bnd_px = bnd / pixel_size          # µm → pixel coords
        try:
            hull  = _ConvexHull(bnd_px)
            verts = bnd_px[hull.vertices]
            pts   = np.vstack([verts, verts[:1]])   # close the polygon
            type_to_segs.setdefault(label, []).extend(
                [[tuple(pts[j]), tuple(pts[j + 1])] for j in range(len(pts) - 1)]
            )
        except Exception:
            pass

    type_names = sorted(type_to_segs.keys())
    color_map  = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(type_names)}

    legend_handles: list = []
    for t in type_names:
        segs  = type_to_segs[t]
        color = color_map[t]
        ax.add_collection(
            LineCollection(segs, linewidths=0.8, colors=color, alpha=0.85)
        )
        legend_handles.append(
            mpatches.Patch(
                color=color,
                label=f'{t}  (n={type_to_count.get(t, 0):,})',
            )
        )

    # ── Tumor boundary polygons ──────────────────────────────────────────────
    # tumor.geometry.coordinates are in µm → divide by pixel_size for display
    tumor_polys = []
    for tumor in tumor_regions:
        if tumor.geometry is not None and len(tumor.geometry.coordinates) >= 3:
            coords_px = np.asarray(tumor.geometry.coordinates) / pixel_size
            tumor_polys.append(
                MplPolygon(coords_px, closed=True)
            )
    if tumor_polys:
        ax.add_collection(
            PatchCollection(
                tumor_polys,
                facecolor='yellow', alpha=0.15,
                edgecolor='gold',   linewidths=2.5,
            )
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor='yellow', alpha=0.5, edgecolor='gold', linewidth=2,
                label=f'Tumor boundary  (n={len(tumor_polys)})',
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc='upper right', fontsize=8,
            framealpha=0.75, title='Cell type / region',
        )

    n_drawn = sum(
        1 for c in cells
        if c.boundary is not None and len(c.boundary) >= 4
    )
    ax.set_title(
        f'{n_drawn:,} cells (convex hull outlines)'
        f' · {len(tumor_regions)} tumor region(s)',
        fontsize=10,
    )
    fig.suptitle(
        f"Cell Segmentation & Tumor Boundaries — {results['sample_id']}",
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    plt.show()


def display_figure_panel(results: Dict) -> None:
    """
    Show a 2×2 figure panel:
      [0,0] SHG image          [0,1] orientation heatmap
      [1,0] alignment heatmap  [1,1] TACS overlay

    Reads the saved PNG files produced by the workflow.
    """
    figs  = results['saved_figures']
    pairs = [
        ('SHG image',           None),                        # rendered directly
        ('Orientation heatmap', figs.get('heatmap_orientation')),
        ('Alignment heatmap',   figs.get('heatmap_alignment')),
        ('TACS overlay',        figs.get('overlay_tacs')),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"CurveAlign Results — {results['sample_id']}",
        fontsize=14, fontweight='bold',
    )

    shg = results['shg_image']
    for ax, (title, path) in zip(axes.flat, pairs):
        if path is None:
            # SHG image slot
            ax.imshow(shg, cmap='gray')
            ax.set_title('SHG image', fontsize=11)
        elif path is not None and Path(path).exists():
            img = io.imread(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
        else:
            ax.text(0.5, 0.5, f'{title}\n(file not found)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
        ax.axis('off')

    # TACS legend on the overlay panel
    tacs_ax = axes[1, 1]
    legend_handles = [
        mpatches.Patch(color=np.array(get_tacs_color('TACS-3-like')) / 255,
                       label='TACS-3-like (perpendicular)'),
        mpatches.Patch(color=np.array(get_tacs_color('TACS-2-like')) / 255,
                       label='TACS-2-like (parallel)'),
        mpatches.Patch(color=np.array(get_tacs_color('TACS-1-like')) / 255,
                       label='TACS-1-like (random)'),
    ]
    tacs_ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=7,
        framealpha=0.7,
    )

    fig.tight_layout()
    plt.show()

    _display_cell_segmentation_figure(results)
    _display_tacs_zone_figure(results)


def _display_tacs_zone_figure(results: Dict) -> None:
    """
    Plot TACS zone heatmaps (one figure per detected tumor region).

    Calls ``plot_tacs_heatmap`` from ``example_analyze_tacs_zone`` — the same
    two-panel visualisation (spatial heatmap + TACS distribution bar chart)
    used in the standalone TACS zone example.  The pixel-map path was run
    with CurveAlign orientation, so the spatial panel shows per-pixel
    angle-to-boundary-tangent coloured by the RdYlBu_r colormap.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from example_analyze_tacs_zone import plot_tacs_heatmap

    tacs_zone_results = results.get('tacs_zone_results', [])
    if not tacs_zone_results:
        print('[TACS display] no TACS zone results to plot')
        return

    pixel_size = results.get('pixel_size', 1.0)
    thr_px     = int(round(100.0 / pixel_size))   # 100 µm → pixels

    for entry in tacs_zone_results:
        plot_tacs_heatmap(
            result                     = entry['result'],
            roi                        = entry['roi'],
            fiber_objects              = None,        # pixel-map path only
            inside_roi_pixels          = None,
            inside_roi_fibers          = None,
            boundary_dist_threshold_px = thr_px,
            pixel_size                 = pixel_size,
            title                      = (
                f"TACS zone — {results['sample_id']}  "
                f"tumor {entry['tumor_idx']}  (CurveAlign orientation)"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results: Dict) -> None:
    """Print measurement tables and show the figure panel."""
    print_measurement_tables(results)
    display_figure_panel(results)


if __name__ == '__main__':
    results = workflow_curvealign_complete(
        he_image_path='data/patient_001_HE.tif',
        shg_image_path='data/patient_001_SHG.tif',
        output_dir='output/patient_001_curvealign',
        pixel_size=0.5,           # 0.5 µm/px
        sample_id='patient_001_curvealign',
        skip_registration=True,
        skip_tumor_detection=True,  # use synthetic ellipse ROI for TACS demo
    )
    display_results(results)
