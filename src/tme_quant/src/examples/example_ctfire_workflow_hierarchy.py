"""
TMEQuant Example — Workflow 2: CT-FIRE Individual Fiber Extraction (2-D)
========================================================================

End-to-end demonstration of CT-FIRE-based TACS individual fiber analysis:

  1.  Load H&E and SHG images.
  2.  Register H&E → SHG (affine, Keikhosravi 2020).
  3.  CT-FIRE fiber extraction (two-stage pipeline):
        Stage 1 (CT): multi-scale curvelet transform → fiber mask.
        Stage 2 (FIRE): distance transform → ridge-trace → (row, col, radius)
          centerlines; fiber width integral to tracing.
  4.  Segment cells from the registered H&E (StarDist).
  5.  Detect tumor boundaries (DBSCAN clustering).
  6.  TME analysis: fiber-tumor interactions + TACS classification (with
      straightness criterion).
  7.  Run InteractionAnalysisPipeline.
  8.  Compute per-fiber metrics: spatial, K-NN alignment, density, TACS.
  9.  Network analysis on interaction pairs.
  10. Generate heatmaps (orientation, alignment, density) saved to disk.
  11. Create TACS-coloured fiber overlay saved to disk.
  12. Print measurement tables (summary stats + per-fiber sample).
  13. Display a 2×2 figure panel: SHG | length heatmap |
                                  alignment heatmap | TACS overlay.

CT-FIRE vs Skeleton distinction
---------------------------------
  CT-FIRE FIRE:  mask → distance transform → ridge trace → (row, col, radius_px).
    Width is integral to tracing (not post-hoc); touching fibers of different
    thickness are correctly separated.
  Skeleton:      mask → 1-px medial axis → component trace → (row, col).
    Width estimated afterward from intensity profiles (FWHM).

TACS angle convention (boundary tangent)
-----------------------------------------
  compute_angle_to_boundary_normal() → angle to NORMAL (0–90°).
  angle_to_tangent = 90° − angle_to_normal.
  classify_fiber_tacs(angle_to_tangent=..., straightness=...) → TACS-1/2/3.
    TACS-3:  60–90° (perpendicular, INVASIVE)  — RED
    TACS-2:   0–30° (parallel)                 — GREEN
    TACS-1:  30–60° or curly                   — BLUE

Usage
-----
  python example_ctfire_workflow.py
"""

from __future__ import annotations

import sys
# Ensure UTF-8 output on Windows (→ and other non-ASCII chars in print calls).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
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
from tme_quant.fiber_analysis import FiberAnalyzer
from tme_quant.fiber_analysis.config import CTFireParams, FiberProperties
from tme_quant.fiber_analysis.utils import ctfire_backend_status
from tme_quant.fiber_analysis.utils.geometry_utils import compute_angle_to_boundary_normal
from tme_quant.fiber_analysis.tacs import classify_fiber_tacs, get_tacs_color

# ── Cell analysis ─────────────────────────────────────────────────────────────
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.cell_analysis.config import SegmentationMode

try:
    from tme_quant.cell_analysis.config import SegmentationParams
except ImportError:
    from dataclasses import dataclass
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
)
from tme_quant.tme_analysis.pipelines.interaction_analysis_pipeline import (
    InteractionAnalysisPipeline,
    PipelineConfig,
)
from tme_quant.tme_analysis.interaction_network import InteractionNetworkAnalyzer
from tme_quant.tme_analysis.io import export_tme_analysis_results

# ── TACS zone analysis ────────────────────────────────────────────────────────
from tme_quant.core.base_models import Geometry, GeometryType, TMEObject, TMEType
from tme_quant.core.hierarchy import TMEHierarchy
from tme_quant.core.image_entry import ImageEntry
from tme_quant.core.roi_manager import ROIObject
from tme_quant.tme_analysis.pipelines import analyze_tacs_zone

# ── Project-level IO ─────────────────────────────────────────────────────────
from tme_quant.core.io import save_project, load_project, export_project_summary


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


class _FiberAdapter:
    """Lightweight adapter: FiberProperties → interface expected by analyze_tacs_zone.

    ``analyze_tacs_zone`` looks for ``.object_id``, ``.angle``, and
    ``.centerline`` (row-col in pixels).  ``FiberProperties`` stores ``fiber_id``
    instead of ``object_id``; this class bridges the gap.

    Note: ``FiberAnalyzer.extract_2d`` already converts ``FiberProperties``
    to ``FiberObject`` (which has ``.object_id``, ``.angle``, ``.centerline``).
    This adapter is only needed if you call the raw CT-FIRE extraction method
    directly (``CTFireExtraction.extract_2d``) which returns ``FiberProperties``.
    """
    __slots__ = ('object_id', 'angle', 'centerline')

    def __init__(self, fp) -> None:
        # FiberObject has object_id; FiberProperties has fiber_id
        if hasattr(fp, 'object_id'):
            self.object_id = fp.object_id
        else:
            self.object_id = f'fiber_{fp.fiber_id}'
        self.angle      = float(fp.angle) if fp.angle is not None else 0.0
        self.centerline = fp.centerline   # (N, 2) row-col pixels, or None


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
    Compute spatial, K-NN alignment, density, and full TACS metrics for
    individual CT-FIRE fibers.

    FiberProperties fields used:
      .length, .width (distance-transform), .straightness, .angle,
      .centerline [(N,2) float32 in (row, col)].
    """
    if not fibers:
        return pd.DataFrame()

    def _fiber_center(f) -> np.ndarray:
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
            'width':       fiber.width,
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

        fiber_data['tacs_type'] = (
            classify_fiber_tacs(
                angle_to_tangent=angle_to_tangent,
                straightness=straightness,
                distance_to_boundary=min_dist,
                tacs_zone_width=tacs_zone_width,
                straightness_threshold=straightness_threshold,
            )
            if (
                min_dist <= tacs_zone_width
                and angle_to_tangent is not None
                and straightness is not None
            )
            else None
        )

        if len(fibers) > k_neighbors and angle is not None:
            _, indices = tree.query(center, k=k_neighbors + 1)
            neighbor_angles = [fibers[idx].angle for idx in indices[1:]]
            neighbor_angles = [a for a in neighbor_angles if a is not None]
            if neighbor_angles:
                diffs = np.array(
                    [abs((angle - na) % 180) for na in neighbor_angles]
                )
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


def _generate_fiber_heatmaps(
    shg_image: np.ndarray,
    fiber_metrics: pd.DataFrame,
    tumor_regions: List,
    output_dir: Path,
    sample_id: str,
) -> Dict[str, Path]:
    """
    Generate orientation, alignment, and density heatmaps for CT-FIRE fibers.
    Returns dict of saved paths keyed by metric name.
    """
    if len(fiber_metrics) == 0:
        return {}

    h, w = shg_image.shape[:2]
    resolution = 512
    gx, gy = np.meshgrid(
        np.linspace(0, w, resolution),
        np.linspace(0, h, resolution),
    )
    positions = fiber_metrics[['midpoint_x', 'midpoint_y']].values
    saved: Dict[str, Path] = {}

    for col, cmap, label in [
        ('orientation',         'hsv',    'Orientation (°)'),
        ('length',              'plasma', 'Fiber Length (µm)'),
        ('local_alignment',     'RdYlGn', 'K-NN Alignment'),
        ('local_fiber_density', 'viridis','Density (fibers/mm²)'),
    ]:
        values = fiber_metrics[col].fillna(0).values
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


def _create_fiber_overlay(
    shg_image: np.ndarray,
    registered_he: np.ndarray,
    fibers: List[FiberProperties],
    fiber_metrics: pd.DataFrame,
    tumor_regions: List,
    output_dir: Path,
    sample_id: str,
) -> Path:
    """
    Paint TACS-coloured fiber centerlines on an SHG+H&E composite.
    Centerlines are (N, 2) in (row, col); cv2 needs (x, y) so columns are
    flipped before drawing.
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

    for i, fiber in enumerate(fibers):
        tacs_type = (
            fiber_metrics.iloc[i].get('tacs_type')
            if i < len(fiber_metrics) else None
        )
        if tacs_type and fiber.centerline is not None and len(fiber.centerline) > 1:
            pts = fiber.centerline[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, get_tacs_color(tacs_type), 2)

    for tumor in tumor_regions:
        if hasattr(tumor, 'roi') and hasattr(tumor.roi, 'polygon'):
            coords = np.array(
                tumor.roi.polygon.exterior.coords
            ).astype(np.int32)
            cv2.polylines(overlay, [coords], True, (255, 255, 0), 3)

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

def workflow_ctfire_complete(
    he_image_path: str,
    shg_image_path: str,
    output_dir: str,
    pixel_size: Optional[float] = None,
    sample_id: str = 'patient_001_ctfire',
    skip_registration: bool = False,
    skip_tumor_detection: bool = False,
) -> Dict:
    """
    Complete CT-FIRE workflow for individual fiber analysis with full TACS.

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
          • Fiber length / width measurements
    skip_registration : bool, optional
        If True, step 2 (H&E → SHG registration) is skipped and the
        H&E image is used as-is (assumed already registered).  Default False.
    skip_tumor_detection : bool, optional
        If True, skip DBSCAN tumor-boundary detection and replace it with a
        single synthetic ellipse ROI centred on the image with area ≈ 1/5
        of the total image area.  This is useful for demos or when the
        clustering result is unreliable (too few cells, noisy tissue).
        Downstream steps that require real ``tumor_regions`` (TME analysis,
        interaction pipeline, network analysis, heatmaps) receive an empty
        list.  The TACS zone step always receives the ellipse ROI.
        Default False.

    Returns
    -------
    Dict with keys: sample_id, registered_he, shg_image, fibers,
    fiber_metrics, cells, tumor_regions, tme_result, pipeline_result,
    network_results, summary_stats, saved_figures, hierarchy.
    The ``hierarchy`` value is a ``TMEHierarchy`` whose root contains an
    ``ImageEntry`` node populated with all ``FiberObject`` and ``CellObject``
    children (per-fiber metrics written back onto each node) plus any
    ``TumorRegion`` nodes each carrying a ``tacs_zone_result`` in
    ``.metadata``.  Query examples::

        h = result['hierarchy']
        fibers = h.get_objects_by_type(TMEType.FIBER)
        tacs3  = [f for f in fibers if f.tacs_type == 'TACS-3']
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('  Workflow: CT-FIRE Individual Fiber Analysis')
    print(f'  Sample  : {sample_id}')
    print('=' * 72)

    fire_status = ctfire_backend_status()
    print(f'\n  CT-FIRE backend: {fire_status}')
    if not fire_status['cpp_available']:
        print('  NOTE: C++ FIRE extension not compiled — Python fallback in use.')
    if not fire_status['3d_supported']:
        print('  NOTE: 3-D CT-FIRE not available. Use SkeletonParams for 3-D.')

    # ── [1/12] Load images ────────────────────────────────────────────────────
    print('\n[1/12] Loading images...')
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
    _size_scale = pixel_size ** 2          # px² per µm²
    _min_cell_px2 = max(20.0 / _size_scale, 20.0)
    _max_cell_px2 = max(500.0 / _size_scale, 500.0)

    # ── Build TME object hierarchy ────────────────────────────────────────────
    # The hierarchy is populated incrementally as analysis steps complete.
    # Final tree:
    #   root (project)
    #   └── ImageEntry
    #       ├── TumorRegion_*  .metadata['tacs_zone_result']
    #       ├── FiberObject_*  .tacs_type / .relative_angle_to_boundary_tangent /
    #       │                  .metadata['local_alignment', 'local_fiber_density']
    #       └── CellObject_*
    _root = TMEObject(
        object_id=f'{sample_id}_project',
        name=sample_id,
        tme_type=TMEType.PROJECT,
    )
    image_entry = ImageEntry(
        object_id=f'{sample_id}_image',
        name=f'{sample_id} (SHG + H&E)',
        pixel_size=(pixel_size, pixel_size),
        modality='SHG+HE',
        channel_names=['SHG', 'HE'],
    )
    hierarchy = TMEHierarchy(root=_root)
    hierarchy.add_object(image_entry)

    # ── [2/12] Register H&E → SHG ────────────────────────────────────────────
    if skip_registration:
        print('\n[2/12] Registration skipped — H&E assumed already registered.')
        registered_he = he_image
    else:
        print('\n[2/12] Registering H&E → SHG (affine)...')
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

    # ── [3/12] CT-FIRE fiber extraction ──────────────────────────────────────
    print('\n[3/12] Extracting individual fibers (CT-FIRE)...')
    extraction_params = CTFireParams(
        pixel_size=pixel_size,
        ctfire_threshold=0.1,
        ctfire_n_levels=5,
        ctfire_n_angles=16,
        straightness_threshold=0.0,
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
    fiber_result   = fiber_analyzer.extract_2d(shg_image, extraction_params)
    fibers         = fiber_result.fibers
    print(f'  Extracted {len(fibers):,} individual fibers')
    if fibers:
        print(f'  Mean length      : {np.mean([f.length for f in fibers]):.2f} µm')
        print(f'  Mean width (DT)  : {np.mean([f.width for f in fibers]):.2f} µm')
        print(f'  Mean straightness: {np.mean([f.straightness for f in fibers]):.3f}')
    print(f'  Candidates before filter: {fiber_result.n_candidates}')
    print(f'  Fiber mask coverage: {fiber_result.fiber_mask.mean() * 100:.1f}%')

    # ── [4/12] Cell segmentation ──────────────────────────────────────────────
    print('\n[4/12] Segmenting cells (StarDist)...')
    seg_params    = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        stardist_model='2D_versatile_he',   # H&E model; use '2D_versatile_fluo' for IF
        pixel_size=pixel_size,
        # Detection thresholds
        stardist_prob_thresh=0.5,           # lower → more detections, more false positives
        stardist_nms_thresh=0.4,            # non-max suppression overlap tolerance
        # Size filter in µm² (scaled from pixel² using pixel_size)
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

    # ── [5/12] Tumor boundary detection ──────────────────────────────────────
    tme_analyzer = TMEAnalyzer()
    if skip_tumor_detection:
        print('\n[5/12] Tumor boundary detection SKIPPED → using synthetic ellipse ROI')
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
        print('\n[5/12] Detecting tumor boundaries (DBSCAN)...')
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

    # ── Populate hierarchy with analysis objects ──────────────────────────────
    # Tumor regions go directly under the image node so they can each hold
    # their own TACS zone result in .metadata later.
    tumor_nodes: List = []
    for _tr in tumor_regions:
        hierarchy.add_object(_tr, parent=image_entry)
        tumor_nodes.append(_tr)
    # Fibers and cells are added under the image node; in_tumor_boundary
    # (set during per-fiber metric pushback below) is the spatial filter.
    for _fn in fibers:
        hierarchy.add_object(_fn, parent=image_entry)
    for _cn in cells:
        hierarchy.add_object(_cn, parent=image_entry)
    print(f'  Hierarchy: {len(tumor_nodes)} tumor region(s), '
          f'{len(fibers)} fiber(s), {len(cells)} cell(s) added under image node')

    # ── [6/12] TME analysis ───────────────────────────────────────────────────
    print('\n[6/12] TME analysis (fiber-tumor interactions)...')
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
        analysis_id=f'{sample_id}_tme',
    )
    print('  TME analysis complete')
    if tme_result.tacs_features:
        tf = tme_result.tacs_features
        print(f'  TACS-1: {tf["tacs1_ratio"] * 100:.1f}%')
        print(f'  TACS-2: {tf["tacs2_ratio"] * 100:.1f}%')
        print(f'  TACS-3: {tf["tacs3_ratio"] * 100:.1f}%  (INVASIVE)')

    # ── [7/12] InteractionAnalysisPipeline ───────────────────────────────────
    print('\n[7/12] Running InteractionAnalysisPipeline...')
    pipeline_cfg = PipelineConfig(
        boundary_distance=100.0,
        tacs_zone_width=100.0,
        contact_threshold=5.0,
        compute_tacs=True,
        compute_mechanical=True,
        compute_contact_patterns=True,
        compute_prognostic=True,
        export_dir=out,
    )
    pipeline        = InteractionAnalysisPipeline(pipeline_cfg)
    pipeline_result = pipeline.run(
        cells=cells,
        fibers=fibers,
        tumors=tumor_regions,
        image_id=sample_id,
    )
    print(f'  {len(pipeline_result.interaction_pairs)} interaction pairs detected')

    # ── [8/12] Per-fiber metrics ──────────────────────────────────────────────
    print('\n[8/12] Computing per-fiber metrics...')
    fiber_metrics = _compute_individual_fiber_metrics(
        fibers=fibers,
        tumor_regions=tumor_regions,
        k_neighbors=10,
        bbox_size=100.0,
        tacs_zone_width=100.0,
        straightness_threshold=0.7,
        pixel_size=pixel_size,
    )
    print(f'  Metrics for {len(fiber_metrics):,} fibers')
    if len(fiber_metrics):
        print(f'  Mean K-NN alignment : {fiber_metrics["local_alignment"].mean():.3f}')
        print(f'  Mean density        : '
              f'{fiber_metrics["local_fiber_density"].mean():.1f} fibers/mm²')

    # ── Push per-fiber metrics back onto FiberObject hierarchy nodes ──────────
    # fiber_metrics rows are indexed by 'fiber_id' which equals FiberObject.object_id.
    # Writes typed attributes (.tacs_type, .relative_angle_to_boundary_tangent,
    # .nearest_boundary_distance, .in_tumor_boundary) and stashes computed metrics
    # in .metadata so the hierarchy is fully self-contained for queries.
    _fm_idx = {str(r['fiber_id']): r for _, r in fiber_metrics.iterrows()}
    for _fnode in fibers:
        _row = _fm_idx.get(str(_fnode.object_id))
        if _row is None:
            continue
        _fnode.tacs_type = _row.get('tacs_type')
        _at = _row.get('angle_to_tangent')
        if _at is not None:
            _fnode.relative_angle_to_boundary_tangent = float(_at)
        _dt = _row.get('distance_to_tumor')
        if _dt is not None and not np.isinf(float(_dt)):
            _fnode.nearest_boundary_distance = float(_dt)
            _fnode.in_tumor_boundary = float(_dt) <= 100.0
            _fnode.in_stroma = float(_dt) > 100.0
        _fnode.metadata['local_alignment']     = float(_row.get('local_alignment', 0.0))
        _fnode.metadata['local_fiber_density'] = float(_row.get('local_fiber_density', 0.0))
        _fnode.metadata['mean_angle_diff']     = float(_row.get('mean_angle_diff', 0.0))

    # ── TACS zone analysis (CTFire fibers + tumor boundaries) ────────────────
    # Runs analyze_tacs_zone for every detected tumor region.  The pixel-map
    # path is skipped (NaN orientation map) — fiber objects provide all TACS
    # information.  Each TumorRegion geometry (µm) is converted to pixel-space
    # before being wrapped as an ROIObject.
    print('\n[TACS] TACS zone analysis (CTFire fibers, per-tumor ROI)...')
    tacs_zone_results: List[Dict] = []
    _blank_omap = np.full(shg_image.shape[:2], np.nan, dtype=np.float32)
    for _ti, _roi in enumerate(_tacs_rois):
        _tz  = analyze_tacs_zone(
            orientation_map = _blank_omap,
            alignment_map   = None,
            roi             = _roi,
            fiber_objects   = fibers,   # FiberObject list; has object_id/angle/centerline
            pixel_size      = pixel_size,
            tacs_zone_width = 100.0,
            subsample       = 1,
            dense_boundary  = True,
            image_size      = shg_image.shape[:2],
        )
        tacs_zone_results.append({'tumor_idx': _ti, 'roi': _roi, 'result': _tz})
        _frs = _tz['fiber_results']
        if _frs:
            _mean_ang = np.mean([f['angle_to_boundary_tangent'] for f in _frs])
            print(f'  ROI {_ti}: {len(_frs)} fibers in zone, '
                  f'mean angle {_mean_ang:.1f}°  '
                  f'(combined: {_tz["combined_mean_angle_to_tangent"]:.1f}°)')
        else:
            print(f'  ROI {_ti}: 0 fibers in zone')
    print(f'  TACS zone analysis done for {len(tacs_zone_results)} ROI(s)')

    # ── Store TACS zone results on hierarchy nodes ────────────────────────────
    if skip_tumor_detection:
        # No real tumor nodes — store on the image entry itself.
        image_entry.metadata['tacs_zone_result'] = (
            tacs_zone_results[0]['result'] if tacs_zone_results else None
        )
    else:
        for _ti, _tz_entry in enumerate(tacs_zone_results):
            if _ti < len(tumor_nodes):
                tumor_nodes[_ti].metadata['tacs_zone_result'] = _tz_entry['result']

    # ── [9/12] Network analysis ───────────────────────────────────────────────
    print('\n[9/12] Interaction network analysis...')
    network_analyzer = InteractionNetworkAnalyzer(
        weight_by='distance',
        community_method='greedy',
    )
    network_results = network_analyzer.analyze(
        pairs=pipeline_result.interaction_pairs,
        top_n_hubs=10,
    )
    nm = network_results['network_metrics']
    print(f'  Graph: {nm["n_nodes"]} nodes, {nm["n_edges"]} edges')
    print(f'  Density: {nm["density"]:.3f}')

    # ── [10/12] Generate heatmaps ─────────────────────────────────────────────
    print('\n[10/12] Generating heatmaps...')
    heatmap_paths = _generate_fiber_heatmaps(
        shg_image=shg_image,
        fiber_metrics=fiber_metrics,
        tumor_regions=tumor_regions,
        output_dir=out,
        sample_id=sample_id,
    )
    print(f'  Heatmaps written to {out}')

    # ── [11/12] Create overlay ────────────────────────────────────────────────
    print('\n[11/12] Creating TACS overlay...')
    overlay_path = _create_fiber_overlay(
        shg_image=shg_image,
        registered_he=registered_he,
        fibers=fibers,
        fiber_metrics=fiber_metrics,
        tumor_regions=tumor_regions,
        output_dir=out,
        sample_id=sample_id,
    )
    print(f'  Overlay written to {overlay_path}')

    # ── [12/12] Export ────────────────────────────────────────────────────────
    print('\n[12/12] Exporting results...')
    export_tme_analysis_results(
        tme_result,
        output_dir=out,
        formats=['csv', 'excel', 'json'],
        prefix=sample_id,
    )
    fiber_metrics.to_csv(out / f'{sample_id}_fiber_metrics.csv', index=False)

    summary_stats: Dict = {
        'sample_id':          sample_id,
        'n_fibers':           len(fibers),
        'n_cells':            len(cells),
        'n_tumors':           len(tumor_regions),
        'mean_fiber_length':  float(fiber_metrics['length'].mean())      if len(fiber_metrics) else 0.0,
        'mean_fiber_width':   float(fiber_metrics['width'].mean())       if len(fiber_metrics) else 0.0,
        'mean_straightness':  float(fiber_metrics['straightness'].mean())if len(fiber_metrics) else 0.0,
        'mean_alignment':     float(fiber_metrics['local_alignment'].mean()) if len(fiber_metrics) else 0.0,
        'mean_density':       float(fiber_metrics['local_fiber_density'].mean()) if len(fiber_metrics) else 0.0,
        'n_interaction_pairs':len(pipeline_result.interaction_pairs),
        'network_n_nodes':    nm['n_nodes'],
        'network_n_edges':    nm['n_edges'],
        'network_density':    nm['density'],
    }
    if tme_result.tacs_features:
        tf = tme_result.tacs_features
        summary_stats.update({
            'tacs1_count': tf['tacs1_count'],
            'tacs2_count': tf['tacs2_count'],
            'tacs3_count': tf['tacs3_count'],
            'tacs1_ratio': tf['tacs1_ratio'],
            'tacs2_ratio': tf['tacs2_ratio'],
            'tacs3_ratio': tf['tacs3_ratio'],
            'dominant_tacs': tf['dominant_tacs_type'],
            'mean_angle_to_tangent': tf.get('mean_angle_to_tangent'),
        })
    if tme_result.prognostic_scores:
        summary_stats['tme_risk_score'] = tme_result.prognostic_scores.get(
            'overall_tme_risk_score'
        )
    summary_stats.update(pipeline_result.prognostic_scores)

    pd.DataFrame([summary_stats]).to_csv(
        out / f'{sample_id}_summary.csv', index=False
    )
    print(f'  Exported to {out}')

    # ── Finalise hierarchy: global metadata on image node ────────────────────
    image_entry.metadata.update({
        'summary_stats':  summary_stats,
        'tacs_features':  tme_result.tacs_features,
        'network_metrics': nm,
        'pixel_size':     pixel_size,
    })

    print('\n' + '=' * 72)
    print('  CT-FIRE WORKFLOW COMPLETE')
    print('=' * 72)

    return {
        'sample_id':          sample_id,
        'pixel_size':         pixel_size,
        'shg_image':          shg_image,
        'registered_he':      registered_he,
        'fibers':             fibers,
        'fiber_result':       fiber_result,
        'fiber_metrics':      fiber_metrics,
        'cells':              cells,
        'tumor_regions':      tumor_regions,
        'tme_result':         tme_result,
        'pipeline_result':    pipeline_result,
        'network_results':    network_results,
        'tacs_zone_results':  tacs_zone_results,
        'summary_stats':      summary_stats,
        'output_dir':         out,
        'saved_figures': {
            'heatmap_length':    heatmap_paths.get('length'),
            'heatmap_alignment': heatmap_paths.get('local_alignment'),
            'heatmap_density':   heatmap_paths.get('local_fiber_density'),
            'overlay_tacs':      overlay_path,
        },
        'hierarchy':          hierarchy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(summary_stats: Dict) -> None:
    """Print a formatted summary statistics table."""
    rows = [
        ('Fibers extracted',    summary_stats['n_fibers']),
        ('Cells',               summary_stats['n_cells']),
        ('Tumor regions',       summary_stats['n_tumors']),
        ('Interaction pairs',   summary_stats['n_interaction_pairs']),
        ('Network nodes',       summary_stats['network_n_nodes']),
        ('Network edges',       summary_stats['network_n_edges']),
        ('Network density',     f"{summary_stats['network_density']:.3f}"),
        ('Mean length (µm)',    f"{summary_stats['mean_fiber_length']:.2f}"),
        ('Mean width (µm)',     f"{summary_stats['mean_fiber_width']:.2f}"),
        ('Mean straightness',   f"{summary_stats['mean_straightness']:.3f}"),
        ('Mean K-NN alignment', f"{summary_stats['mean_alignment']:.4f}"),
        ('Mean density (f/mm²)',f"{summary_stats['mean_density']:.1f}"),
    ]
    if 'tacs1_count' in summary_stats:
        rows += [
            ('TACS-1 count',  summary_stats['tacs1_count']),
            ('TACS-2 count',  summary_stats['tacs2_count']),
            ('TACS-3 count',  summary_stats['tacs3_count']),
            ('TACS-1 ratio',  f"{summary_stats['tacs1_ratio']*100:.1f}%"),
            ('TACS-2 ratio',  f"{summary_stats['tacs2_ratio']*100:.1f}%"),
            ('TACS-3 ratio',  f"{summary_stats['tacs3_ratio']*100:.1f}%"),
            ('Dominant TACS', summary_stats.get('dominant_tacs', 'n/a')),
        ]

    width = 44
    print()
    print('┌' + '─' * width + '┐')
    print('│{:^{w}}│'.format(' CT-FIRE Summary Statistics ', w=width))
    print('├' + '─' * 24 + '┬' + '─' * (width - 25) + '┤')
    for label, value in rows:
        print('│ {:<22} │ {:<{w}} │'.format(label, str(value), w=width - 27))
    print('└' + '─' * 24 + '┴' + '─' * (width - 25) + '┘')


def _print_fiber_properties_table(
    fiber_metrics: pd.DataFrame,
    max_rows: int = 25,
) -> None:
    """
    Print a sample of per-fiber property measurements, sorted by TACS type
    so TACS-3 (invasive) fibers appear first.
    """
    if len(fiber_metrics) == 0:
        print('  (no fiber metrics available)')
        return

    tacs_order = {'TACS-3': 0, 'TACS-2': 1, 'TACS-1': 2, None: 3}
    display_df = fiber_metrics.copy()
    display_df['_sort_key'] = display_df['tacs_type'].apply(
        lambda t: next((v for k, v in tacs_order.items() if k and t and k in str(t)), 3)
    )
    display_df = display_df.sort_values('_sort_key').drop(columns='_sort_key')

    cols = [
        'fiber_id', 'length', 'width', 'straightness', 'orientation',
        'distance_to_tumor', 'angle_to_tangent', 'local_alignment',
        'local_fiber_density', 'tacs_type',
    ]
    sample_df = display_df.head(max_rows)[
        [c for c in cols if c in display_df.columns]
    ].copy()

    fmt = {
        'length':              '{:.1f}',
        'width':               '{:.1f}',
        'straightness':        '{:.3f}',
        'orientation':         '{:.1f}',
        'distance_to_tumor':   '{:.1f}',
        'angle_to_tangent':    '{:.1f}',
        'local_alignment':     '{:.3f}',
        'local_fiber_density': '{:.1f}',
    }
    for col, f in fmt.items():
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].apply(
                lambda v: f.format(v) if pd.notna(v) else 'n/a'
            )

    print()
    print(f'  Per-fiber measurements  '
          f'(showing {len(sample_df)} of {len(fiber_metrics):,} fibers, sorted by TACS type)')
    print('  ' + '-' * 110)
    with pd.option_context(
        'display.max_columns', None,
        'display.width',       130,
        'display.max_colwidth', 14,
    ):
        print(sample_df.to_string(index=False))
    print()


def _print_tacs_distribution_table(fiber_metrics: pd.DataFrame) -> None:
    """Print a TACS type distribution summary table."""
    if 'tacs_type' not in fiber_metrics.columns:
        return
    total = len(fiber_metrics)
    in_zone = fiber_metrics['tacs_type'].notna().sum()

    print()
    print('  TACS distribution (fibers within 100 µm of tumor boundary,')
    print('  straightness >= 0.7):')
    print('  ┌──────────────────────────────────────────────────────┐')
    print(f'  │ {"TACS type":<18} {"Count":>6}  {"% of zone":>10}  '
          f'{"% of all":>9} │')
    print('  ├──────────────────────────────────────────────────────┤')
    for tacs_type in ['TACS-1', 'TACS-2', 'TACS-3']:
        count = int(fiber_metrics['tacs_type'].str.contains(
            tacs_type, na=False
        ).sum())
        pct_zone = count / max(in_zone, 1) * 100
        pct_all  = count / max(total, 1) * 100
        print(f'  │ {tacs_type:<18} {count:>6}  {pct_zone:>9.1f}%  '
              f'{pct_all:>8.1f}% │')
    not_classified = total - in_zone
    print('  ├──────────────────────────────────────────────────────┤')
    print(f'  │ {"Not classified":<18} {not_classified:>6}  {"—":>10}  '
          f'{not_classified/max(total,1)*100:>8.1f}% │')
    print(f'  │ {"Total":<18} {total:>6}                         │')
    print('  └──────────────────────────────────────────────────────┘')


def print_measurement_tables(results: Dict) -> None:
    """Print summary, per-fiber, and TACS distribution tables to stdout."""
    _print_summary_table(results['summary_stats'])
    _print_tacs_distribution_table(results['fiber_metrics'])
    _print_fiber_properties_table(results['fiber_metrics'])


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
      [0,0] SHG + fiber mask overlay  [0,1] fiber length heatmap
      [1,0] K-NN alignment heatmap    [1,1] TACS overlay
    """
    figs     = results['saved_figures']
    shg      = results['shg_image']
    fr       = results['fiber_result']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"CT-FIRE Results — {results['sample_id']}",
        fontsize=14, fontweight='bold',
    )

    # Panel [0,0]: SHG + fiber mask
    ax = axes[0, 0]
    ax.imshow(shg, cmap='gray')
    if fr is not None and hasattr(fr, 'fiber_mask'):
        mask_rgba       = np.zeros((*shg.shape[:2], 4), dtype=np.float32)
        mask_rgba[..., 0] = 1.0   # red channel
        mask_rgba[..., 3] = (fr.fiber_mask > 0).astype(np.float32) * 0.5
        ax.imshow(mask_rgba)
    ax.set_title('SHG + fiber mask', fontsize=11)
    ax.axis('off')

    # Panels [0,1], [1,0], [1,1]: saved heatmaps / overlay
    panel_defs = [
        (axes[0, 1], 'heatmap_length',    'Fiber length heatmap'),
        (axes[1, 0], 'heatmap_alignment', 'K-NN alignment heatmap'),
        (axes[1, 1], 'overlay_tacs',      'TACS overlay'),
    ]
    for ax, key, title in panel_defs:
        path = figs.get(key)
        if path is not None and Path(path).exists():
            ax.imshow(io.imread(path))
        else:
            ax.text(0.5, 0.5, f'{title}\n(file not found)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # TACS legend on overlay panel
    tacs_ax = axes[1, 1]
    legend_handles = [
        mpatches.Patch(color=np.array(get_tacs_color('TACS-3')) / 255,
                       label='TACS-3 (perpendicular, invasive)'),
        mpatches.Patch(color=np.array(get_tacs_color('TACS-2')) / 255,
                       label='TACS-2 (parallel)'),
        mpatches.Patch(color=np.array(get_tacs_color('TACS-1')) / 255,
                       label='TACS-1 (random/curly)'),
    ]
    tacs_ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=7,
        framealpha=0.7,
    )

    fig.tight_layout()
    plt.show()

    # Additional figure: fiber property distributions
    fm = results['fiber_metrics']
    if len(fm) > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig2.suptitle(
            f"Fiber Property Distributions — {results['sample_id']}",
            fontsize=12, fontweight='bold',
        )
        for ax, col, xlabel, color in [
            (axes2[0], 'length',      'Fiber length (µm)',              'steelblue'),
            (axes2[1], 'straightness','Straightness',                   'tomato'),
            (axes2[2], 'orientation', 'Absolute orientation (°)',       'goldenrod'),
        ]:
            vals = fm[col].dropna().values
            ax.hist(vals, bins=30, color=color, edgecolor='white', linewidth=0.5)
            ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.2,
                       label=f'mean={vals.mean():.2f}')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(fontsize=9)
            ax.set_title(col.capitalize(), fontsize=11)
        fig2.tight_layout()
        plt.show()

    _display_cell_segmentation_figure(results)
    _display_tacs_zone_figure(results)


def _display_tacs_zone_figure(results: Dict) -> None:
    """
    Plot TACS zone heatmaps (one figure per detected tumor region).

    Calls ``plot_tacs_heatmap`` from ``example_analyze_tacs_zone`` — the same
    two-panel visualisation (spatial heatmap + TACS distribution bar chart)
    used in the standalone TACS zone example.  The SHG+H&E composite image is
    passed as a background so tissue structure is visible behind the fiber
    scatter plot.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from example_analyze_tacs_zone import plot_tacs_heatmap  # type: ignore[import]

    tacs_zone_results = results.get('tacs_zone_results', [])
    if not tacs_zone_results:
        print('[TACS display] no TACS zone results to plot')
        return

    pixel_size    = results.get('pixel_size', 1.0)
    fibers        = results.get('fibers', [])
    thr_px        = int(round(100.0 / pixel_size))   # 100 µm → pixels
    shg_image     = results.get('shg_image')
    registered_he = results.get('registered_he')

    # Build SHG+H&E composite (same 60/40 blend used by _create_fiber_overlay)
    background = None
    if shg_image is not None:
        import cv2 as cv2 # noqa: PLC0415
        shg_norm = (shg_image / shg_image.max() * 255).astype(np.uint8)
        shg_rgb  = (
            cv2.cvtColor(shg_norm, cv2.COLOR_GRAY2RGB)
            if shg_image.ndim == 2 else shg_norm.copy()
        )
        if registered_he is not None:
            he_rgb = (
                registered_he if registered_he.ndim == 3
                else cv2.cvtColor(
                    (registered_he * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
                )
            )
            if he_rgb.shape[:2] != shg_rgb.shape[:2]:
                he_rgb = cv2.resize(he_rgb, (shg_rgb.shape[1], shg_rgb.shape[0]))
            background = cv2.addWeighted(shg_rgb, 0.6, he_rgb, 0.4, 0)
        else:
            background = shg_rgb

    for entry in tacs_zone_results:
        plot_tacs_heatmap(
            result                     = entry['result'],
            roi                        = entry['roi'],
            fiber_objects              = fibers,
            inside_roi_pixels          = None,
            inside_roi_fibers          = None,
            boundary_dist_threshold_px = thr_px,
            pixel_size                 = pixel_size,
            title                      = (
                f"TACS zone — {results['sample_id']}  "
                f"tumor {entry['tumor_idx']}  (CT-FIRE fibers)"
            ),
            background_image           = background,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def demo_project_io(results: Dict) -> None:
    """
    Demonstrate project-level save / load / export using the completed
    CT-FIRE workflow results.

    Three capabilities are shown:

    1. ``save_project`` — Snapshot the full TMEProject to disk as a set of
       JSON files (hierarchy, images, orientation maps, fiber populations,
       and a manifest).  The saved state is fully self-contained and can be
       shared or archived.

    2. ``load_project`` — Reconstruct a ``TMEProject`` from the snapshot.
       The hierarchy tree, fiber metadata (centerlines, TACS type, angles),
       and image metadata are restored exactly.  Pixel data is *not* reloaded
       by default (pass ``reload_images=True`` to load image arrays from
       their original paths).

    3. ``export_project_summary`` — Write a human-readable Excel workbook
       (and matching CSV files) with 7 sheets:
         • Project      — top-level sample metadata
         • Images       — per-image path, pixel size, modality
         • FiberPopulations — population-level orientation statistics
         • OrientationMaps  — region-level mean angle / alignment score
         • TACS_Summary — TACS-1 / -2 / -3 counts and percentages
         • Fibers       — per-fiber measurements incl. local density
           and local alignment score (KD-tree, ``local_radius`` µm)
         • SpatialGrid  — hexagonal spatial binning of fiber metrics
           over a ``grid_bin_size`` µm grid
    """
    from tme_quant.core.project import TMEProject

    out        = results['output_dir']
    sample_id  = results['sample_id']
    hierarchy  = results['hierarchy']

    # ── Build a minimal TMEProject to pass to the IO functions ───────────────
    # In a real pipeline the project would already exist; here we construct
    # one from the workflow's hierarchy so the demo is self-contained.
    project = TMEProject(
        name=f'CT-FIRE demo — {sample_id}',
    )
    project.hierarchy = hierarchy

    # Add every ImageEntry that lives in the hierarchy into project.images so
    # the images.json snapshot includes full path / modality / channel data.
    for img in hierarchy.get_objects_by_type(TMEType.IMAGE):
        project.images[img.object_id] = img

    # ── 1. SAVE ──────────────────────────────────────────────────────────────
    print('\n' + '─' * 72)
    print('  Project IO — Demo')
    print('─' * 72)

    snapshot_dir = out / f'{sample_id}_snapshot'
    print(f'\n[IO 1/3] Saving project snapshot → {snapshot_dir}')
    saved_path = save_project(project, snapshot_dir, overwrite=True)
    print(f'  Saved to : {saved_path}')
    import json, os
    saved_files = sorted(os.listdir(saved_path))
    print(f'  Files    : {saved_files}')
    # Peek at the manifest to confirm what was written
    with open(saved_path / 'project_manifest.json') as _f:
        manifest = json.load(_f)
    print(f'  Manifest : name={manifest["name"]!r}  '
          f'saved_at={manifest.get("saved_at", "n/a")}')

    # ── 2. LOAD ──────────────────────────────────────────────────────────────
    print(f'\n[IO 2/3] Loading project from {snapshot_dir}')
    restored = load_project(snapshot_dir, reload_images=False)
    r_fibers = restored.hierarchy.get_objects_by_type(TMEType.FIBER)
    r_tacs3  = [f for f in r_fibers if getattr(f, 'tacs_type', None) == 'TACS-3']
    r_tacs2  = [f for f in r_fibers if getattr(f, 'tacs_type', None) == 'TACS-2']
    print(f'  Restored project name: {restored.name}')
    print(f'  Restored fibers     : {len(r_fibers)}')
    print(f'  Restored TACS-3     : {len(r_tacs3)}')
    print(f'  Restored TACS-2     : {len(r_tacs2)}')
    # Spot-check: first fiber centerline should survive the JSON round-trip
    if r_fibers:
        _f0 = r_fibers[0]
        _cl = getattr(_f0, 'centerline', None)
        _cl_info = f'shape={_cl.shape}' if _cl is not None else 'None'
        print(f'  Fiber[0] id={_f0.object_id!r}  '
              f'tacs_type={getattr(_f0, "tacs_type", None)!r}  '
              f'centerline {_cl_info}')

    # ── 3. EXPORT ─────────────────────────────────────────────────────────────
    export_dir = out / f'{sample_id}_summary_export'
    print(f'\n[IO 3/3] Exporting human-readable summary → {export_dir}')
    exported = export_project_summary(
        project,
        output_dir=export_dir,
        formats=['csv', 'excel'],
        prefix=sample_id,
        local_radius=50.0,    # KD-tree radius for local density / alignment
        grid_bin_size=100.0,  # spatial grid cell size in µm
    )
    print('  Exported files:')
    for table, path in sorted(exported.items()):
        print(f'    {table:<20} → {Path(path).name}')

    print('\n  Project IO demo complete.')
    print('─' * 72)


def display_results(results: Dict) -> None:
    """Print measurement tables and show the figure panels."""
    print_measurement_tables(results)
    display_figure_panel(results)


if __name__ == '__main__':
    results = workflow_ctfire_complete(
        he_image_path='data/patient_001_HE.tif',
        shg_image_path='data/patient_001_SHG.tif',
        output_dir='output/patient_001_ctfire',
        pixel_size=0.5,           # 0.5 µm/px (estimated from nucleus diameter; matches StarDist 2D_versatile_he training resolution)
        sample_id='patient_001_ctfire',
        skip_registration=True, # set True if H&E already registered to SHG
        skip_tumor_detection=True,  # use synthetic ellipse ROI for TACS demo
    )
    display_results(results)

    # ── Hierarchy queries ─────────────────────────────────────────────────────
    h          = results['hierarchy']
    _sid       = results['sample_id']
    all_fibers = h.get_objects_by_type(TMEType.FIBER)
    all_cells  = h.get_objects_by_type(TMEType.CELL)
    tacs3      = [f for f in all_fibers if getattr(f, 'tacs_type', None) == 'TACS-3']
    tacs2      = [f for f in all_fibers if getattr(f, 'tacs_type', None) == 'TACS-2']
    boundary_f = [f for f in all_fibers if getattr(f, 'in_tumor_boundary', False)]
    img_node   = h.get_object(f'{_sid}_image')
    net_m      = img_node.metadata.get('network_metrics', {}) if img_node else {}
    print('\n── Hierarchy queries ──────────────────────────────────────────────')
    print(f'  Total fibers      : {len(all_fibers)}')
    print(f'  Total cells       : {len(all_cells)}')
    print(f'  TACS-3 (invasive) : {len(tacs3)}')
    print(f'  TACS-2 (parallel) : {len(tacs2)}')
    print(f'  Boundary-zone     : {len(boundary_f)}')
    print(f'  Network           : {net_m.get("n_nodes", 0)} nodes, '
          f'{net_m.get("n_edges", 0)} edges')
    issues = h.validate_hierarchy()
    print(f'  Hierarchy valid   : {len(issues) == 0}  ({len(issues)} issue(s))')

    # ── Project IO demo ───────────────────────────────────────────────────────
    demo_project_io(results)
