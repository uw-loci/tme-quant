"""
ROI + CurveAlign Relative Orientation Example
==============================================

Demonstrates loading ROIs from two sources and computing per-fiber
boundary-relative orientation (angle_to_tangent) using CurveAlign.

Two ROI sources compared side-by-side
--------------------------------------
  Source A — Automatic:
    TumorRegion produced by RegionManager.detect_tumor_regions()
    (DBSCAN clustering on StarDist-segmented cells from the H&E image).
    Converted into an ROIManager-managed annotation via
    ROIManager.from_tumor_region().

  Source B — Manual:
    Polygon drawn by the user (e.g. in napari or QuPath) and imported
    directly into ROIManager.add_polygon() or from_qupath_geojson().

Workflow
--------
  1. Register H&E to SHG  (HESHGRegistration)
  2. Segment cells         (StarDist on registered H&E)
  3. Auto-detect boundary  (RegionManager DBSCAN → TumorRegion)
  4. Manual boundary       (ROIManager.add_polygon)
  5. Load both into ROIManager and attach to the TME hierarchy
  6. CurveAlign orientation (FiberAnalyzer.analyze_orientation_2d on SHG)
  7. Relative orientation  (compute_angle_to_boundary_normal per orientation point,
                            for each of the two ROI boundaries)
  8. Compare + summarise   (mean angle_to_tangent, TACS-like distribution per ROI)

How TumorRegion ↔ ROIManager work together
--------------------------------------------
  RegionManager returns List[TumorRegion].  Each TumorRegion carries a
  Geometry (coordinates array).  ROIManager.from_tumor_region() reads
  those coordinates and creates an ROIObject so the boundary lives in the
  unified annotation store alongside manually drawn boundaries.

  Both ROIs are then passed identically to the orientation analysis step —
  there is no distinction at the analysis level between automatic and manual.

Run
---
  # With editable install:
  python src/examples/roi_curvealign_orientation_example.py

  # With PYTHONPATH:
  PYTHONPATH=src python src/examples/roi_curvealign_orientation_example.py

Requirements
------------
  Core dependencies only (numpy, scipy, scikit-image, shapely, matplotlib,
  opencv, tifffile, imageio, openpyxl).
  StarDist (dl group) for cell segmentation.
  Image files:  data/patient_001_HE.tif   (H&E, any RGB TIFF)
                data/patient_001_SHG.tif  (SHG, single-channel TIFF)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import io

# ── TMEQuant imports ─────────────────────────────────────────────────────────

# Registration
from tme_quant.image_registration.methods.intensity_based import HESHGRegistration
from tme_quant.image_registration.config import RegistrationParams, TransformType

# Fiber / orientation analysis
from tme_quant.fiber_analysis import FiberOrientationAnalyzer
from tme_quant.fiber_analysis.config import CurveAlignParams

# Orientation relative to ROI boundary
from tme_quant.tme_analysis.utils import compute_orientation_relative_to_roi

# Cell segmentation
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

# Automated tumor detection
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import (
    TumorDetectionParams, TumorDetectionMethod,
)

# ROI and hierarchy management
from tme_quant.core.roi_manager import ROIManager, ROIObject
from tme_quant.core.hierarchy import TMEHierarchy
from tme_quant.core.base_models import TMEObject, TMEType
from tme_quant.core.image_entry import ImageEntry

# TACS colour map (for visualisation)
from tme_quant.fiber_analysis.tacs import get_tacs_color


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_TACS_COLORS = {
    'TACS-1-like': (0.2, 0.4, 0.8),   # blue
    'TACS-2-like': (0.2, 0.7, 0.2),   # green
    'TACS-3-like': (0.85, 0.15, 0.15), # red
}


def visualise_results(
    shg_image:       np.ndarray,
    orientation_map: np.ndarray,
    roi_auto:        ROIObject,
    roi_manual:      ROIObject,
    result_auto:     dict,
    result_manual:   dict,
    output_dir:      Path,
    sample_id:       str,
) -> None:
    """
    Save a 3-panel figure:
      Left  — SHG + orientation map overlay with TACS-coloured points
      Centre — auto boundary with angle_to_tangent heatmap in the zone
      Right  — manual boundary with angle_to_tangent heatmap in the zone
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{sample_id} — CurveAlign boundary-relative orientation",
        fontsize=13, fontweight='bold',
    )

    # Panel 1: SHG + both boundaries + TACS-coloured orientation points
    ax = axes[0]
    ax.imshow(shg_image, cmap='gray', alpha=0.7)
    ax.set_title("SHG + boundaries + TACS-like\n(coloured orientation points)")

    for result, roi in [(result_auto, roi_auto), (result_manual, roi_manual)]:
        for pt in result.get('points', []):
            color = _TACS_COLORS.get(pt['tacs_like'], (0.6, 0.6, 0.6))
            ax.plot(pt['x'], pt['y'], '.', color=color, markersize=2, alpha=0.6)

    # Draw boundaries
    for roi, color, ls, lw in [
        (roi_auto,   'cyan',   '--', 2),
        (roi_manual, 'yellow', '-',  2),
    ]:
        c = roi.coordinates
        if c is not None:
            ax.plot(
                np.append(c[:, 0], c[0, 0]),
                np.append(c[:, 1], c[0, 1]),
                color=color, linestyle=ls, linewidth=lw,
                label=roi.label,
            )
    ax.legend(loc='upper right', fontsize=7)
    # TACS legend
    handles = [mpatches.Patch(color=v, label=k) for k, v in _TACS_COLORS.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=7)
    ax.axis('off')

    # Panels 2 & 3: angle_to_tangent heatmap per ROI
    for idx, (result, roi, title) in enumerate([
        (result_auto,   roi_auto,   f"Auto boundary\n({roi_auto.label})"),
        (result_manual, roi_manual, f"Manual boundary\n({roi_manual.label})"),
    ]):
        ax = axes[idx + 1]
        ax.imshow(shg_image, cmap='gray', alpha=0.5)

        pts = result.get('points', [])
        if pts:
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            vs = [p['angle_to_tangent'] for p in pts]
            sc = ax.scatter(xs, ys, c=vs, cmap='RdYlGn_r',
                            vmin=0, vmax=90, s=4, alpha=0.8)
            fig.colorbar(sc, ax=ax, label='angle_to_tangent (°)',
                         fraction=0.03, pad=0.02)

        # Draw this ROI's boundary
        c = roi.coordinates
        if c is not None:
            ax.plot(
                np.append(c[:, 0], c[0, 0]),
                np.append(c[:, 1], c[0, 1]),
                color='white', linewidth=2,
            )

        mean_att  = result.get('mean_angle_to_tangent', np.nan)
        n_pts     = result.get('n_points_in_zone', 0)
        tacs_dist = result.get('tacs_distribution', {})
        t3        = tacs_dist.get('TACS-3-like', 0)
        t3_pct    = 100 * t3 / n_pts if n_pts > 0 else 0
        ax.set_title(
            f"{title}\n"
            f"mean angle_to_tangent={mean_att:.1f}°  "
            f"TACS-3-like={t3_pct:.0f}%  n={n_pts}",
            fontsize=8,
        )
        ax.axis('off')

    fig.tight_layout()
    out_path = output_dir / f"{sample_id}_roi_curvealign_orientation.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

def run_roi_curvealign_orientation(
    he_image_path:  str,
    shg_image_path: str,
    output_dir:     str  = "output/roi_orientation_demo",
    pixel_size:     float = 0.5,
    sample_id:      str  = "patient_001",
    # Manual polygon — provide your own or leave None to use synthetic coords
    manual_polygon: Optional[List[Tuple[float, float]]] = None,
) -> dict:
    """
    Full pipeline: registration → segmentation → auto boundary → manual
    boundary → CurveAlign → boundary-relative orientation comparison.

    Parameters
    ----------
    he_image_path  : Path to H&E RGB TIFF.
    shg_image_path : Path to SHG single-channel TIFF.
    output_dir     : Directory for output figures and CSVs.
    pixel_size     : µm per pixel.
    sample_id      : Used in output file names.
    manual_polygon : List of (x, y) pixel-coordinate vertices defining a
                     manually drawn tumor boundary.  If None a synthetic
                     elliptical polygon is generated from the SHG image
                     centre so the example runs without real annotation data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ROI + CurveAlign Boundary-Relative Orientation Demo")
    print(f"Sample: {sample_id}")
    print("=" * 72)

    # ── [1/7] Load images ────────────────────────────────────────────────────
    print("\n[1/7] Loading images...")
    he_image  = io.imread(he_image_path)
    shg_image = io.imread(shg_image_path)
    if shg_image.ndim == 3:
        shg_image = shg_image[..., 0]   # take first channel if multi-channel
    print(f"  ✓ H&E: {he_image.shape},  SHG: {shg_image.shape}")

    H, W = shg_image.shape[:2]

    # ── [2/7] Register H&E → SHG ─────────────────────────────────────────────
    print("\n[2/7] Registering H&E to SHG...")
    reg_params    = RegistrationParams(
        transform_type=TransformType.AFFINE,
        use_multiresolution=True, pyramid_levels=3, num_iterations=200,
    )
    registration  = HESHGRegistration()
    reg_result    = registration.register(shg_image, he_image, reg_params)
    registered_he = reg_result.registered_image
    print(f"  ✓ MI score: {reg_result.mutual_information:.4f}")
    io.imsave(
        output_dir / f"{sample_id}_HE_registered.tif",
        (registered_he * 255).astype(np.uint8),
    )

    # ── [3/7] Cell segmentation (for auto boundary) ──────────────────────────
    print("\n[3/7] Cell segmentation (StarDist on registered H&E)...")
    seg_params    = SegmentationParams(
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
    print(f"  ✓ {len(cells)} cells segmented")

    # ── [4/7] Auto boundary via RegionManager ────────────────────────────────
    print("\n[4/7] Automated tumor boundary (RegionManager DBSCAN)...")
    tumor_params  = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        clustering_algorithm='dbscan',
        dbscan_eps=80.0,
        dbscan_min_samples=8,
        min_tumor_area=5000.0,
        smooth_boundary=True,
    )
    tme_analyzer  = TMEAnalyzer()
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    print(f"  ✓ {len(tumor_regions)} tumor region(s) detected")

    if not tumor_regions:
        print("  WARNING: No tumor regions detected. "
              "Using image-centre synthetic boundary for the demo.")
        # Synthetic fallback: ellipse centred on image
        from tme_quant.core.base_models import Geometry, GeometryType
        angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
        cx, cy = W//2, H//2
        rx, ry = W//5, H//6
        coords = np.column_stack([cx + rx*np.cos(angles), cy + ry*np.sin(angles)])
        from tme_quant.core.tme_models.tumor_model import TumorRegion
        fallback = TumorRegion(
            object_id="synthetic_auto",
            name="Synthetic auto boundary",
            geometry=Geometry(type=GeometryType.POLYGON, coordinates=coords),
        )
        tumor_regions = [fallback]

    # Use the largest detected tumor region for the comparison
    if not tumor_regions:
        raise RuntimeError(
            "No tumor regions available even after synthetic fallback. "
            "This should not happen — check the fallback block above."
        )
    auto_region = max(
        tumor_regions,
        key=lambda r: (r.geometry.area()
                       if r.geometry is not None and r.geometry.bounds is not None
                       else 0.0),
    )
    print(f"  ✓ Using region '{auto_region.object_id}' "
          f"(area ≈ {auto_region.geometry.area():.0f} px²)")

    # ── [5/7] Build ROIManager — load both boundary types ────────────────────
    print("\n[5/7] Building ROIManager — loading auto and manual boundaries...")

    roi_mgr = ROIManager(image_id=sample_id, pixel_size=pixel_size)

    # Source A: convert the TumorRegion → ROIObject
    roi_auto = roi_mgr.from_tumor_region(
        auto_region,
        label="Auto boundary (DBSCAN)",
        locked=True,
    )
    if roi_auto is None:
        raise RuntimeError("Could not convert auto TumorRegion to ROIObject.")

    # Source B: manual polygon
    # Use the caller-supplied vertices, or build a synthetic one for the demo.
    if manual_polygon is None:
        # Synthetic manual boundary: a rectangle offset from the auto centre
        cx, cy = roi_auto.centroid
        w_px   = max(100, min(W // 4, 300))
        h_px   = max(80,  min(H // 4, 240))
        # Offset slightly so the two boundaries differ visibly
        cx_m   = min(max(cx + w_px // 3, w_px // 2), W - w_px // 2)
        cy_m   = min(max(cy + h_px // 3, h_px // 2), H - h_px // 2)
        manual_polygon = [
            (cx_m - w_px // 2, cy_m - h_px // 2),
            (cx_m + w_px // 2, cy_m - h_px // 2),
            (cx_m + w_px // 2, cy_m + h_px // 2),
            (cx_m - w_px // 2, cy_m + h_px // 2),
        ]
        print(f"  (No manual_polygon supplied — using synthetic rectangle "
              f"centred at ({cx_m:.0f}, {cy_m:.0f}))")

    roi_manual = roi_mgr.add_polygon(
        vertices=manual_polygon,
        annotation_type="tumor_boundary",
        label="Manual boundary (user polygon)",
        locked=False,
    )

    print(f"  ✓ ROIManager: {len(roi_mgr)} annotations loaded")
    print(f"    Source A (auto):   {roi_auto}")
    print(f"    Source B (manual): {roi_manual}")

    # Wire ROIs into the TME hierarchy
    root      = TMEObject(object_id=sample_id, tme_type=TMEType.PROJECT)
    img_entry = ImageEntry(
        object_id=f"{sample_id}_image",
        image_data=shg_image,
        pixel_size=(pixel_size, pixel_size),
        modality="SHG",
        parent=root,
    )
    hierarchy = TMEHierarchy(root=root)
    hierarchy.add_object(img_entry)
    roi_mgr.attach_to_hierarchy(hierarchy, parent_id=img_entry.object_id)

    all_annotations = hierarchy.get_objects_by_type(TMEType.ANNOTATION)
    print(f"  ✓ Hierarchy: {len(all_annotations)} annotations "
          f"under '{img_entry.object_id}'")

    # ── [6/7] CurveAlign orientation on SHG ──────────────────────────────────
    print("\n[6/7] CurveAlign orientation analysis on SHG image...")
    ca_params = CurveAlignParams(
        pixel_size=pixel_size,
        window_size=64,
        overlap=0.5,
        curvelet_levels=4,
        curvelet_angles=8,
        compute_coherency=True,
        compute_energy=True,
        keep_values=['angles', 'alignment', 'energy'],
        compute_statistics=True,
    )
    fiber_analyzer = FiberOrientationAnalyzer()
    orient_result  = fiber_analyzer.analyze_2d(
        shg_image.astype(np.float32), ca_params
    )
    print(f"  ✓ Mean orientation: {orient_result.mean_orientation:.2f}°  "
          f"alignment: {orient_result.mean_alignment:.4f}  "
          f"windows: {orient_result.n_windows_analyzed}")

    # ── [7/7] Boundary-relative orientation for each ROI ─────────────────────
    print("\n[7/7] Computing boundary-relative orientation...")

    result_auto   = compute_orientation_relative_to_roi(
        orientation_map=orient_result.orientation_map,
        alignment_map=orient_result.alignment_map,
        roi=roi_auto,
        pixel_size=pixel_size,
        tacs_zone_width=100.0,
        subsample=2,
    )
    result_manual = compute_orientation_relative_to_roi(
        orientation_map=orient_result.orientation_map,
        alignment_map=orient_result.alignment_map,
        roi=roi_manual,
        pixel_size=pixel_size,
        tacs_zone_width=100.0,
        subsample=2,
    )

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPARISON: Auto boundary vs Manual boundary")
    print("=" * 72)
    print(f"\n  {'Metric':<35} {'Auto':>12} {'Manual':>12}")
    print("  " + "─" * 62)

    for label, r in [("Auto (DBSCAN)", result_auto),
                     ("Manual (polygon)", result_manual)]:
        n       = r.get('n_points_in_zone', 0)
        mean_at = r.get('mean_angle_to_tangent', np.nan)
        std_at  = r.get('std_angle_to_tangent',  np.nan)
        tacs    = r.get('tacs_distribution', {})
        t3_pct  = 100 * tacs.get('TACS-3-like', 0) / n if n > 0 else 0

        print(f"\n  {label}")
        print(f"    Points in TACS zone (100µm):  {n:>6}")
        print(f"    Mean angle_to_tangent (°):    {mean_at:>6.1f}  ± {std_at:.1f}")
        print(f"    TACS-1-like (30–60°):         "
              f"{tacs.get('TACS-1-like',0):>6d}  "
              f"({100*tacs.get('TACS-1-like',0)/n:.1f}%)" if n > 0 else "    n/a")
        print(f"    TACS-2-like (parallel, 0–30°):{tacs.get('TACS-2-like',0):>6d}  "
              f"({100*tacs.get('TACS-2-like',0)/n:.1f}%)" if n > 0 else "    n/a")
        print(f"    TACS-3-like (perp, 60–90°):   {tacs.get('TACS-3-like',0):>6d}  "
              f"({t3_pct:.1f}%)")

    print("\n  Interpretation:")
    mean_auto   = result_auto.get('mean_angle_to_tangent', np.nan)
    mean_manual = result_manual.get('mean_angle_to_tangent', np.nan)
    if not np.isnan(mean_auto) and not np.isnan(mean_manual):
        diff = abs(mean_auto - mean_manual)
        print(f"  Δ mean_angle_to_tangent between boundaries: {diff:.1f}°")
        if diff < 10:
            print("  → Both boundaries give consistent orientation measurements.")
        else:
            print("  → Boundaries differ; manual review recommended to confirm "
                  "which boundary is more representative.")

    # ── Save figure ───────────────────────────────────────────────────────────
    visualise_results(
        shg_image, orient_result.orientation_map,
        roi_auto, roi_manual, result_auto, result_manual,
        output_dir, sample_id,
    )

    # ── Export ROIs to GeoJSON ────────────────────────────────────────────────
    roi_mgr.save_geojson(output_dir / f"{sample_id}_roi_annotations.geojson")
    print(f"  ✓ ROI annotations saved → "
          f"{output_dir / f'{sample_id}_roi_annotations.geojson'}")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)

    return {
        'roi_auto':     roi_auto,
        'roi_manual':   roi_manual,
        'result_auto':  result_auto,
        'result_manual':result_manual,
        'orient_result': orient_result,
        'hierarchy':    hierarchy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_roi_curvealign_orientation(
        he_image_path  = "data/patient_001_HE.tif",
        shg_image_path = "data/patient_001_SHG.tif",
        output_dir     = "output/roi_orientation_demo",
        pixel_size     = 0.5,
        sample_id      = "patient_001",
        # manual_polygon: leave None to use a synthetic rectangle,
        # or supply your own, e.g.:
        # manual_polygon = [(120,80),(320,80),(320,280),(120,280)],
    )