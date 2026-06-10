"""
TMEQuant Example — curvealign_ctfire_mode_pipeline
===================================================

Demonstrates the ``curvealign_ctfire_mode_pipeline`` on a real SHG image
with a binary boundary mask.  Two scenarios are shown:

**Scenario 1 — Full CT-FIRE** (``use_ct_reconstruction=True``, default):

  • Curvelet reconstruction (``ct_reconstruction``) → FIRE fiber extraction.
  • Requires curvelops / curvelet library in addition to ``ctfire_py``.
  • Full boundary analysis, TACS classification, TMEHierarchy, TACS viewer.

**Scenario 2 — FIRE only** (``use_ct_reconstruction=False``):

  • ``fire_2d_angle`` called directly on the normalised image.
  • **No curvelops or curvelet transform library required** — only
    ``ctfire_py`` must be installed.  Useful for pre-processed images,
    non-SHG modalities, or when isolating FIRE from curvelet effects.
  • Same boundary analysis, TACS classification, hierarchy, and viewer.

Both scenarios share helper functions and produce matching output files.
Output files are written to an ``output/`` folder next to this script.

Prerequisites
-------------
  - ``ctfire_py`` must be available in the active Python environment.
    See the Prerequisites section of ``curvealign_ctfireMode_pipeline.py``
    for build and install instructions (MSYS2 UCRT64 / .venv-curvelops).
  - Scenario 1 additionally requires curvelops (or another curvelet backend).

  - Real image files at:
      H:/GitHub.06.2022/tme-quant/tests/test_images/real1.tif
      H:/GitHub.06.2022/tme-quant/tests/test_images/CA_Boundary/mask_real1.tiff

Usage
-----
  python examples/example_curvealign_ctfire_pipeline.py                        # both (default)
  python examples/example_curvealign_ctfire_pipeline.py --scenario ctfire      # CT-FIRE only
  python examples/example_curvealign_ctfire_pipeline.py --scenario fire-only   # FIRE-only
  python examples/example_curvealign_ctfire_pipeline.py --scenario both        # both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")   # interactive; change to "Agg" for headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons, RadioButtons
import numpy as np
import pandas as pd
import tifffile

from tme_quant.tme_analysis.pipelines import curvealign_ctfire_mode_pipeline
from tme_quant.fiber_analysis.utils.boundary_tif_utils import (
    extract_boundary_coords_from_mask,
)
from tme_quant.fiber_analysis.visualization.draw_utils import (
    generate_fiber_overlay,
    generate_fiber_heatmap,
)
from tme_quant.fiber_analysis.io import export_dataframe_to_excel
from tme_quant import TMEHierarchy
from tme_quant.core.image_entry import ImageEntry
from tme_quant.core.base_models import Geometry, GeometryType, TMEType
from tme_quant.core.tme_objects.fiber_objects import FiberObject
from tme_quant.core.roi_manager import ROIObject
from tme_quant.fiber_analysis.tacs import classify_fiber_tacs

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _resolve_test_path(win_path: str) -> Path:
    """Translate Windows H:/ paths to /mnt/h/ on Linux/WSL."""
    if sys.platform.startswith("linux"):
        drive, rest = win_path[0].lower(), win_path[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(win_path)


REAL_IMAGE_PATH = _resolve_test_path(
    "H:/GitHub.06.2022/tme-quant/tests/test_images/real1.tif"
)
REAL_MASK_PATH = _resolve_test_path(
    "H:/GitHub.06.2022/tme-quant/tests/test_images/CA_Boundary/mask_real1.tiff"
)


# ── Image loaders ─────────────────────────────────────────────────────────────

def _load_grayscale(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    return img.astype(np.float32)


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = tifffile.imread(str(path))
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] <= 4 else mask[0]
    return (mask > 0).astype(np.uint8)


# ── Save figures + xlsx ───────────────────────────────────────────────────────

def _save_figures_and_xlsx(
    tag: str,
    img: np.ndarray,
    result,
    coordinates: dict | None = None,
) -> None:
    """Save overlay figure, heatmap figure, and xlsx to OUT_DIR."""
    if result is None:
        print(f"  [{tag}] No result — skipping figures and xlsx.\n")
        return

    fs                = result.fiber_structure
    boundary          = result.boundary_measurement
    in_curvs_flag     = result.in_curvs_flag
    nearest_angles    = result.nearest_angles
    fiber_features_df = result.fiber_features_df
    tif_boundary      = 3 if boundary else 0

    angles = fs["angle"].values

    if in_curvs_flag is not None:
        out_curvs_flag = ~in_curvs_flag
    else:
        in_curvs_flag  = np.ones(len(fs), dtype=bool)
        out_curvs_flag = np.zeros(len(fs), dtype=bool)

    nearest_angles_arr = (
        np.asarray(nearest_angles) if nearest_angles is not None else angles
    )

    # Reconstruct measured_boundary with column names expected by generate_fiber_overlay.
    # boundary_point_row/col are stored in MATLAB (x,y) order (col→row label is swapped).
    measured_boundary = None
    if boundary and "boundary_point_row" in fiber_features_df.columns:
        measured_boundary = fiber_features_df[
            ["nearest_distance_to_boundary", "inside_epicenter_region",
             "nearest_relative_boundary_angle", "extension_point_distance",
             "extension_point_angle", "boundary_point_col", "boundary_point_row"]
        ].rename(columns={
            "nearest_distance_to_boundary":   "nearest_boundary_distance",
            "inside_epicenter_region":         "nearest_region_distance",
            "nearest_relative_boundary_angle": "nearest_boundary_angle",
        })

    # ── Overlay figure ────────────────────────────────────────────────────────
    fig_ov, ax_ov = generate_fiber_overlay(
        img=img,
        fiber_structure=fs,
        coordinates=coordinates,
        in_curvs_flag=in_curvs_flag,
        out_curvs_flag=out_curvs_flag,
        nearest_angles=nearest_angles_arr,
        measured_boundary=measured_boundary,
        fiber_mode=1,               # CT-FIRE mode → individual fiber segments
        tif_boundary=tif_boundary,
        boundary_measurement=boundary,
        make_associations=boundary,
    )
    ax_ov.set_title(f"{tag} — CT-FIRE overlay", fontsize=9)
    overlay_path = OUT_DIR / f"{tag}_overlay.png"
    fig_ov.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"  Overlay saved  -> {overlay_path}")

    # ── Heatmap figure ────────────────────────────────────────────────────────
    distances = (
        fiber_features_df["nearest_distance_to_boundary"].values
        if "nearest_distance_to_boundary" in fiber_features_df.columns
        else None
    )
    fig_hm, *_ = generate_fiber_heatmap(
        img=img,
        fiber_structure=fs,
        in_curvs_flag=in_curvs_flag,
        angles=nearest_angles_arr,
        distances=distances,
        tif_boundary=tif_boundary,
        boundary_measurement=boundary,
    )
    fig_hm.axes[0].set_title(f"{tag} — angle heatmap", fontsize=9)
    heatmap_path = OUT_DIR / f"{tag}_heatmap.png"
    fig_hm.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig_hm)
    print(f"  Heatmap saved  -> {heatmap_path}")

    # ── xlsx export ───────────────────────────────────────────────────────────
    xlsx_path = str(OUT_DIR / f"{tag}_results.xlsx")
    export_dataframe_to_excel(fiber_features_df, xlsx_path, sheet_name="fiber_features")
    if result.roi_summary_df is not None and not result.roi_summary_df.empty:
        export_dataframe_to_excel(
            result.roi_summary_df, xlsx_path, sheet_name="roi_summary", mode="a"
        )
    print(f"  Results saved  -> {xlsx_path}\n")


# ── ROI assignment ────────────────────────────────────────────────────────────

def _assign_fibers_to_rois(
    fiber_objects: list[FiberObject],
    coordinates: dict,
) -> dict[str, object]:
    """Return {fobj.object_id: roi_key} by assigning each fiber to its nearest boundary contour.

    Uses minimum Euclidean distance from the fiber's center_point to any vertex
    on each contour.  Fibers in the stroma (outside all polygons) are correctly
    handled because distance-to-contour works regardless of containment.
    """
    roi_keys   = list(coordinates.keys())
    roi_arrays = [np.asarray(coordinates[k]) for k in roi_keys]
    result: dict[str, object] = {}
    for fobj in fiber_objects:
        if fobj.centerline is None or len(fobj.centerline) == 0:
            result[fobj.object_id] = roi_keys[0] if roi_keys else None
            continue
        # centerline stores (row, col); boundary coords also store (row, col)
        fiber_row = float(fobj.centerline[0, 0])
        fiber_col = float(fobj.centerline[0, 1])
        best_key, best_dist = None, float("inf")
        for key, arr in zip(roi_keys, roi_arrays):
            d = float(np.sqrt((arr[:, 0] - fiber_row) ** 2 + (arr[:, 1] - fiber_col) ** 2).min())
            if d < best_dist:
                best_dist, best_key = d, key
        result[fobj.object_id] = best_key
    return result


# ── TACS + hierarchy integration ──────────────────────────────────────────────

def _tacs_hierarchy_integration_ctfire(
    tag: str,
    img: np.ndarray,
    result,
    distance_threshold: float = 100.0,
    pixel_size: float = 1.0,
    coordinates: dict | None = None,
) -> tuple[TMEHierarchy, dict] | None:
    """Convert CT-FIRE results into FiberObject nodes, classify TACS, build TMEHierarchy.

    Each row in ``fiber_structure`` becomes one ``FiberObject`` node with full
    CT-FIRE morphology (``length``, ``width``, ``straightness`` = end/total length).
    ``tacs_type`` is set from the pre-computed boundary angle stored in
    ``fiber_features_df["nearest_relative_boundary_angle"]`` and real straightness.

    When ``coordinates`` is provided, an ``ROIObject`` node is created for each
    boundary contour and fibers are parented under their nearest ROI in the tree:
    ``ImageEntry → ROIObject → FiberObject``.

    Parameters
    ----------
    distance_threshold :
        TACS zone width in pixels — must match the ``distance_threshold`` used
        in the pipeline call.
    pixel_size :
        Micrometres per pixel; scales ``length`` and ``width`` to physical units.
    coordinates :
        Boundary contour dict from ``extract_boundary_coords_from_mask``.

    Returns
    -------
    (hierarchy, fiber_roi_map) or None
    """
    if result is None:
        print(f"  [{tag}] No result — skipping hierarchy integration.\n")
        return None

    print(f"\n{'-' * 60}")
    print(f"[{tag}] TACS + hierarchy integration")
    print(f"{'-' * 60}")

    fs           = result.fiber_structure
    ffd          = result.fiber_features_df
    has_boundary = result.boundary_measurement

    # ── 1. Build FiberObject nodes ────────────────────────────────────────────
    fiber_objects: list[FiberObject] = []
    for i, (_, frow) in enumerate(fs.iterrows()):
        feat = ffd.iloc[i] if i < len(ffd) else None

        nb_dist          = None
        in_epictr        = None
        angle_to_tangent = None
        bdry_pt: np.ndarray | None = None

        if feat is not None:
            raw_dist  = feat.get("nearest_distance_to_boundary")
            raw_epict = feat.get("inside_epicenter_region")
            raw_angle = feat.get("nearest_relative_boundary_angle")
            # boundary_point_row/col naming is swapped (MATLAB x,y convention):
            #   boundary_point_col stores the image row of the boundary point
            #   boundary_point_row stores the image col of the boundary point
            raw_bpt_r = feat.get("boundary_point_col")   # image row
            raw_bpt_c = feat.get("boundary_point_row")   # image col

            if raw_dist is not None and not pd.isna(raw_dist):
                nb_dist = float(raw_dist)
            if raw_epict is not None:
                in_epictr = bool(raw_epict)
            if raw_angle is not None and not pd.isna(raw_angle):
                angle_to_tangent = float(raw_angle)
            if (raw_bpt_r is not None and not pd.isna(raw_bpt_r)
                    and raw_bpt_c is not None and not pd.isna(raw_bpt_c)):
                bdry_pt = np.array([float(raw_bpt_r), float(raw_bpt_c)])  # (row, col)

        # CT-FIRE morphology
        total_len    = float(frow.get("total_length", 1.0) or 1.0)
        end_len      = float(frow.get("end_length",   total_len) or total_len)
        straightness = end_len / total_len if total_len > 0 else 1.0
        width_px     = float(frow.get("width", 1.0) or 1.0)

        # TACS classification — uses real straightness from CT-FIRE
        tacs = None
        if has_boundary and angle_to_tangent is not None and nb_dist is not None:
            tacs = classify_fiber_tacs(
                angle_to_tangent=angle_to_tangent,
                straightness=straightness,
                distance_to_boundary=nb_dist,
                tacs_zone_width=distance_threshold,
            )

        in_flag = None
        if result.in_curvs_flag is not None and i < len(result.in_curvs_flag):
            in_flag = bool(result.in_curvs_flag[i])

        fobj = FiberObject(
            object_id=f"{tag}_ctfire_{i:04d}",
            centerline=np.array([[frow["center_row"], frow["center_col"]]]),
            angle=float(frow["angle"]),
            length=total_len * pixel_size,
            width=width_px * pixel_size,
            straightness=straightness,
            nearest_boundary_distance=nb_dist,
            nearest_boundary_point=bdry_pt,
            relative_angle_to_boundary_tangent=angle_to_tangent,
            in_tumor_boundary=in_epictr,
            tacs_type=tacs,
            extraction_mode="ctfire",
            metadata={
                "source":          "curvealign_ctfire_mode_pipeline",
                "end_length":      end_len * pixel_size,
                "in_curvs_flag":   in_flag,
            },
        )
        fiber_objects.append(fobj)

    print(f"  FiberObject nodes built : {len(fiber_objects)}")

    # ── 2. Assign fibers to ROI contours ──────────────────────────────────────
    fiber_roi_map: dict[str, object] = {}
    if coordinates:
        fiber_roi_map = _assign_fibers_to_rois(fiber_objects, coordinates)
        for fobj in fiber_objects:
            fobj.metadata["roi_key"] = fiber_roi_map.get(fobj.object_id)

    # ── 3. Build TMEHierarchy ─────────────────────────────────────────────────
    hierarchy = TMEHierarchy()

    img_entry = ImageEntry(
        object_id=f"{tag}_image",
        path=str(REAL_IMAGE_PATH),
        modality="SHG",
        pixel_size=(pixel_size, pixel_size),
        metadata={"shape": list(img.shape)},
    )
    hierarchy.add_object(img_entry)

    # Create ROIObject nodes for each boundary contour
    roi_nodes: dict[object, ROIObject] = {}
    if coordinates:
        for roi_key, coords_arr in coordinates.items():
            roi_node = ROIObject(
                object_id=f"{tag}_roi_{roi_key}",
                label=f"ROI {roi_key}",
                shape_type=GeometryType.POLYGON,
                geometry=Geometry(
                    type=GeometryType.POLYGON,
                    coordinates=np.asarray(coords_arr),
                ),
                annotation_type="tumor_boundary",
            )
            hierarchy.add_object(roi_node, parent=img_entry)
            roi_nodes[roi_key] = roi_node

    # Add FiberObjects under their respective ROI node (or ImageEntry if no ROI)
    for fobj in fiber_objects:
        rk = fobj.metadata.get("roi_key")
        parent_node = roi_nodes.get(rk, img_entry) if rk is not None else img_entry
        hierarchy.add_object(fobj, parent=parent_node)

    n_fiber_nodes = len(hierarchy.get_objects_by_type(TMEType.FIBER))
    n_roi_nodes   = len(hierarchy.get_objects_by_type(TMEType.ANNOTATION))
    print(
        f"  Hierarchy nodes total   : {1 + n_roi_nodes + n_fiber_nodes}"
        f"  (1 ImageEntry + {n_roi_nodes} ROIObjects + {n_fiber_nodes} FiberObjects)"
    )

    # ── 4. TACS summary ───────────────────────────────────────────────────────
    if has_boundary:
        tacs_counts: dict[str | None, int] = {"TACS-1": 0, "TACS-2": 0, "TACS-3": 0, None: 0}
        for f in fiber_objects:
            tacs_counts[f.tacs_type] = tacs_counts.get(f.tacs_type, 0) + 1

        print(f"\n  TACS classification (zone <= {distance_threshold:.0f} px):")
        for label in ("TACS-1", "TACS-2", "TACS-3", None):
            n   = tacs_counts.get(label, 0)
            pct = 100.0 * n / len(fiber_objects) if fiber_objects else 0.0
            name = label if label is not None else "unclassified (outside zone)"
            print(f"    {name:<30}: {n:4d}  ({pct:.1f}%)")
    else:
        print("  No boundary — TACS classification skipped.")

    # ── 5. Hierarchy queries ──────────────────────────────────────────────────
    print("\n  Hierarchy queries:")

    all_fibers = hierarchy.get_objects_by_type(TMEType.FIBER)
    print(f"    get_objects_by_type('FIBER')             -> {len(all_fibers)} nodes")

    sample_id = f"{tag}_ctfire_0000"
    sample = hierarchy.get_object(sample_id)
    if sample is not None:
        print(f"    get_object('{sample_id}') -> angle={sample.angle:.1f} deg, TACS={sample.tacs_type}")

    tacs3 = [f for f in all_fibers if f.tacs_type == "TACS-3"]
    print(f"    TACS-3 (invasive) fibers                 -> {len(tacs3)}")

    with_dist = [f for f in all_fibers if f.nearest_boundary_distance is not None]
    if with_dist:
        nearest = min(with_dist, key=lambda f: f.nearest_boundary_distance)
        print(
            f"    Nearest-to-boundary fiber                -> {nearest.object_id}"
            f"  dist={nearest.nearest_boundary_distance:.1f} px  TACS={nearest.tacs_type}"
        )

    if all_fibers:
        first = all_fibers[0]
        ancestors = first.get_ancestors()
        print(f"    {first.object_id}.get_ancestors() -> {[a.object_id for a in ancestors]}")

    print()
    return hierarchy, fiber_roi_map


# ── Interactive TACS viewer ────────────────────────────────────────────────────

def _launch_tacs_viewer(
    tag: str,
    img: np.ndarray,
    fiber_objects: list[FiberObject],
    coordinates: dict | None,
    line_length: float = 5.0,
    fiber_roi_map: dict | None = None,
) -> None:
    """Interactive TACS overlay viewer with ROI filter, TACS filter, and fiber table.

    Controls
    --------
    ROI radio buttons (bottom left):
      All ROIs | ROI 0 | ROI 1 | …  — filter fibers by boundary contour.
    TACS radio buttons (bottom centre):
      All fibers | TACS-3 only | TACS-2 only | TACS-1 only | Outside zone
    Check button (bottom right):
      Show associations — draws a dashed line from each fiber center to its
      nearest boundary point (colored by TACS type).
    Fiber table (right panel):
      Lists measurements for currently visible fibers.  Click a row to
      highlight the fiber in the image.  Click a fiber dot to scroll the
      table to and highlight the corresponding row.  Scroll wheel scrolls
      the table.

    Color coding
    ------------
    TACS-3 — red        (perpendicular / invasive)
    TACS-2 — lime       (parallel)
    TACS-1 — dodgerblue (random / intermediate)
    Outside zone — lightgray
    """
    TACS_STYLE: dict[str | None, dict] = {
        "TACS-3": {"color": "red",         "label": "TACS-3  (perpendicular / invasive)"},
        "TACS-2": {"color": "lime",         "label": "TACS-2  (parallel)"},
        "TACS-1": {"color": "dodgerblue",   "label": "TACS-1  (random / intermediate)"},
        None:     {"color": "lightgray",    "label": "Outside zone / no boundary"},
    }

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    ax       = fig.add_axes([0.02, 0.12, 0.54, 0.83])   # image (left ~56%)
    ax_table = fig.add_axes([0.59, 0.12, 0.40, 0.83])   # fiber table (right ~40%)

    roi_keys_sorted: list = sorted(coordinates.keys()) if coordinates else []
    roi_labels: list[str] = ["All ROIs"] + [f"ROI {k}" for k in roi_keys_sorted]
    n_roi_labels = len(roi_labels)

    ax_roi_radio  = fig.add_axes([0.02, 0.01, 0.16, 0.09])
    ax_tacs_radio = fig.add_axes([0.20, 0.01, 0.22, 0.09])
    ax_check      = fig.add_axes([0.44, 0.01, 0.13, 0.09])

    # ── Image axes ────────────────────────────────────────────────────────────
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title(f"{tag} — TACS viewer", fontsize=9)
    ax.axis("off")

    # Draw boundary contours (color-coded by ROI key); store artists for highlight
    _roi_colors: list[str] = ["yellow", "cyan", "orange", "magenta", "lime"]
    _roi_contour_artists: dict[object, object] = {}
    if coordinates:
        for idx, (roi_key, roi_coords) in enumerate(coordinates.items()):
            c = _roi_colors[idx % len(_roi_colors)]
            roi_line, = ax.plot(roi_coords[:, 1], roi_coords[:, 0], "-",
                                lw=0.8, alpha=0.7, color=c)
            _roi_contour_artists[roi_key] = roi_line

    # ── Table axes ────────────────────────────────────────────────────────────
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    TABLE_MAX_ROWS = 18
    TABLE_COLS     = ["#", "(r,c)", "Ang°", "Rel∠", "Len", "Wid", "Str", "TACS", "ROI", "Dist"]
    COL_X          = [0.01, 0.07, 0.21, 0.30, 0.39, 0.45, 0.52, 0.59, 0.70, 0.83]
    ROW_H          = 0.045

    # ── Visibility state ──────────────────────────────────────────────────────
    _ALL_ON  = {k: True  for k in TACS_STYLE}
    _ALL_OFF = {k: False for k in TACS_STYLE}
    VISIBILITY: dict[str, dict] = {
        "All fibers":   _ALL_ON,
        "TACS-3 only":  {**_ALL_OFF, "TACS-3": True},
        "TACS-2 only":  {**_ALL_OFF, "TACS-2": True},
        "TACS-1 only":  {**_ALL_OFF, "TACS-1": True},
        "Outside zone": {**_ALL_OFF, None: True},
    }
    _current_vis: dict[str | None, bool] = dict(_ALL_ON)
    _roi_state: list[object] = ["ALL"]   # "ALL" or a coordinates key (list for mutation)

    # Fast artist lookups (built during pre-draw loop)
    _dot_to_id:     dict[int, str]    = {}   # id(dot)  → fobj.object_id
    _id_to_artists: dict[str, tuple]  = {}   # fobj.object_id → (line, dot) or (None, None)
    _id_to_assoc:   dict[str, object] = {}   # fobj.object_id → assoc Line2D | None

    # Table selection state
    _selected_id:   list[str | None] = [None]
    _table_offset:  list[int]        = [0]

    # ── Table value formatter ─────────────────────────────────────────────────
    def _fmt(v: object, fmt: str = ".1f") -> str:
        if v is None:
            return "—"
        try:
            if pd.isna(v):  # type: ignore[arg-type]
                return "—"
        except Exception:
            pass
        try:
            return format(float(v), fmt)
        except Exception:
            return str(v)

    # ── Pre-draw fiber artists ─────────────────────────────────────────────────
    groups: dict[str | None, list] = {k: [] for k in TACS_STYLE}   # for legend counts

    for fobj in fiber_objects:
        key   = fobj.tacs_type if fobj.tacs_type in TACS_STYLE else None
        color = TACS_STYLE[key]["color"]
        _id_to_assoc[fobj.object_id] = None

        if fobj.centerline is None or len(fobj.centerline) == 0:
            _id_to_artists[fobj.object_id] = (None, None)
            continue

        row_f     = float(fobj.centerline[0, 0])
        col_f     = float(fobj.centerline[0, 1])
        angle_rad = np.deg2rad(fobj.angle)
        dx        = line_length * np.cos(angle_rad)
        dy        = line_length * np.sin(angle_rad)

        line, = ax.plot(
            [col_f - dx, col_f + dx], [row_f + dy, row_f - dy],
            color=color, lw=0.8, alpha=0.85,
        )
        dot, = ax.plot(col_f, row_f, ".", color=color, ms=2.5, picker=6)

        groups[key].extend([line, dot])
        _dot_to_id[id(dot)]              = fobj.object_id
        _id_to_artists[fobj.object_id]   = (line, dot)

        # Association line: fiber center → nearest boundary point
        if (fobj.nearest_boundary_point is not None
                and not np.any(np.isnan(fobj.nearest_boundary_point))):
            bdry_row_f = float(fobj.nearest_boundary_point[0])
            bdry_col_f = float(fobj.nearest_boundary_point[1])
            aline, = ax.plot(
                [col_f, bdry_col_f], [row_f, bdry_row_f],
                color=color, lw=0.5, alpha=0.6, ls="--", visible=False,
            )
            _id_to_assoc[fobj.object_id] = aline

    # Highlight overlay — single artist repositioned on selection; sized to ring the dot
    _highlight_dot, = ax.plot([], [], "o", ms=9, mfc="none",
                              mec="white", mew=1.5, zorder=10)

    # Legend
    counts = {k: len(v) // 2 for k, v in groups.items()}
    legend_handles = [
        Line2D([0], [0], color=s["color"], lw=2,
               label=f"{s['label']}  (n={counts[k]})")
        for k, s in TACS_STYLE.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.7)

    # ── Helper functions ──────────────────────────────────────────────────────

    def _visible_fiber_ids() -> list[str]:
        """Return object_ids passing both the ROI filter and the TACS filter."""
        ids = []
        for fobj in fiber_objects:
            key = fobj.tacs_type if fobj.tacs_type in TACS_STYLE else None
            if not _current_vis.get(key, False):
                continue
            if _roi_state[0] != "ALL":
                rk = (fiber_roi_map or {}).get(fobj.object_id)
                if rk != _roi_state[0]:
                    continue
            ids.append(fobj.object_id)
        return ids

    def _move_highlight(fid: str | None) -> None:
        if fid is None:
            _highlight_dot.set_data([], [])
            return
        fobj = next((f for f in fiber_objects if f.object_id == fid), None)
        if fobj is None or fobj.centerline is None or len(fobj.centerline) == 0:
            _highlight_dot.set_data([], [])
            return
        # centerline stores (row, col); matplotlib plot takes (x=col, y=row)
        row_h = float(fobj.centerline[0, 0])
        col_h = float(fobj.centerline[0, 1])
        _highlight_dot.set_data([col_h], [row_h])

    def _rebuild_table() -> None:
        ax_table.cla()
        ax_table.set_xlim(0, 1)
        ax_table.set_ylim(0, 1)
        ax_table.axis("off")

        visible_ids = _visible_fiber_ids()
        n = len(visible_ids)
        offset = min(_table_offset[0], max(0, n - TABLE_MAX_ROWS))
        _table_offset[0] = offset

        # Title
        ax_table.text(
            0.5, 0.988, f"Fiber table  [{n} visible]",
            ha="center", va="top", fontsize=8, fontweight="bold",
            transform=ax_table.transAxes,
        )
        # Column headers
        for ci, col_label in enumerate(TABLE_COLS):
            ax_table.text(
                COL_X[ci], 0.958, col_label,
                ha="left", va="top", fontsize=7, fontweight="bold",
                transform=ax_table.transAxes, color="#333333",
            )
        ax_table.plot([0, 1], [0.948, 0.948], color="#cccccc", lw=0.5,
                      transform=ax_table.transAxes)

        # Data rows
        _id_lookup = {fobj.object_id: fobj for fobj in fiber_objects}
        for ri in range(TABLE_MAX_ROWS):
            fi = offset + ri
            if fi >= n:
                break
            fid  = visible_ids[fi]
            fobj = _id_lookup.get(fid)
            if fobj is None:
                continue

            y_top = 0.935 - ri * ROW_H

            is_sel   = (fid == _selected_id[0])
            bg_color = "#cce5ff" if is_sel else ("#f5f5f5" if ri % 2 else "white")

            rect = Rectangle(
                (0.0, y_top - ROW_H * 0.95),
                1.0, ROW_H * 0.95,
                facecolor=bg_color, edgecolor="none",
                transform=ax_table.transAxes, clip_on=True,
            )
            ax_table.add_patch(rect)

            # Center coordinates from centerline (row, col)
            if fobj.centerline is not None and len(fobj.centerline) > 0:
                rc_str = f"{int(round(fobj.centerline[0, 0]))},{int(round(fobj.centerline[0, 1]))}"
            else:
                rc_str = "—"
            # ROI label from fiber_roi_map
            rk_f = (fiber_roi_map or {}).get(fobj.object_id)
            roi_str = f"ROI {rk_f}" if rk_f is not None else "—"
            # CT-FIRE fibers always have real morphology (extraction_mode="ctfire")
            is_curvelet = (fobj.extraction_mode == "curvelets")
            vals = [
                str(fi + 1),
                rc_str,
                _fmt(fobj.angle),
                _fmt(fobj.relative_angle_to_boundary_tangent),
                "—" if is_curvelet else _fmt(fobj.length),
                "—" if is_curvelet else _fmt(fobj.width, ".2f"),
                "—" if is_curvelet else _fmt(fobj.straightness, ".2f"),
                fobj.tacs_type or "—",
                roi_str,
                _fmt(fobj.nearest_boundary_distance),
            ]
            tacs_color = TACS_STYLE.get(fobj.tacs_type, TACS_STYLE[None])["color"]
            for ci, val in enumerate(vals):
                text_color = tacs_color if ci == 7 else "black"
                ax_table.text(
                    COL_X[ci], y_top - ROW_H * 0.05,
                    val, ha="left", va="top", fontsize=6.5,
                    transform=ax_table.transAxes, color=text_color,
                )

        # Scroll hint
        if n > TABLE_MAX_ROWS:
            ax_table.text(
                0.5, 0.012,
                f"↑↓ scroll  rows {offset + 1}–{min(offset + TABLE_MAX_ROWS, n)} / {n}",
                ha="center", va="bottom", fontsize=6.5, color="#888888",
                transform=ax_table.transAxes,
            )
        ax_table.figure.canvas.draw_idle()

    def _update_assoc_visibility() -> None:
        show = check.get_status()[0]
        visible_set = set(_visible_fiber_ids()) if show else set()
        for fobj in fiber_objects:
            aline = _id_to_assoc.get(fobj.object_id)
            if aline is not None:
                aline.set_visible(fobj.object_id in visible_set)

    def _apply_visibility() -> None:
        visible_set = set(_visible_fiber_ids())
        for fobj in fiber_objects:
            line, dot = _id_to_artists.get(fobj.object_id, (None, None))
            if line is not None:
                vis = fobj.object_id in visible_set
                line.set_visible(vis)
                dot.set_visible(vis)
        _update_assoc_visibility()
        _rebuild_table()
        fig.canvas.draw_idle()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_select(label: str) -> None:
        _current_vis.update(VISIBILITY[label])
        ax.set_title(f"{tag} — TACS viewer  [{label}]", fontsize=9)
        _apply_visibility()

    def _on_roi_select(label: str) -> None:
        if label == "All ROIs":
            _roi_state[0] = "ALL"
        else:
            idx = roi_labels.index(label) - 1
            _roi_state[0] = roi_keys_sorted[idx]
        # Highlight the selected ROI contour; dim the others
        for rk, rline in _roi_contour_artists.items():
            if _roi_state[0] == "ALL":
                rline.set_linewidth(0.8)
                rline.set_alpha(0.7)
            elif rk == _roi_state[0]:
                rline.set_linewidth(2.5)
                rline.set_alpha(1.0)
            else:
                rline.set_linewidth(0.4)
                rline.set_alpha(0.25)
        _apply_visibility()

    def _on_check(_: str) -> None:
        _update_assoc_visibility()
        fig.canvas.draw_idle()

    def _on_table_click(event) -> None:
        if event.inaxes is not ax_table or event.ydata is None:
            return
        row_idx = int((0.935 - event.ydata) / ROW_H)
        if row_idx < 0:
            return
        visible_ids = _visible_fiber_ids()
        fi = _table_offset[0] + row_idx
        if 0 <= fi < len(visible_ids):
            _selected_id[0] = visible_ids[fi]
            _move_highlight(_selected_id[0])
            _rebuild_table()

    def _on_scroll(event) -> None:
        if event.inaxes is not ax_table:
            return
        delta = -1 if event.button == "up" else 1
        n = len(_visible_fiber_ids())
        _table_offset[0] = max(0, min(_table_offset[0] + delta,
                                       max(0, n - TABLE_MAX_ROWS)))
        _rebuild_table()

    def _on_pick(event) -> None:
        fid = _dot_to_id.get(id(event.artist))
        if fid is None:
            return
        _selected_id[0] = fid
        _move_highlight(fid)
        visible_ids = _visible_fiber_ids()
        if fid in visible_ids:
            fi = visible_ids.index(fid)
            if not (_table_offset[0] <= fi < _table_offset[0] + TABLE_MAX_ROWS):
                _table_offset[0] = max(0, fi - TABLE_MAX_ROWS // 2)
        _rebuild_table()

    # ── Widgets ───────────────────────────────────────────────────────────────

    # ROI selector (hidden if only one or no ROI)
    if n_roi_labels > 1:
        radio_roi = RadioButtons(ax_roi_radio, roi_labels, active=0)
        radio_roi.on_clicked(_on_roi_select)
    else:
        ax_roi_radio.axis("off")

    # TACS selector
    tacs_labels = ("All fibers", "TACS-3 only", "TACS-2 only", "TACS-1 only", "Outside zone")
    radio = RadioButtons(ax_tacs_radio, labels=tacs_labels, active=0)
    radio.on_clicked(_on_select)

    # Associations toggle
    check = CheckButtons(ax_check, labels=["Show associations"], actives=[False])
    check.on_clicked(_on_check)

    # ── Event connections ─────────────────────────────────────────────────────
    fig.canvas.mpl_connect("button_press_event", _on_table_click)
    fig.canvas.mpl_connect("scroll_event",        _on_scroll)
    fig.canvas.mpl_connect("pick_event",          _on_pick)

    # Initial table render
    _rebuild_table()

    plt.show(block=True)


# ── Scenario: real SHG image + boundary mask ──────────────────────────────────

def scenario_real_image() -> None:
    print("=" * 60)
    print("CT-FIRE pipeline — real SHG image + boundary mask")
    print("=" * 60)

    if not REAL_IMAGE_PATH.exists():
        print(f"  SKIPPED — image not found: {REAL_IMAGE_PATH}")
        return
    if not REAL_MASK_PATH.exists():
        print(f"  SKIPPED — mask not found: {REAL_MASK_PATH}")
        return

    img  = _load_grayscale(REAL_IMAGE_PATH)
    mask = _load_binary_mask(REAL_MASK_PATH)
    print(f"  Image shape : {img.shape}   Mask shape : {mask.shape}")

    # Normalize to [0, 1]
    if img.max() > 0:
        img = img / img.max()

    # ── CT-FIRE parameters ────────────────────────────────────────────────────
    # Adjust these to tune fiber extraction for the image.
    # coefficient_percentile : fraction of curvelet coefficients kept (top N%)
    #   lower = fewer, stronger responses; higher = more responses (more seeds)
    # thresh_LMPdist : FIRE seed suppression radius (px); raise to reduce seed count
    # thresh_flen    : minimum fiber arc length (px) to keep after tracing
    ctfire_params = {
        "coefficient_percentile": 0.2,
        "num_scales":             3,
        "fiber_threshold":        0.5,
        "widMAX":                 20,
        # LL1: post-tracing minimum fiber arc-length (px); fibers shorter than this
        # are removed by _build_fiber_dataframe after FIRE extraction.
        # Default in ct_fire.py is 30; set to 0 to keep all traced fibers.
        "LL1":                    30,
        "widcon": {
            "wid_mm":    1,
            "wid_mp":   10,
            "wid_sigma": 1,
            "wid_max":   0,
            "wid_opt":   1,
        },
        "value": {
            "sigma_im":            0,
            "sigma_d":             0.3,
            "dtype":               "cityblock",
            "thresh_im":           [],
            "thresh_im2":          5,
            "thresh_Dxlink":       1.5,
            "s_xlinkbox":          8,
            "thresh_LMP":          0.2,
            "thresh_LMPdist":      2,
            "thresh_ext":          0.342,
            "lam_dirdecay":        0.5,
            "s_minstep":           2,
            "s_maxstep":           6,
            "thresh_dang_aextend": 0.9848,
            "thresh_dang_L":       15,
            "thresh_short_L":      15,
            "s_fiberdir":          4,
            "thresh_linkd":        15,
            "thresh_linka":        -0.866,
            "thresh_flen":         15,
            "thresh_numv":         3,
            "scale":               [1.0, 1.0, 1.0],
            "s_boundthick":        10,
            "blist":               1,
            "s_maxspace":          5,
            "lambda":              0.01,
            "ang_interval":        3,
        },
    }

    result = curvealign_ctfire_mode_pipeline(
        image=img,
        boundary_img=mask,
        fiber_mode=1,
        tif_boundary=3,
        distance_threshold=100.0,
        ctfire_params=ctfire_params,
    )

    if result is None:
        print("  No fibers detected — pipeline returned None.")
        return

    fs = result.fiber_structure
    print(f"  Fibers detected       : {len(fs)}")
    print(f"  boundary_measurement  : {result.boundary_measurement}")
    print(f"  fiber_structure cols  : {list(fs.columns)}")

    if result.roi_summary_df is not None and not result.roi_summary_df.empty:
        print(f"\n  ROI summary ({len(result.roi_summary_df)} ROIs):")
        print(result.roi_summary_df.to_string(index=False))

    # Extract boundary contour coordinates for hierarchy + viewer
    coordinates = extract_boundary_coords_from_mask(mask)
    print(f"\n  Boundary contours extracted: {list(coordinates.keys())}")

    # Save overlay, heatmap, and xlsx to output/
    _save_figures_and_xlsx("ctfire_real", img, result, coordinates=coordinates)

    # Build TMEHierarchy + FiberObjects with TACS classification
    hier_result = _tacs_hierarchy_integration_ctfire(
        tag="ctfire_real",
        img=img,
        result=result,
        distance_threshold=100.0,
        pixel_size=1.0,
        coordinates=coordinates,
    )

    # Launch interactive TACS viewer
    if hier_result is not None:
        hierarchy, fiber_roi_map = hier_result
        fiber_objects = hierarchy.get_objects_by_type(TMEType.FIBER)
        _launch_tacs_viewer(
            tag="ctfire_real",
            img=img,
            fiber_objects=fiber_objects,
            coordinates=coordinates,
            line_length=5.0,
            fiber_roi_map=fiber_roi_map,
        )


# ── Scenario: FIRE-only (no curvelet reconstruction) ─────────────────────────

def scenario_fire_only() -> None:
    """Run the FIRE-only pipeline (``use_ct_reconstruction=False``).

    FIRE-only mode: no curvelops / curvelet library required.
    Only ctfire_py (and its C++ fiber_backend extension) must be installed.

    ``fire_2d_angle`` operates directly on the normalised image.  Without the
    curvelet enhancement step, the FIRE seed-detection thresholds become the
    primary lever for noise control.  Key parameters to tune:

    * ``thresh_im2``   — absolute background threshold (out of 255); raise to
                         suppress dim background pixels before seed detection.
    * ``thresh_LMPdist`` — FIRE seed suppression radius (px); raise to reduce
                           seed count on noisy raw images.
    * ``thresh_flen``  — minimum fiber arc length (px) to keep after tracing.
    """
    print("=" * 60)
    print("CT-FIRE pipeline — FIRE only (no curvelet reconstruction)")
    print("=" * 60)

    if not REAL_IMAGE_PATH.exists():
        print(f"  SKIPPED — image not found: {REAL_IMAGE_PATH}")
        return
    if not REAL_MASK_PATH.exists():
        print(f"  SKIPPED — mask not found: {REAL_MASK_PATH}")
        return

    img  = _load_grayscale(REAL_IMAGE_PATH)
    mask = _load_binary_mask(REAL_MASK_PATH)
    print(f"  Image shape : {img.shape}   Mask shape : {mask.shape}")

    # Normalize to [0, 1] — the pipeline normalises internally to [0, 255].
    if img.max() > 0:
        img = img / img.max()

    # ── FIRE-only parameters ──────────────────────────────────────────────────
    # coefficient_percentile / num_scales are curvelet parameters — they are
    # ignored when use_ct_reconstruction=False but kept here for documentation.
    #
    # In CT mode the curvelet reconstruction produces a sparse fiber-only image
    # before FIRE runs, so seed density is naturally low.  In FIRE-only mode FIRE
    # sees the raw normalised image; the two thresholds below are the main levers
    # to keep seed count low enough to avoid a std::bad_alloc MemoryError:
    #
    #   thresh_im2    — absolute background threshold (out of 255 after
    #                   normalisation).  Pixels below this are zeroed before
    #                   the distance transform.  Start at 60 and tune down if
    #                   too few fibers are detected.
    #   thresh_LMPdist — FIRE seed suppression radius (px).  Any two local-max
    #                   candidates within this radius collapse to one seed.
    #                   Raising it is the most effective memory guard: it reduces
    #                   seed count regardless of how many foreground pixels pass
    #                   thresh_im2.  Use 10–15 on full-resolution SHG images.
    #
    # LL1: post-tracing minimum fiber arc-length filter (px).  Fibers shorter
    # than LL1 are discarded by _build_fiber_dataframe.  Raise on noisy images.
    fire_only_params = {
        # curvelet keys (not used in this mode — included for reference):
        "coefficient_percentile": 0.2,
        "num_scales":             3,
        "fiber_threshold":        0.5,
        # post-processing filters:
        "widMAX":                 20,
        # LL1: post-tracing minimum fiber arc-length (px); same key as CT path.
        "LL1":                    30,
        "widcon": {
            "wid_mm":    1,
            "wid_mp":   10,
            "wid_sigma": 1,
            "wid_max":   0,
            "wid_opt":   1,
        },
        # FIRE 2D parameters (forwarded directly to fire_2d_angle):
        "value": {
            "sigma_im":            0,      # image smoothing (0 = off)
            "sigma_d":             0.3,    # distance-transform smoothing
            "dtype":               "cityblock",
            "thresh_im":           [],     # relative intensity threshold ([] = off)
            "thresh_im2":          10,     # ← keep only brightest 76% of pixels;
                                           #   lower if too few fibers extracted
            "thresh_Dxlink":       1.5,
            "s_xlinkbox":          8,
            "thresh_LMP":          0.2,
            "thresh_LMPdist":      8,     # ← large suppression radius: primary
                                           #   guard against std::bad_alloc on raw
                                           #   images; lower only if too few seeds
            "thresh_ext":          0.342,
            "lam_dirdecay":        0.5,
            "s_minstep":           2,
            "s_maxstep":           6,
            "thresh_dang_aextend": 0.9848,
            "thresh_dang_L":       15,
            "thresh_short_L":      15,
            "s_fiberdir":          4,
            "thresh_linkd":        15,
            "thresh_linka":        -0.866,
            "thresh_flen":         20,     # ← slightly higher than CT mode (15)
            "thresh_numv":         3,
            "scale":               [1.0, 1.0, 1.0],
            "s_boundthick":        10,
            "blist":               1,
            "s_maxspace":          5,
            "lambda":              0.01,
            "ang_interval":        3,
        },
    }

    result = curvealign_ctfire_mode_pipeline(
        image=img,
        boundary_img=mask,
        ctfire_params=fire_only_params,
        use_ct_reconstruction=False,   # skip curvelet step — no curvelops needed
        fiber_mode=2,
        tif_boundary=3,
        distance_threshold=100.0,
    )

    if result is None:
        print("  No fibers detected — pipeline returned None.")
        return

    fs = result.fiber_structure
    print(f"  Fibers detected       : {len(fs)}")
    print(f"  boundary_measurement  : {result.boundary_measurement}")
    print(f"  use_ct_reconstruction : {result.params.get('use_ct_reconstruction')}")
    print(f"  fiber_structure cols  : {list(fs.columns)}")

    if result.roi_summary_df is not None and not result.roi_summary_df.empty:
        print(f"\n  ROI summary ({len(result.roi_summary_df)} ROIs):")
        print(result.roi_summary_df.to_string(index=False))

    coordinates = extract_boundary_coords_from_mask(mask)
    print(f"\n  Boundary contours extracted: {list(coordinates.keys())}")

    # Save overlay, heatmap, and xlsx to output/
    _save_figures_and_xlsx("ctfire_fire_only", img, result, coordinates=coordinates)

    # Build TMEHierarchy + FiberObjects with TACS classification
    hier_result = _tacs_hierarchy_integration_ctfire(
        tag="ctfire_fire_only",
        img=img,
        result=result,
        distance_threshold=100.0,
        pixel_size=1.0,
        coordinates=coordinates,
    )

    # Launch interactive TACS viewer
    if hier_result is not None:
        hierarchy, fiber_roi_map = hier_result
        fiber_objects = hierarchy.get_objects_by_type(TMEType.FIBER)
        _launch_tacs_viewer(
            tag="ctfire_fire_only",
            img=img,
            fiber_objects=fiber_objects,
            coordinates=coordinates,
            line_length=5.0,
            fiber_roi_map=fiber_roi_map,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CT-FIRE pipeline example — choose which scenario to run."
    )
    parser.add_argument(
        "--scenario",
        choices=["ctfire", "fire-only", "both"],
        default="both",
        help=(
            "ctfire    : full CT-FIRE (curvelet reconstruction + FIRE; requires curvelops)\n"
            "fire-only : FIRE only (no curvelet step; no curvelops needed)\n"
            "both      : run both in sequence (default)"
        ),
    )
    args = parser.parse_args()

    # Check ctfire_py availability before running
    try:
        from ctfire_py.ct_fire import ct_fire  # noqa: F401
    except ImportError as e:
        print("ERROR: ctfire_py is not available in this environment.")
        print(str(e))
        print(
            "\nInstallation steps (MSYS2 UCRT64 shell, .venv-curvelops active):\n"
            "  cd H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src/ctfire_py/CPP\n"
            "  make -f Makefile.ucrt64 && cp fiber_backend.* ../\n"
            "  echo 'H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src' \\\n"
            "    > .venv-curvelops/lib/python3.14/site-packages/ctfire_py_src.pth\n"
        )
        sys.exit(1)

    print(f"ctfire_py available — running scenario: {args.scenario}")

    if args.scenario in ("ctfire", "both"):
        scenario_real_image()
    if args.scenario in ("fire-only", "both"):
        scenario_fire_only()

    print(f"\nDone. Output figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
