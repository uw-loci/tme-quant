"""
TMEQuant Example — curvealign_curvelets_mode_pipeline
=====================================================

Demonstrates the ``curvealign_curvelets_mode_pipeline`` in three scenarios:

  1. Synthetic image — no boundary (minimal, no real data required).
  2. Synthetic image + pre-computed fiber orientations — no boundary.
  3. Real SHG image (real1.tif) + real boundary mask (mask_real1.tiff).

Scenario 3 additionally demonstrates:
  • TACS classification of curvelet orientation regions using pre-computed
    boundary angles from the pipeline (TACS-1 / TACS-2 / TACS-3).
  • TME hierarchy construction: ImageEntry → FiberObject nodes.
  • Representative hierarchy queries (type lookup, ID lookup, ancestry).
  • Interactive TACS viewer: matplotlib figure with RadioButtons to
    selectively display TACS-1 / TACS-2 / TACS-3 / all / outside-zone fibers.

Scenarios 1 and 2 save figures to disk only.
Scenario 3 saves figures, prints TACS + hierarchy summary, then opens
the interactive TACS viewer (closes on window close).

Output files are written to an ``output/`` folder next to this script.

Usage
-----
  python src/examples/example_curvealign_curvelets_mode_pipeline.py

Requirements
------------
  curvelops  (Scenarios 1 and 3 call build_fiber_structure_from_curvelets).
  Scenario 2 works without curvelops (fiber_structure is pre-supplied).
  Real image paths (Scenario 3):
    H:/GitHub.06.2022/tme-quant/tests/test_images/real1.tif
    H:/GitHub.06.2022/tme-quant/tests/test_images/CA_Boundary/mask_real1.tiff
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")   # interactive; change to "Agg" for headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, RadioButtons
import numpy as np
import pandas as pd
import tifffile

from tme_quant.tme_analysis.pipelines import curvealign_curvelets_mode_pipeline
from tme_quant.fiber_analysis.visualization.draw_utils import (
    generate_fiber_overlay,
    generate_fiber_heatmap,
)
from tme_quant.fiber_analysis.io import export_dataframe_to_excel
from tme_quant.fiber_analysis.utils.boundary_tif_utils import (
    extract_boundary_coords_from_mask,
)
from tme_quant import TMEHierarchy
from tme_quant.core.image_entry import ImageEntry
from tme_quant.core.base_models import TMEType
from tme_quant.core.tme_objects.fiber_objects import FiberObject
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_shg(h: int = 128, w: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w)).astype(np.float32)


def _fiber_dataframe(n: int = 10, h: int = 128, w: int = 128, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "center_row": rng.uniform(10, h - 10, n),
        "center_col": rng.uniform(10, w - 10, n),
        "angle":      rng.uniform(0, 180, n),
        "weight":     np.ones(n),
    })


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


def _print_result(label_str: str, result: dict | None) -> None:
    if result is None:
        print(f"  {label_str}: pipeline returned None (no fibers detected)\n")
        return
    fs  = result["fiber_structure"]
    ffd = result["fiber_features_df"]
    ia  = result["in_curvs_flag"]
    na  = result["nearest_angles"]
    print(f"  {label_str}:")
    print(f"    fiber_structure rows  : {len(fs)}")
    print(f"    fiber_features_df cols: {list(ffd.columns)}")
    print(f"    boundary_measurement  : {result['boundary_measurement']}")
    print(
        f"    in_curvs_flag         : "
        f"{ia.sum() if ia is not None else 'None'} / "
        f"{len(ia) if ia is not None else 'N/A'}"
    )
    print(f"    nearest_angles        : {'set' if na is not None else 'None'}")
    print()


def _save_figures_and_xlsx(
    tag: str,
    img: np.ndarray,
    result: dict | None,
    coordinates: dict | None = None,
    show: bool = False,
) -> None:
    """Save overlay + heatmap figures and xlsx; optionally display interactively."""
    if result is None:
        print(f"  [{tag}] No result — skipping figures and xlsx.\n")
        return

    fs                   = result["fiber_structure"]
    boundary_measurement = result["boundary_measurement"]
    in_curvs_flag        = result["in_curvs_flag"]
    nearest_angles       = result["nearest_angles"]
    fiber_features_df    = result["fiber_features_df"]
    tif_boundary         = 3 if boundary_measurement else 0

    angles = fs["angle"].values

    if in_curvs_flag is not None:
        out_curvs_flag = ~in_curvs_flag
    else:
        in_curvs_flag  = np.ones(len(fs), dtype=bool)
        out_curvs_flag = np.zeros(len(fs), dtype=bool)

    nearest_angles_arr = (
        np.asarray(nearest_angles) if nearest_angles is not None else angles
    )

    # Reconstruct measured_boundary with the column names expected by generate_fiber_overlay.
    # Note: boundary_point_row/col are stored in MATLAB (x,y) order (col→row label is swapped).
    measured_boundary = None
    if boundary_measurement and "boundary_point_row" in fiber_features_df.columns:
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
        fiber_mode=0,                       # curvelet mode → 4 px orientation lines
        tif_boundary=tif_boundary,
        boundary_measurement=boundary_measurement,
        make_associations=boundary_measurement,  # blue lines when boundary present
    )
    ax_ov.set_title(f"{tag} — curvelet overlay", fontsize=9)
    overlay_path = OUT_DIR / f"{tag}_overlay.png"
    fig_ov.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"  Overlay saved  -> {overlay_path}")
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig_ov)

    # ── Heatmap figure ────────────────────────────────────────────────────────
    distances = (
        fiber_features_df["nearest_distance_to_boundary"].values
        if "nearest_distance_to_boundary" in fiber_features_df.columns
        else None
    )
    fig_hm, rawmap, procmap = generate_fiber_heatmap(
        img=img,
        fiber_structure=fs,
        in_curvs_flag=in_curvs_flag,
        angles=nearest_angles_arr,
        distances=distances,
        tif_boundary=tif_boundary,
        boundary_measurement=boundary_measurement,
    )
    fig_hm.axes[0].set_title(f"{tag} — angle heatmap", fontsize=9)
    heatmap_path = OUT_DIR / f"{tag}_heatmap.png"
    fig_hm.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved  -> {heatmap_path}")
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig_hm)

    # ── xlsx export ───────────────────────────────────────────────────────────
    xlsx_path = str(OUT_DIR / f"{tag}_results.xlsx")
    export_dataframe_to_excel(fiber_features_df, xlsx_path, sheet_name="fiber_features")
    roi_summary = result.get("roi_summary_df")
    if roi_summary is not None and not roi_summary.empty:
        export_dataframe_to_excel(roi_summary, xlsx_path, sheet_name="roi_summary", mode="a")
    print(f"  Results saved  -> {xlsx_path}\n")


# ── TACS + hierarchy integration ──────────────────────────────────────────────

def _tacs_hierarchy_integration(
    tag: str,
    img: np.ndarray,
    result: dict | None,
    distance_threshold: float = 50.0,
    pixel_size: float = 1.0,
) -> TMEHierarchy | None:
    """Convert curvealign results into FiberObject nodes, classify TACS, build TMEHierarchy.

    Each row in ``fiber_structure`` becomes one ``FiberObject`` node whose
    ``tacs_type`` is set from the pre-computed boundary angle stored in
    ``fiber_features_df["nearest_relative_boundary_angle"]``.

    Angle convention note
    ---------------------
    ``nearest_relative_boundary_angle`` is stored in the pycurvelets complement
    convention (90° - angle_to_boundary_tangent).  It is converted here before
    passing to ``classify_fiber_tacs()``, which expects angle_to_tangent in [0, 90°].

    Parameters
    ----------
    distance_threshold :
        TACS zone width in pixels — must match the ``distance_threshold`` used
        in the pipeline call so fibers beyond the zone are correctly marked None.
    pixel_size :
        Micrometres per pixel; used for ``FiberObject.length`` / ``width`` units.
    """
    if result is None:
        print(f"  [{tag}] No result — skipping hierarchy integration.\n")
        return None

    print(f"\n{'─' * 60}")
    print(f"[{tag}] TACS + hierarchy integration")
    print(f"{'─' * 60}")

    fs          = result["fiber_structure"]
    ffd         = result["fiber_features_df"]
    has_boundary = result["boundary_measurement"]

    # ── 1. Build FiberObject nodes ────────────────────────────────────────────
    fiber_objects: list[FiberObject] = []
    for i, (_, frow) in enumerate(fs.iterrows()):
        feat = ffd.iloc[i] if i < len(ffd) else None

        nb_dist = None
        in_epictr = None
        angle_to_tangent = None

        bdry_pt: np.ndarray | None = None

        if feat is not None:
            raw_dist  = feat.get("nearest_distance_to_boundary")
            raw_epict = feat.get("inside_epicenter_region")
            raw_angle = feat.get("nearest_relative_boundary_angle")
            # boundary_point_row/col naming is swapped (MATLAB x,y convention):
            #   boundary_point_row stores the col/x coordinate of the boundary point
            #   boundary_point_col stores the row/y coordinate of the boundary point
            raw_bpt_r = feat.get("boundary_point_col")   # image row
            raw_bpt_c = feat.get("boundary_point_row")   # image col

            if raw_dist is not None and not pd.isna(raw_dist):
                nb_dist = float(raw_dist)
            if raw_epict is not None:
                in_epictr = bool(raw_epict)
            # nearest_relative_boundary_angle is in pycurvelets convention:
            #   value = arcsin(circ_r([2*fiber_deg, 2*tangent_deg]))
            #         = 90° − |fiber_angle − boundary_tangent_angle|
            #         = 90° − angle_to_boundary_tangent
            # classify_fiber_tacs() expects angle_to_boundary_tangent, so convert:
            if raw_angle is not None and not pd.isna(raw_angle):
                angle_to_tangent = 90.0 - float(raw_angle)
            if (raw_bpt_r is not None and not pd.isna(raw_bpt_r)
                    and raw_bpt_c is not None and not pd.isna(raw_bpt_c)):
                bdry_pt = np.array([float(raw_bpt_r), float(raw_bpt_c)])  # (row, col)

        # TACS classification using pre-computed boundary metrics
        tacs = None
        if has_boundary and angle_to_tangent is not None and nb_dist is not None:
            tacs = classify_fiber_tacs(
                angle_to_tangent=angle_to_tangent,
                straightness=1.0,           # curvelet regions have no straightness metric
                distance_to_boundary=nb_dist,
                tacs_zone_width=distance_threshold,
            )

        in_flag = None
        if result["in_curvs_flag"] is not None and i < len(result["in_curvs_flag"]):
            in_flag = bool(result["in_curvs_flag"][i])

        fobj = FiberObject(
            object_id=f"{tag}_curvelet_{i:04d}",
            centerline=np.array([[frow["center_row"], frow["center_col"]]]),
            angle=float(frow["angle"]),
            width=float(frow.get("weight", 1.0)) * pixel_size,
            length=pixel_size,              # orientation region, not a traced fiber
            straightness=1.0,
            nearest_boundary_distance=nb_dist,
            nearest_boundary_point=bdry_pt,
            relative_angle_to_boundary_tangent=angle_to_tangent,
            in_tumor_boundary=in_epictr,
            tacs_type=tacs,
            extraction_mode="curvelets",
            metadata={
                "source": "curvealign_curvelets_mode_pipeline",
                "in_curvs_flag": in_flag,
            },
        )
        fiber_objects.append(fobj)

    print(f"  FiberObject nodes built : {len(fiber_objects)}")

    # ── 2. Build TMEHierarchy ─────────────────────────────────────────────────
    hierarchy = TMEHierarchy()

    img_entry = ImageEntry(
        object_id=f"{tag}_image",
        path=str(REAL_IMAGE_PATH),
        modality="SHG",
        pixel_size=(pixel_size, pixel_size),
        metadata={"shape": list(img.shape)},
    )
    hierarchy.add_object(img_entry)

    for fobj in fiber_objects:
        hierarchy.add_object(fobj, parent=img_entry)

    n_fiber_nodes = len(hierarchy.get_objects_by_type(TMEType.FIBER))
    print(f"  Hierarchy nodes total   : {n_fiber_nodes + 1}  (1 ImageEntry + {n_fiber_nodes} FiberObjects)")

    # ── 3. TACS summary ───────────────────────────────────────────────────────
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

    # ── Angle diagnostic (verify convention) ─────────────────────────────────
    # nearest_relative_boundary_angle: high value (→90°) = fiber parallel to boundary tangent
    #                                  low  value (→ 0°) = fiber perpendicular (invasive)
    # angle_to_tangent = 90 - nearest_relative_boundary_angle
    #   high angle_to_tangent (60-90°) → TACS-3 (perpendicular)
    #   low  angle_to_tangent ( 0-30°) → TACS-2 (parallel)
    if has_boundary:
        print("\n  Angle diagnostic — 5 sample fibers in TACS zone:")
        print(f"    {'nearest_relative_boundary_angle':>35}  {'angle_to_tangent':>17}  TACS")
        n_shown = 0
        for i, (_, frow) in enumerate(fs.iterrows()):
            f = fiber_objects[i]
            if f.relative_angle_to_boundary_tangent is not None and f.nearest_boundary_distance is not None:
                raw = 90.0 - f.relative_angle_to_boundary_tangent   # recover raw stored value
                print(f"    {raw:>35.1f}  {f.relative_angle_to_boundary_tangent:>17.1f}  {f.tacs_type}")
                n_shown += 1
                if n_shown >= 5:
                    break

    # ── 4. Hierarchy queries ──────────────────────────────────────────────────
    print("\n  Hierarchy queries:")

    all_fibers = hierarchy.get_objects_by_type(TMEType.FIBER)
    print(f"    get_objects_by_type('FIBER')             -> {len(all_fibers)} nodes")

    sample = hierarchy.get_object(f"{tag}_curvelet_0000")
    print(f"    get_object('{tag}_curvelet_0000') -> angle={sample.angle:.1f} deg, TACS={sample.tacs_type}")

    tacs3 = [f for f in all_fibers if f.tacs_type == "TACS-3"]
    print(f"    TACS-3 (invasive) fibers                 -> {len(tacs3)}")

    with_dist = [f for f in all_fibers if f.nearest_boundary_distance is not None]
    if with_dist:
        nearest = min(with_dist, key=lambda f: f.nearest_boundary_distance)
        print(
            f"    Nearest-to-boundary fiber                -> {nearest.object_id}"
            f"  dist={nearest.nearest_boundary_distance:.1f} px  TACS={nearest.tacs_type}"
        )

    first = all_fibers[0]
    ancestors = first.get_ancestors()
    print(f"    {first.object_id}.get_ancestors() -> {[a.object_id for a in ancestors]}")

    print()
    return hierarchy


# ── Interactive TACS viewer ────────────────────────────────────────────────────

def _launch_tacs_viewer(
    tag: str,
    img: np.ndarray,
    fiber_objects: list[FiberObject],
    coordinates: dict | None,
    line_length: float = 5.0,
) -> None:
    """Interactive TACS overlay viewer with matplotlib RadioButtons.

    Pre-draws one group of matplotlib line/dot artists per TACS type, then
    toggles their visibility based on radio-button selection.  No redraw is
    needed on each click, keeping interaction fast even for large fiber counts.

    Controls
    --------
    Radio buttons (left panel):
      All fibers | TACS-3 only | TACS-2 only | TACS-1 only | Outside zone
    Check button:
      Show associations — draws a dashed line from each fiber center to
      its nearest boundary point (colored by TACS type).

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

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0.22)
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title(f"{tag} — TACS viewer  (use radio buttons to filter)", fontsize=9)
    ax.axis("off")

    # Draw boundary contours
    if coordinates:
        for roi_coords in coordinates.values():
            ax.plot(roi_coords[:, 1], roi_coords[:, 0], "y-", lw=0.8, alpha=0.6)

    # Pre-draw each TACS group; accumulate artists per type key
    groups: dict[str | None, list] = {k: [] for k in TACS_STYLE}
    # Association lines (fiber center → nearest boundary point), hidden by default
    assoc_lines: list = []

    for fobj in fiber_objects:
        key = fobj.tacs_type if fobj.tacs_type in groups else None
        color = TACS_STYLE[key]["color"]
        if fobj.centerline is not None and len(fobj.centerline) > 0:
            row = float(fobj.centerline[0, 0])
            col = float(fobj.centerline[0, 1])
            angle_rad = np.deg2rad(fobj.angle)
            dx = line_length * np.cos(angle_rad)
            dy = line_length * np.sin(angle_rad)
            line, = ax.plot(
                [col - dx, col + dx], [row - dy, row + dy],
                color=color, lw=0.8, alpha=0.85,
            )
            dot, = ax.plot(col, row, ".", color=color, ms=2.5)
            groups[key].extend([line, dot])

            # Association line: fiber center → nearest boundary point
            if (fobj.nearest_boundary_point is not None
                    and not np.any(np.isnan(fobj.nearest_boundary_point))):
                bdry_row = float(fobj.nearest_boundary_point[0])
                bdry_col = float(fobj.nearest_boundary_point[1])
                aline, = ax.plot(
                    [col, bdry_col], [row, bdry_row],
                    color=color, lw=0.5, alpha=0.6, ls="--", visible=False,
                )
                assoc_lines.append(aline)

    # Legend with per-type counts
    counts = {k: len(v) // 2 for k, v in groups.items()}
    legend_handles = [
        Line2D([0], [0], color=s["color"], lw=2,
               label=f"{s['label']}  (n={counts[k]})")
        for k, s in TACS_STYLE.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.7)

    # ── Radio buttons (TACS type filter) ─────────────────────────────────────
    ax_radio = plt.axes([0.01, 0.42, 0.19, 0.30])
    radio_labels = ("All fibers", "TACS-3 only", "TACS-2 only", "TACS-1 only", "Outside zone")
    radio = RadioButtons(ax_radio, labels=radio_labels, active=0)

    _ALL_ON  = {k: True  for k in TACS_STYLE}
    _ALL_OFF = {k: False for k in TACS_STYLE}
    VISIBILITY: dict[str, dict] = {
        "All fibers":   _ALL_ON,
        "TACS-3 only":  {**_ALL_OFF, "TACS-3": True},
        "TACS-2 only":  {**_ALL_OFF, "TACS-2": True},
        "TACS-1 only":  {**_ALL_OFF, "TACS-1": True},
        "Outside zone": {**_ALL_OFF, None: True},
    }

    # Track currently visible TACS keys so association-line toggle respects the filter
    _current_vis: dict[str | None, bool] = dict(_ALL_ON)

    def _on_select(label: str) -> None:
        _current_vis.update(VISIBILITY[label])
        for key, artists in groups.items():
            visible = _current_vis.get(key, False)
            for artist in artists:
                artist.set_visible(visible)
        # Keep association lines in sync with current filter if they are shown
        if check.get_status()[0]:
            _update_assoc_visibility()
        ax.set_title(f"{tag} — TACS viewer  [{label}]", fontsize=9)
        fig.canvas.draw_idle()

    radio.on_clicked(_on_select)

    # ── Check button (association lines toggle) ───────────────────────────────
    ax_check = plt.axes([0.01, 0.32, 0.19, 0.08])
    check = CheckButtons(ax_check, labels=["Show associations"], actives=[False])

    def _update_assoc_visibility() -> None:
        show = check.get_status()[0]
        for aline in assoc_lines:
            # Only show if the fiber's TACS group is currently visible
            aline.set_visible(show and _current_vis.get(aline.get_color(), True))
        # Simpler: just toggle all; color already matches the TACS type filter
        for aline in assoc_lines:
            aline.set_visible(show)

    def _on_check(_: str) -> None:
        _update_assoc_visibility()
        fig.canvas.draw_idle()

    check.on_clicked(_on_check)

    plt.show(block=True)


# ── Scenario 1: synthetic, no boundary, curvelet grouping via curvelops ───────

def scenario_1_no_boundary() -> None:
    print("=" * 60)
    print("Scenario 1 — Synthetic image, no boundary (requires curvelops)")
    print("=" * 60)
    try:
        import curvelops  # noqa: F401
    except ImportError:
        print("  curvelops not installed — skipping Scenario 1.\n")
        return

    img = _synthetic_shg()
    result = curvealign_curvelets_mode_pipeline(
        image=img, keep=0.05, scale=1, radius=4.0,
    )
    _print_result("no-boundary run", result)
    _save_figures_and_xlsx("scenario1_no_boundary", img, result, show=False)


# ── Scenario 2: synthetic, pre-computed fiber_structure, no boundary ──────────

def scenario_2_precomputed_fibers() -> None:
    print("=" * 60)
    print("Scenario 2 — Pre-computed fiber_structure, no boundary")
    print("=" * 60)
    img    = _synthetic_shg()
    fibers = _fiber_dataframe(n=12)

    result = curvealign_curvelets_mode_pipeline(image=img, fiber_structure=fibers)
    _print_result("pre-computed fibers", result)
    _save_figures_and_xlsx("scenario2_precomputed", img, result, show=False)


# ── Scenario 3: real SHG image + real boundary mask ──────────────────────────

def scenario_3_real_image() -> None:
    print("=" * 60)
    print("Scenario 3 — Real SHG image + real boundary mask (requires curvelops)")
    print("=" * 60)

    if not REAL_IMAGE_PATH.exists():
        print(f"  Image not found: {REAL_IMAGE_PATH} — skipping.\n")
        return
    if not REAL_MASK_PATH.exists():
        print(f"  Mask not found: {REAL_MASK_PATH} — skipping.\n")
        return
    try:
        import curvelops  # noqa: F401
    except ImportError:
        print("  curvelops not installed — skipping Scenario 3.\n")
        return

    img  = _load_grayscale(REAL_IMAGE_PATH)
    mask = _load_binary_mask(REAL_MASK_PATH)
    print(f"  Image shape: {img.shape}   Mask shape: {mask.shape}")

    result = curvealign_curvelets_mode_pipeline(
        image=img,
        boundary_img=mask,
        tif_boundary=3,
        distance_threshold=50.0,
        keep=0.01,
        scale=1,
        radius=4.0,
    )
    _print_result("real image + boundary", result)

    coordinates = extract_boundary_coords_from_mask(mask)
    print(f"  Boundary contours extracted: {list(coordinates.keys())}")

    # Save overlay + heatmap to disk (show=False; TACS viewer is the interactive display)
    _save_figures_and_xlsx(
        "scenario3_real_image", img, result, coordinates=coordinates, show=False
    )

    if result is not None and result["roi_summary_df"] is not None:
        print("  roi_summary_df:")
        print(result["roi_summary_df"].to_string(index=False))
        print()

    # ── TACS classification + TME hierarchy ───────────────────────────────────
    hierarchy = _tacs_hierarchy_integration(
        tag="scenario3_real_image",
        img=img,
        result=result,
        distance_threshold=50.0,   # must match pipeline call above
        pixel_size=1.0,
    )

    # ── Interactive TACS viewer ───────────────────────────────────────────────
    if hierarchy is not None:
        fiber_objects = hierarchy.get_objects_by_type(TMEType.FIBER)
        _launch_tacs_viewer(
            tag="scenario3_real_image",
            img=img,
            fiber_objects=fiber_objects,
            coordinates=coordinates,
            line_length=5.0,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # scenario_1_no_boundary()
    # scenario_2_precomputed_fibers()
    scenario_3_real_image()
    print(f"\nAll outputs written to: {OUT_DIR.resolve()}")
