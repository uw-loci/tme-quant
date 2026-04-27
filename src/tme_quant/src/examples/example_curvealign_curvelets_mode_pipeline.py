"""
TMEQuant Example — curvealign_curvelets_mode_pipeline
=====================================================

Demonstrates the ``curvealign_curvelets_mode_pipeline`` in three scenarios:

  1. Synthetic image — no boundary (minimal, no real data required).
  2. Synthetic image + pre-computed fiber orientations — no boundary.
  3. Real SHG image (real1.tif) + real boundary mask (mask_real1.tiff).

Scenarios 1 and 2 save figures to disk only.
Scenario 3 saves AND shows figures interactively.

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
    print(f"  Overlay saved  → {overlay_path}")
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
    print(f"  Heatmap saved  → {heatmap_path}")
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
    print(f"  Results saved  → {xlsx_path}\n")


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

    _save_figures_and_xlsx(
        "scenario3_real_image", img, result, coordinates=coordinates, show=True
    )

    if result is not None and result["roi_summary_df"] is not None:
        print("  roi_summary_df:")
        print(result["roi_summary_df"].to_string(index=False))
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # scenario_1_no_boundary()
    # scenario_2_precomputed_fibers()
    scenario_3_real_image()
    print(f"\nAll outputs written to: {OUT_DIR.resolve()}")
    plt.show()   # block until the user closes all Scenario-3 windows
