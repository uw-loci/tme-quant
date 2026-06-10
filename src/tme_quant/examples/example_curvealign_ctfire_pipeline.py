"""
TMEQuant Example — curvealign_ctfire_mode_pipeline
===================================================

Demonstrates the ``curvealign_ctfire_mode_pipeline`` in three scenarios:

  1. Synthetic stripe image — no boundary.
     Validates the pipeline runs and returns individual fiber segments with
     full CT-FIRE morphology columns (length, curvature, width).

  2. Real SHG image (real1.tif) + binary boundary mask (mask_real1.tiff).
     Full boundary analysis: per-ROI alignment angles + global boundary metrics.
     Skipped automatically when the image files are not present.

  3. Synthetic image + TMEHierarchy attachment.
     Builds an ``ImageEntry → TissueRegion → FiberObject`` tree from the
     CT-FIRE result, prints a hierarchy summary, and demonstrates queries.

Prerequisites
-------------
  - ``ctfire_py`` must be installed in the active Python environment.
    See the Prerequisites section of ``curvealign_ctfireMode_pipeline.py``
    for build and install instructions (MSYS2 UCRT64 / .venv-curvelops).

  - Scenario 2 also needs the real image files at the paths below.

Usage
-----
  python examples/example_curvealign_ctfire_pipeline.py

Output
------
  Figures saved to ``examples/output/``.  Console prints fiber counts,
  column names, and hierarchy summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless; change to "TkAgg" for interactive display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# ── Synthetic image helpers ───────────────────────────────────────────────────

def _synthetic_fiber_image(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """Create a synthetic image with diagonal stripe-like features.

    The stripes mimic SHG fiber patterns and give CT-FIRE something real to
    trace, unlike pure noise where it detects nothing.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)

    # Add diagonal lines at ~45° with Gaussian blur to simulate fibers
    for offset in range(-w, w, 20):
        for r in range(h):
            c = r + offset
            if 0 <= c < w:
                img[r, c] += 1.0

    # Add some noise
    img += rng.normal(0, 0.05, (h, w)).astype(np.float32)
    img = np.clip(img, 0, None)

    # Normalise to [0, 1]
    if img.max() > 0:
        img /= img.max()

    return img


# ── Fiber overlay plot ────────────────────────────────────────────────────────

def _plot_fiber_overlay(
    img: np.ndarray,
    fiber_structure: pd.DataFrame,
    title: str,
    save_path: Path,
    fiber_len: float = 10.0,
) -> None:
    """Draw fiber orientation lines over the image and save to disk."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="upper")

    for _, row in fiber_structure.iterrows():
        row_c = row["center_row"]
        col_c = row["center_col"]
        angle_rad = np.radians(row["angle"])
        dr = fiber_len * np.sin(angle_rad)
        dc = fiber_len * np.cos(angle_rad)
        ax.plot(
            [col_c - dc, col_c + dc],
            [row_c - dr, row_c + dr],
            color="lime", linewidth=0.8, alpha=0.7,
        )
        ax.plot(col_c, row_c, ".", color="red", markersize=2)

    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Hierarchy attachment helper ───────────────────────────────────────────────

def _attach_ctfire_fibers_to_hierarchy(
    tag: str,
    result,
    hierarchy,
    parent_node,
    pixel_size: float = 1.0,
) -> list:
    """Create FiberObject nodes from CTFirePipelineResult and attach to hierarchy.

    CT-FIRE provides individual traced fibers with full morphology.  Each row
    of ``fiber_structure`` becomes one FiberObject with:
      - ``centerline`` — single-point (row, col)
      - ``length``     — ``total_length`` (pixels × pixel_size)
      - ``straightness`` — ``curvature`` (end_length / total_length)
      - ``width``      — fiber width in µm

    Parameters
    ----------
    result : CTFirePipelineResult
    hierarchy : TMEHierarchy
    parent_node : TMEObject
        Node to attach fibers under (ImageEntry or TissueRegion).
    pixel_size : float
        µm per pixel; scales length and width to physical units.

    Returns
    -------
    list of FiberObject
    """
    from tme_quant.core.tme_objects.fiber_objects import FiberObject
    from tme_quant.fiber_analysis.tacs import classify_fiber_tacs

    fs  = result.fiber_structure
    ffd = result.fiber_features_df
    has_boundary = result.boundary_measurement

    fiber_objects = []
    for i, (_, frow) in enumerate(fs.iterrows()):
        feat = ffd.iloc[i] if i < len(ffd) else None

        nb_dist         = None
        angle_to_tangent = None
        bdry_pt         = None

        if feat is not None:
            raw_dist  = feat.get("nearest_distance_to_boundary")
            raw_angle = feat.get("nearest_relative_boundary_angle")
            raw_bpt_r = feat.get("boundary_point_col")   # row coord (naming swapped)
            raw_bpt_c = feat.get("boundary_point_row")   # col coord (naming swapped)

            if raw_dist is not None and not pd.isna(raw_dist):
                nb_dist = float(raw_dist)
            if raw_angle is not None and not pd.isna(raw_angle):
                angle_to_tangent = float(raw_angle)
            if (raw_bpt_r is not None and not pd.isna(raw_bpt_r)
                    and raw_bpt_c is not None and not pd.isna(raw_bpt_c)):
                bdry_pt = np.array([float(raw_bpt_r), float(raw_bpt_c)])

        total_len  = float(frow.get("total_length", 1.0) or 1.0)
        end_len    = float(frow.get("end_length",   total_len) or total_len)
        straightness = end_len / total_len if total_len > 0 else 1.0
        width_px   = float(frow.get("width", 1.0) or 1.0)

        tacs = None
        if has_boundary and angle_to_tangent is not None and nb_dist is not None:
            tacs = classify_fiber_tacs(
                angle_to_tangent=angle_to_tangent,
                straightness=straightness,
                distance_to_boundary=nb_dist,
                tacs_zone_width=50.0,
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
            tacs_type=tacs,
            extraction_mode="ctfire",
            metadata={
                "source":       "curvealign_ctfire_mode_pipeline",
                "end_length":   end_len * pixel_size,
                "curvature":    straightness,
                "in_curvs_flag": in_flag,
            },
        )
        parent_node.add_child(fobj)
        hierarchy._index.add(fobj)
        fiber_objects.append(fobj)

    return fiber_objects


# ── Scenario 1 — synthetic image, no boundary ─────────────────────────────────

def scenario_1() -> None:
    print("\n" + "=" * 60)
    print("Scenario 1 — Synthetic stripe image, no boundary")
    print("=" * 60)

    from tme_quant.tme_analysis.pipelines import curvealign_ctfire_mode_pipeline

    img = _synthetic_fiber_image(h=256, w=256, seed=42)
    print(f"  Image shape : {img.shape}")

    result = curvealign_ctfire_mode_pipeline(image=img)

    if result is None:
        print("  No fibers detected (CT-FIRE found nothing in synthetic image).")
        print("  This is normal — try with a real SHG image for meaningful results.")
        return

    fs = result.fiber_structure
    print(f"  Fibers detected   : {len(fs)}")
    print(f"  fiber_structure columns : {list(fs.columns)}")
    print(f"  fiber_features_df columns : {list(result.fiber_features_df.columns)}")
    print(f"  boundary_measurement : {result.boundary_measurement}")

    if not fs.empty:
        print(f"\n  Sample fibers (first 5):")
        print(
            fs[["center_row", "center_col", "angle", "total_length", "curvature", "width"]]
            .head()
            .to_string(index=False)
        )

        _plot_fiber_overlay(
            img, fs,
            title=f"CT-FIRE — synthetic image ({len(fs)} fibers)",
            save_path=OUT_DIR / "ctfire_scenario1_fibers.png",
        )


# ── Scenario 2 — real SHG image + boundary mask ───────────────────────────────

def scenario_2() -> None:
    print("\n" + "=" * 60)
    print("Scenario 2 — Real SHG image + boundary mask")
    print("=" * 60)

    if not REAL_IMAGE_PATH.exists():
        print(f"  SKIPPED — image not found: {REAL_IMAGE_PATH}")
        return
    if not REAL_MASK_PATH.exists():
        print(f"  SKIPPED — mask not found: {REAL_MASK_PATH}")
        return

    import tifffile
    from tme_quant.tme_analysis.pipelines import curvealign_ctfire_mode_pipeline

    img  = tifffile.imread(str(REAL_IMAGE_PATH)).astype(np.float32)
    mask = tifffile.imread(str(REAL_MASK_PATH))
    if img.ndim == 3:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    if img.max() > 0:
        img = img / img.max()

    print(f"  Image shape : {img.shape}")

    result = curvealign_ctfire_mode_pipeline(
        image=img,
        boundary_img=(mask > 0).astype(np.uint8),
        tif_boundary=3,
        distance_threshold=50.0,
    )

    if result is None:
        print("  No fibers detected.")
        return

    fs = result.fiber_structure
    print(f"  Fibers detected       : {len(fs)}")
    print(f"  boundary_measurement  : {result.boundary_measurement}")

    if result.roi_summary_df is not None and not result.roi_summary_df.empty:
        print(f"\n  ROI summary ({len(result.roi_summary_df)} ROIs):")
        print(result.roi_summary_df.to_string(index=False))

    if not fs.empty:
        _plot_fiber_overlay(
            img, fs,
            title=f"CT-FIRE — real SHG image ({len(fs)} fibers)",
            save_path=OUT_DIR / "ctfire_scenario2_fibers.png",
        )


# ── Scenario 3 — hierarchy attachment ────────────────────────────────────────

def scenario_3() -> None:
    print("\n" + "=" * 60)
    print("Scenario 3 — Synthetic image + TMEHierarchy attachment")
    print("=" * 60)

    from tme_quant.tme_analysis.pipelines import curvealign_ctfire_mode_pipeline
    from tme_quant import TMEHierarchy
    from tme_quant.core.image_entry import ImageEntry
    from tme_quant.core.base_models import TMEType
    from tme_quant.core.tme_objects.tissue_objects import TissueRegion

    img = _synthetic_fiber_image(h=256, w=256, seed=7)

    result = curvealign_ctfire_mode_pipeline(image=img)

    if result is None:
        print("  No fibers detected — hierarchy will be empty.")
        fiber_objects = []
    else:
        print(f"  Fibers detected : {len(result.fiber_structure)}")

    # Build hierarchy: ImageEntry → TissueRegion → FiberObject
    hierarchy = TMEHierarchy()

    img_entry = ImageEntry(
        object_id="scenario3_image",
        path="synthetic",
        modality="SHG",
        pixel_size=(1.0, 1.0),
        metadata={"shape": list(img.shape)},
    )
    hierarchy.add_object(img_entry)

    tissue = TissueRegion(
        object_id="scenario3_tissue",
        label="SHG tissue sample",
    )
    hierarchy.add_object(tissue, parent=img_entry)

    fiber_objects = []
    if result is not None:
        fiber_objects = _attach_ctfire_fibers_to_hierarchy(
            tag="s3",
            result=result,
            hierarchy=hierarchy,
            parent_node=tissue,
            pixel_size=1.0,
        )

    n_fibers = len(hierarchy.get_objects_by_type(TMEType.FIBER))
    print(f"\n  Hierarchy nodes : 1 ImageEntry + 1 TissueRegion + {n_fibers} FiberObjects")

    if fiber_objects:
        sample = fiber_objects[0]
        print(f"\n  Sample FiberObject:")
        print(f"    object_id    : {sample.object_id}")
        print(f"    angle        : {sample.angle:.1f}°")
        print(f"    length       : {sample.length:.1f} µm")
        print(f"    straightness : {sample.straightness:.3f}")
        print(f"    extraction_mode : {sample.extraction_mode}")
        print(f"    parent       : {sample.parent.object_id if sample.parent else None}")

        # Ancestry query
        ancestors = hierarchy.get_ancestors(sample)
        ancestor_ids = [a.object_id for a in ancestors]
        print(f"    ancestors    : {ancestor_ids}")

    print("\n  Hierarchy summary:")
    print(f"    Total objects : {len(list(hierarchy.all_objects()))}")
    by_type = {}
    for obj in hierarchy.all_objects():
        t = obj.tme_type.value if hasattr(obj.tme_type, "value") else str(obj.tme_type)
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"      {t:<20}: {n}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Check ctfire_py availability before running any scenario
    try:
        from ctfire_py.ct_fire import ct_fire  # noqa: F401
    except ImportError as e:
        print("ERROR: ctfire_py is not available in this environment.")
        print(str(e))
        print(
            "\nInstallation steps (MSYS2 UCRT64 shell, .venv-curvelops active):\n"
            "  cd H:/GitHub.06.2022/tmequant_ctfire/tme-quant/src/ctfire_py/CPP\n"
            "  make -f Makefile.ucrt64 && cp fiber_backend.*.so ../\n"
            "  pip install -e H:/GitHub.06.2022/tmequant_ctfire/tme-quant --no-deps\n"
        )
        sys.exit(1)

    print("ctfire_py available — running scenarios.")

    scenario_1()
    scenario_2()
    scenario_3()

    print("\nDone. Output figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
