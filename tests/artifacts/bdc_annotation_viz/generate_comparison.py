"""Generate comparison visualizations for BDcreationHE / HE2 annotation.

Usage:
    python tests/artifacts/bdc_annotation_viz/generate_comparison.py [output_dir]
    python tests/artifacts/bdc_annotation_viz/generate_comparison.py [output_dir] --method hsv
    python tests/artifacts/bdc_annotation_viz/generate_comparison.py [output_dir] --method rgb_kmeans

Default output_dir: tests/artifacts/bdc_annotation_viz/current

``hsv`` (BDcreationHE2) is compared to the committed CA_Boundary goldens.
``rgb_kmeans`` (BDcreationHE) is compared to ``dumps/he_testN/images.mat``
(MATLAB never seeded kmeans; Python uses seed 28).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage
from skimage import io

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pycurvelets._registration_quality import compute_mask_boundary_metrics  # noqa: E402
from pycurvelets.tumor_annotation_from_HE import (  # noqa: E402
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

FIXTURE = ROOT / "tests" / "test_for_shg_he_registration_BDcreation"
HE_ROOT = FIXTURE / "HE"
SHG_DIR = FIXTURE / "SHG"
GOLDEN_DIR = SHG_DIR / "CA_Boundary"
DUMPS = ROOT / "tests" / "matlab_parity" / "dumps"

CASES = [
    ("test1", 1.5, "HE_registered_test1", "BDcreationHE_test1results_mask for patient_001.tif.tif"),
    ("test2", 2.0, "HE_registered_test2", "BDcreationHE_test2results_mask for patient_001.tif.tif"),
    ("test3", 3.0, "HE_registered_test3", "BDcreationHE_test3results_mask for patient_001.tif.tif"),
]


def _load_he(path: Path) -> np.ndarray:
    im = io.imread(str(path))
    if im.ndim == 2:
        im = np.dstack([im, im, im])
    return im[..., :3]


def _as_bool_mask(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).squeeze()
    if a.dtype == np.bool_:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a > 0.5
    return a > 0


def _he01(he: np.ndarray) -> np.ndarray:
    a = he.astype(np.float64)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _boundary(mask: np.ndarray) -> np.ndarray:
    struct = np.ones((3, 3), dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    return m & ~ndimage.binary_erosion(m, structure=struct, border_value=0)


def _overlay_mask(he: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.45) -> np.ndarray:
    out = _he01(he).copy()
    c = np.asarray(color, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return np.clip(out, 0.0, 1.0)


def _contour_overlay(he: np.ndarray, matlab: np.ndarray, python: np.ndarray) -> np.ndarray:
    out = _he01(he).copy()
    mb = _boundary(matlab)
    pb = _boundary(python)
    both = mb & pb
    out[mb] = (0.15, 0.85, 1.0)
    out[pb] = (1.0, 0.2, 0.85)
    out[both] = (1.0, 1.0, 0.2)
    return out


def _tpfpfn_overlay(he: np.ndarray, matlab: np.ndarray, python: np.ndarray) -> np.ndarray:
    he_f = _he01(he)
    ml = np.asarray(matlab, dtype=bool)
    py = np.asarray(python, dtype=bool)
    rgb = np.zeros_like(he_f)
    rgb[py & ml] = (0.15, 0.82, 0.20)
    rgb[py & ~ml] = (0.95, 0.12, 0.10)
    rgb[~py & ml] = (0.15, 0.40, 1.00)
    neither = ~py & ~ml
    out = 0.40 * he_f + 0.60 * rgb
    out[neither] = he_f[neither]
    return np.clip(out, 0.0, 1.0)


def _load_he2_golden(name: str) -> np.ndarray:
    return _as_bool_mask(io.imread(str(GOLDEN_DIR / name)))


def _load_he_dump_mask(case_id: str) -> np.ndarray | None:
    path = DUMPS / f"he_{case_id}" / "images.mat"
    if not path.is_file():
        return None
    import scipy.io as sio  # noqa: E402

    ml = sio.loadmat(str(path), squeeze_me=True)
    return _as_bool_mask(ml["BDmask"])


def generate(
    case_id: str,
    ppm: float,
    he_folder: str,
    golden_name: str,
    method: str,
    out_dir: Path,
) -> dict[str, float] | None:
    he_path = HE_ROOT / he_folder / "patient_001.tif"
    if not he_path.is_file():
        print(f"skip {case_id} {method}: missing {he_path}")
        return None

    if method == "hsv":
        matlab = _load_he2_golden(golden_name)
        method_label = "HE2 / HSV (BDcreationHE2.m)"
        stem = f"comparison_he2_{case_id}_ppm{ppm}_hsv"
    elif method == "rgb_kmeans":
        matlab = _load_he_dump_mask(case_id)
        if matlab is None:
            print(f"skip {case_id} rgb_kmeans: missing dumps/he_{case_id}/images.mat")
            return None
        method_label = "HE / RGB k-means (BDcreationHE.m, seed=28)"
        stem = f"comparison_he_{case_id}_ppm{ppm}_rgb_kmeans"
    else:
        raise ValueError(method)

    he = _load_he(he_path)
    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_path.parent),
        HEfilename=he_path.name,
        pixelpermicron=ppm,
        areaThreshold=5000.0,
        SHGfilepath=str(SHG_DIR),
        annotation_method=method,
    )
    python, debug = tumor_annotation_from_he(params, save_output=False, return_debug=True)
    python = np.asarray(python, dtype=bool)
    matlab = np.asarray(matlab, dtype=bool)
    if python.shape != matlab.shape:
        print(f"skip {case_id} {method}: shape {python.shape} vs {matlab.shape}")
        return None

    metrics = compute_mask_boundary_metrics(python, matlab)
    n_diff = int(np.count_nonzero(python != matlab))
    n_pix = int(python.size)
    exact_pct = 100.0 * (1.0 - n_diff / n_pix)
    py_cov = 100.0 * float(python.mean())
    ml_cov = 100.0 * float(matlab.mean())
    tp = int((python & matlab).sum())
    fp = int((python & ~matlab).sum())
    fn = int((~python & matlab).sum())

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor="white")
    fig.suptitle(
        f"{case_id}  |  pixelpermicron = {ppm}  |  {method_label}\n"
        f"IoU={metrics['iou']:.4f}  |  Dice={metrics['dice']:.4f}  |  "
        f"Acc={metrics['pixel_accuracy']:.4f}  |  "
        f"Hausdorff={metrics['hausdorff_px']:.1f} px  |  "
        f"BoundF1={metrics['boundary_f1']:.4f}  |  "
        f"n_diff={n_diff} ({exact_pct:.2f}% exact)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(he)
    axes[0, 0].set_title("Registered HE Input", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(matlab, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("MATLAB Golden Mask", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(python, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Python Mask", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(_contour_overlay(he, matlab, python))
    axes[0, 3].set_title("Boundaries  (cyan MATLAB / magenta Python / yellow both)", fontsize=9)
    axes[0, 3].axis("off")

    axes[1, 0].imshow(_overlay_mask(he, matlab, (0.15, 0.75, 1.0)))
    axes[1, 0].set_title("MATLAB Mask on HE", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(_overlay_mask(he, python, (1.0, 0.25, 0.75)))
    axes[1, 1].set_title("Python Mask on HE", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(_tpfpfn_overlay(he, matlab, python))
    axes[1, 2].set_title("TP green / FP red / FN blue", fontsize=10)
    axes[1, 2].axis("off")

    xor = np.logical_xor(python, matlab)
    axes[1, 3].imshow(xor, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("Disagreement (XOR)", fontsize=10)
    axes[1, 3].axis("off")

    bound_img = np.zeros((*python.shape, 3), dtype=np.float64)
    bound_img[_boundary(matlab)] = (0.15, 0.85, 1.0)
    bound_img[_boundary(python)] = (1.0, 0.2, 0.85)
    bound_img[_boundary(matlab) & _boundary(python)] = (1.0, 1.0, 0.2)
    axes[2, 0].imshow(bound_img)
    axes[2, 0].set_title("Boundary Pixels Only", fontsize=10)
    axes[2, 0].axis("off")

    # Distance of disagreement pixels to the MATLAB boundary.
    ml_b = _boundary(matlab)
    if ml_b.any() and xor.any():
        dist = ndimage.distance_transform_edt(~ml_b)
        heat = np.zeros(python.shape, dtype=np.float64)
        heat[xor] = dist[xor]
        im = axes[2, 1].imshow(heat, cmap="hot", vmin=0)
        fig.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
    else:
        axes[2, 1].imshow(np.zeros(python.shape), cmap="hot", vmin=0, vmax=1)
    axes[2, 1].set_title("Disagreement Dist. to MATLAB Boundary (px)", fontsize=9)
    axes[2, 1].axis("off")

    axes[2, 2].bar(
        ["TP", "FP", "FN"],
        [tp, fp, fn],
        color=["#26d133", "#f21e1a", "#2666ff"],
        width=0.6,
    )
    axes[2, 2].set_title("Pixel Counts", fontsize=10)
    axes[2, 2].set_ylabel("pixels", fontsize=9)

    axes[2, 3].axis("off")
    axes[2, 3].set_title("Summary Statistics", fontsize=10)
    dbg_method = str(np.asarray(debug.get("annotation_method", [method])).squeeze())
    stats_text = (
        f"annotation:  {method_label}\n"
        f"debug method:{dbg_method}\n"
        f"ppm:         {ppm}\n"
        f"shape:       {python.shape[0]} x {python.shape[1]}\n"
        f"\n"
        f"vs MATLAB golden\n"
        f"  IoU:          {metrics['iou']:.6f}\n"
        f"  Dice:         {metrics['dice']:.6f}\n"
        f"  Pixel acc:    {metrics['pixel_accuracy']:.6f}\n"
        f"  Exact:        {exact_pct:.4f}%\n"
        f"  n_diff:       {n_diff}\n"
        f"  Hausdorff:    {metrics['hausdorff_px']:.2f} px\n"
        f"  Boundary F1:  {metrics['boundary_f1']:.4f}\n"
        f"\n"
        f"coverage\n"
        f"  MATLAB:  {ml_cov:.2f}%\n"
        f"  Python:  {py_cov:.2f}%\n"
        f"  TP/FP/FN:{tp}/{fp}/{fn}"
    )
    box = FancyBboxPatch(
        (0.05, 0.08), 0.9, 0.84,
        boxstyle="round,pad=0.05",
        facecolor="#f0f0f0",
        edgecolor="gray",
        transform=axes[2, 3].transAxes,
    )
    axes[2, 3].add_patch(box)
    axes[2, 3].text(
        0.5, 0.5, stats_text,
        transform=axes[2, 3].transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"{stem}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"  {case_id} {method}: IoU={metrics['iou']:.6f} Dice={metrics['dice']:.6f} "
        f"Acc={metrics['pixel_accuracy']:.6f} Hausdorff={metrics['hausdorff_px']:.2f} "
        f"BoundF1={metrics['boundary_f1']:.4f} n_diff={n_diff}"
    )
    return {
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "hausdorff_px": metrics["hausdorff_px"],
        "boundary_f1": metrics["boundary_f1"],
        "n_diff": float(n_diff),
        "exact_pct": exact_pct,
    }


def main() -> None:
    out_dir = ROOT / "tests" / "artifacts" / "bdc_annotation_viz" / "current"
    methods = ["hsv", "rgb_kmeans"]
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--method":
            methods = [next(it)]
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
        else:
            out_dir = Path(a)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        for case_id, ppm, folder, golden in CASES:
            generate(case_id, ppm, folder, golden, method, out_dir)


if __name__ == "__main__":
    main()
