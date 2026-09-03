"""Generate comparison visualizations for patient_02 test cases (tests 4-9).

Usage:
    python tests/artifacts/bdc_regression_viz/generate_comparison_patient02.py [output_dir]

Default output_dir: tests/artifacts/bdc_regression_viz/new_patient02

Test case definitions:
  Tests 4-7 use BDcreation_reg2.m goldens (HSV-based registration).
  Tests 8-9 use BDcreation_reg.m goldens (RGB-based registration).
  Both are compared against the Python mi_ncc + HSV pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from skimage import io

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pycurvelets._registration_quality import compute_registration_quality_metrics
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

NEW_DATA = ROOT / "tests" / "test_for_shg_he_registration_BDcreation" / "new_test_datasets_tests4-5-6-7"
HE_DIR = NEW_DATA / "HE"
SHG_DIR = NEW_DATA / "SHG"

# (case_id, he_filename, ppm, golden_folder, matlab_reg, notes)
CASES = [
    # Test 4: roi2, reg2 (HSV), ppm=2.6
    ("test4_reg2_roi2_ppm2.6", "patient_02_roi2.tif", 2.6,
     "HE_registered_for_reg2_test4_roi2_ppm2p6", "reg2",
     "roi2, ppm=2.6, BDcreation_reg2 HSV"),
    # Test 5: roi4, reg2 (HSV), ppm=1.5
    ("test5_reg2_roi4_ppm1.5", "patient_02_roi4.tif", 1.5,
     "HE_registered_for_reg2_test5_roi4_ppm1p5", "reg2",
     "roi4, ppm=1.5, BDcreation_reg2 HSV"),
    # Test 6: roi4, reg2 (HSV), ppm=2.6
    ("test6_reg2_roi4_ppm2.6", "patient_02_roi4.tif", 2.6,
     "HE_registered_for_reg2_test6_roi4_ppm2p6", "reg2",
     "roi4, ppm=2.6, BDcreation_reg2 HSV"),
    # Test 7: roi5, reg2 (HSV), ppm=2.6
    ("test7_reg2_roi5_ppm2.6", "patient_02_roi5.tif", 2.6,
     "HE_registered_for_reg2_test7_roi5_ppm2p6", "reg2",
     "roi5, ppm=2.6, BDcreation_reg2 HSV"),
    # Test 8: roi4, reg1 (RGB), ppm=3
    ("test8_reg1_roi4_ppm3.0", "patient_02_roi4.tif", 3.0,
     "HE_registered_for_reg1_test6b_ppm3", "reg1",
     "roi4, ppm=3.0, BDcreation_reg (RGB) golden"),
    # Test 9: roi4, reg1 (RGB), ppm=2.6
    ("test9_reg1_roi4_ppm2.6", "patient_02_roi4.tif", 2.6,
     "HE_registered_for_reg1_test9_ppm2p6", "reg1",
     "roi4, ppm=2.6, BDcreation_reg (RGB) golden"),
]


def _load_uint8(path: Path) -> np.ndarray:
    im = io.imread(str(path))
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def _checkerboard(a: np.ndarray, b: np.ndarray, block: int = 64) -> np.ndarray:
    h, w = a.shape[:2]
    out = np.empty_like(a)
    for y in range(0, h, block):
        for x in range(0, w, block):
            iy, ix = y // block, x // block
            src = a if (iy + ix) % 2 == 0 else b
            out[y : y + block, x : x + block] = src[y : y + block, x : x + block]
    return out


def generate(
    case_id: str,
    he_filename: str,
    ppm: float,
    golden_folder: str,
    matlab_reg: str,
    notes: str,
    out_dir: Path,
    registration_method: str = "mi_ncc",
) -> dict:
    he_path = HE_DIR / he_filename
    shg_path = SHG_DIR / he_filename
    golden_path = HE_DIR / golden_folder / he_filename

    for p in (he_path, shg_path, golden_path):
        if not p.is_file():
            print(f"SKIP {case_id}: missing {p}")
            return {}

    he_raw = _load_uint8(he_path)
    shg_raw = _load_uint8(shg_path)
    matlab_golden = _load_uint8(golden_path)

    params = SHGHERegistrationParameters(
        HEfilepath=str(HE_DIR),
        HEfilename=he_filename,
        pixelpermicron=ppm,
        SHGfilepath=str(SHG_DIR),
        areaThreshold=5000.0,
        registration_method=registration_method,
    )
    py_float = shg_he_registration(params, save_output=False, return_debug=False)
    py_uint8 = np.round(np.clip(py_float, 0, 1) * 255).astype(np.uint8)  # im2uint8 rounds

    diff_signed = py_uint8.astype(np.float64) - matlab_golden.astype(np.float64)
    diff_abs = np.abs(diff_signed)
    metrics = compute_registration_quality_metrics(py_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    rmse = float(metrics["rmse_uint8"])
    max_diff = int(np.max(diff_abs))
    exact_pct = float(metrics["exact_frac"]) * 100
    w5 = float(metrics["within5_frac"]) * 100
    w10 = float(metrics["within10_frac"]) * 100
    w20 = float(metrics["within20_frac"]) * 100
    psnr = float(metrics["psnr"])
    ssim = float(metrics["ssim"])

    matlab_label = f"MATLAB Golden ({matlab_reg})"

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor="white")
    fig.suptitle(
        f"{case_id}  |  ppm={ppm}  |  method=mi_ncc  |  ecm=HSV\n"
        f"MAE={mae:.1f}  |  RMSE={rmse:.1f}  |  Exact={exact_pct:.1f}%  |  "
        f"PSNR={psnr:.1f} dB  |  SSIM={ssim:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(he_raw)
    axes[0, 0].set_title("Raw HE Input", fontsize=10)
    axes[0, 0].axis("off")

    shg_show = shg_raw if shg_raw.ndim == 2 else shg_raw
    axes[0, 1].imshow(shg_show, cmap="gray" if shg_show.ndim == 2 else None)
    axes[0, 1].set_title("SHG Input", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(matlab_golden)
    axes[0, 2].set_title(matlab_label, fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(py_uint8)
    axes[0, 3].set_title("Python Output", fontsize=10)
    axes[0, 3].axis("off")

    h_img, w_img = matlab_golden.shape[:2]
    split = np.copy(matlab_golden)
    split[:, w_img // 2:] = py_uint8[:, w_img // 2:]
    axes[1, 0].imshow(split)
    axes[1, 0].axvline(x=w_img // 2, color="yellow", linewidth=1.5)
    axes[1, 0].set_title("MATLAB (left)  vs  Python (right)", fontsize=10)
    axes[1, 0].axis("off")

    checker = _checkerboard(matlab_golden, py_uint8, block=64)
    axes[1, 1].imshow(checker)
    axes[1, 1].set_title("Checkerboard Overlay", fontsize=10)
    axes[1, 1].axis("off")

    blend = (0.5 * matlab_golden.astype(np.float64) + 0.5 * py_uint8.astype(np.float64)).astype(np.uint8)
    axes[1, 2].imshow(blend)
    axes[1, 2].set_title("50/50 Blend", fontsize=10)
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")

    diff_x3 = np.clip(diff_abs * 3, 0, 255).astype(np.uint8)
    axes[2, 0].imshow(diff_x3)
    axes[2, 0].set_title("|Difference| x3", fontsize=10)
    axes[2, 0].axis("off")

    mean_diff = np.mean(diff_abs, axis=2) if diff_abs.ndim == 3 else diff_abs
    im9 = axes[2, 1].imshow(mean_diff, cmap="hot", vmin=0)
    axes[2, 1].set_title("Per-pixel Mean |Diff|", fontsize=10)
    axes[2, 1].axis("off")
    fig.colorbar(im9, ax=axes[2, 1], fraction=0.046, pad=0.04)

    signed_flat = diff_signed.ravel()
    axes[2, 2].hist(signed_flat, bins=256, range=(-255, 255), color="steelblue", edgecolor="none")
    axes[2, 2].axvline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[2, 2].set_xlabel("Python - MATLAB", fontsize=9)
    axes[2, 2].set_ylabel("Pixel count", fontsize=9)
    axes[2, 2].set_title("Signed Diff Distribution", fontsize=10)

    ax11 = axes[2, 3]
    ax11.axis("off")
    ax11.set_title("Summary Statistics", fontsize=10)
    stats_text = (
        f"registration: mi_ncc\n"
        f"ecm_method:   hsv\n"
        f"ppm:          {ppm}\n"
        f"file:         {he_filename}\n"
        f"matlab_golden: {matlab_reg}\n"
        f"notes: {notes}\n"
        f"\n"
        f"Shape: {py_uint8.shape}\n"
        f"MAE:  {mae:.2f} / 255\n"
        f"RMSE: {rmse:.2f} / 255\n"
        f"PSNR: {psnr:.2f} dB\n"
        f"SSIM: {ssim:.4f}\n"
        f"Max |Diff|: {max_diff}\n"
        f"Exact match: {exact_pct:.1f}%\n"
        f"Within  5: {w5:.1f}%\n"
        f"Within 10: {w10:.1f}%\n"
        f"Within 20: {w20:.1f}%"
    )
    box = FancyBboxPatch(
        (0.05, 0.05), 0.9, 0.88,
        boxstyle="round,pad=0.05",
        facecolor="#f0f0f0",
        edgecolor="gray",
        transform=ax11.transAxes,
    )
    ax11.add_patch(box)
    ax11.text(
        0.5, 0.5, stats_text,
        transform=ax11.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"comparison_{case_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"  {case_id}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, "
        f"Exact={exact_pct:.1f}%"
    )
    return metrics


def main() -> None:
    """``[out_dir] [--method M] [--cases id1,id2]``."""
    out_dir = ROOT / "tests" / "artifacts" / "bdc_regression_viz" / "new_patient02"
    method = "mi_ncc"
    selected: list[str] | None = None
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--method":
            method = next(it)
        elif a == "--cases":
            selected = next(it).split(",")
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
        else:
            out_dir = Path(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASES if selected is None or c[0] in selected]

    print(f"Running {len(cases)} patient_02 test cases ({method} + HSV)...\n")
    all_metrics = {}
    for case_id, he_fn, ppm, golden, matlab_reg, notes in cases:
        m = generate(case_id, he_fn, ppm, golden, matlab_reg, notes, out_dir,
                     registration_method=method)
        if m:
            all_metrics[case_id] = m
        print()

    if all_metrics:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Case':<35s} {'MAE':>6s} {'RMSE':>7s} {'PSNR':>7s} {'SSIM':>7s} {'Exact%':>7s}")
        print("-" * 80)
        for cid, m in all_metrics.items():
            print(
                f"{cid:<35s} "
                f"{m['mae_uint8']:6.2f} "
                f"{m['rmse_uint8']:7.2f} "
                f"{m['psnr']:7.2f} "
                f"{m['ssim']:7.4f} "
                f"{m['exact_frac']*100:6.1f}%"
            )


if __name__ == "__main__":
    main()
