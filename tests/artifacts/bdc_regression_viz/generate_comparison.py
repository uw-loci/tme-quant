"""Generate comparison visualizations for BDcreation_reg2 regression tests.

Usage:
    python tests/artifacts/bdc_regression_viz/generate_comparison.py [output_dir]

Default output_dir: tests/artifacts/bdc_regression_viz/new
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

FIXTURE = ROOT / "tests" / "test_for_shg_he_registration_BDcreation"
HE_DIR = FIXTURE / "HE"
SHG_DIR = FIXTURE / "SHG"

CASES = [
    ("test1", 1.5, "HE_registered_test1"),
    ("test2", 2.0, "HE_registered_test2"),
    ("test3", 3.0, "HE_registered_test3"),
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


def generate(case_id: str, ppm: float, golden_folder: str, out_dir: Path) -> None:
    he_raw = _load_uint8(HE_DIR / "patient_001.tif")
    shg_raw = _load_uint8(SHG_DIR / "patient_001.tif")
    matlab_golden = _load_uint8(HE_DIR / golden_folder / "patient_001.tif")

    params = SHGHERegistrationParameters(
        HEfilepath=str(HE_DIR),
        HEfilename="patient_001.tif",
        pixelpermicron=ppm,
        SHGfilepath=str(SHG_DIR),
        areaThreshold=5000.0,
    )
    py_float = shg_he_registration(params, save_output=False, return_debug=False)
    py_uint8 = (np.clip(py_float, 0, 1) * 255).astype(np.uint8)

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

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor="white")
    fig.suptitle(
        f"{case_id}  |  pixelpermicron = {ppm}  |  MAE = {mae:.1f}  |  "
        f"RMSE = {rmse:.1f}  |  Exact = {exact_pct:.1f}%  |  "
        f"PSNR = {psnr:.1f} dB  |  SSIM = {ssim:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Raw HE | SHG | MATLAB Golden | Python Output
    ax1 = axes[0, 0]
    ax1.imshow(he_raw)
    ax1.set_title("Raw HE Input", fontsize=10)
    ax1.axis("off")

    ax2 = axes[0, 1]
    shg_show = shg_raw if shg_raw.ndim == 2 else shg_raw
    ax2.imshow(shg_show, cmap="gray" if shg_show.ndim == 2 else None)
    ax2.set_title("SHG Input", fontsize=10)
    ax2.axis("off")

    ax3 = axes[0, 2]
    ax3.imshow(matlab_golden)
    ax3.set_title("MATLAB Golden", fontsize=10)
    ax3.axis("off")

    ax4 = axes[0, 3]
    ax4.imshow(py_uint8)
    ax4.set_title("Python Output", fontsize=10)
    ax4.axis("off")

    # Row 2: Side-by-side | Checkerboard | 50/50 Blend | (empty)
    ax5 = axes[1, 0]
    h_img = matlab_golden.shape[0]
    w_img = matlab_golden.shape[1]
    split = np.copy(matlab_golden)
    split[:, w_img // 2 :] = py_uint8[:, w_img // 2 :]
    ax5.imshow(split)
    ax5.axvline(x=w_img // 2, color="yellow", linewidth=1.5)
    ax5.set_title("MATLAB (left)  vs  Python (right)", fontsize=10)
    ax5.axis("off")

    ax6 = axes[1, 1]
    checker = _checkerboard(matlab_golden, py_uint8, block=64)
    ax6.imshow(checker)
    ax6.set_title("Checkerboard Overlay", fontsize=10)
    ax6.axis("off")

    ax7 = axes[1, 2]
    blend = (0.5 * matlab_golden.astype(np.float64) + 0.5 * py_uint8.astype(np.float64)).astype(
        np.uint8
    )
    ax7.imshow(blend)
    ax7.set_title("50/50 Blend", fontsize=10)
    ax7.axis("off")

    axes[1, 3].axis("off")

    # Row 3: |Diff|x3 | Per-pixel Mean|Diff| heatmap | Histogram | Stats
    ax8 = axes[2, 0]
    diff_x3 = np.clip(diff_abs * 3, 0, 255).astype(np.uint8)
    ax8.imshow(diff_x3)
    ax8.set_title("|Difference| x3", fontsize=10)
    ax8.axis("off")

    ax9 = axes[2, 1]
    mean_diff = np.mean(diff_abs, axis=2) if diff_abs.ndim == 3 else diff_abs
    im9 = ax9.imshow(mean_diff, cmap="hot", vmin=0)
    ax9.set_title("Per-pixel Mean |Diff|", fontsize=10)
    ax9.axis("off")
    fig.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    ax10 = axes[2, 2]
    signed_flat = diff_signed.ravel()
    ax10.hist(signed_flat, bins=256, range=(-255, 255), color="steelblue", edgecolor="none")
    ax10.axvline(0, color="gray", linestyle="-", linewidth=0.8)
    ax10.set_xlabel("Python - MATLAB", fontsize=9)
    ax10.set_ylabel("Pixel count", fontsize=9)
    ax10.set_title("Signed Diff Distribution", fontsize=10)

    ax11 = axes[2, 3]
    ax11.axis("off")
    ax11.set_title("Summary Statistics", fontsize=10)
    stats_text = (
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
        (0.05, 0.15), 0.9, 0.7,
        boxstyle="round,pad=0.05",
        facecolor="#f0f0f0",
        edgecolor="gray",
        transform=ax11.transAxes,
    )
    ax11.add_patch(box)
    ax11.text(
        0.5, 0.5, stats_text,
        transform=ax11.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"comparison_{case_id}_ppm{ppm}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"  {case_id} ppm={ppm}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, "
        f"Exact={exact_pct:.1f}%, Within5={w5:.1f}%, Within10={w10:.1f}%, Within20={w20:.1f}%"
    )


def main() -> None:
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    else:
        out_dir = ROOT / "tests" / "artifacts" / "bdc_regression_viz" / "new"
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id, ppm, folder in CASES:
        generate(case_id, ppm, folder, out_dir)


if __name__ == "__main__":
    main()
