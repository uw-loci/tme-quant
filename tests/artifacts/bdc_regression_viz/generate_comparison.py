"""Generate comparison visualizations for BDcreation_reg2 regression tests
(patient_001, tests 1-3: MATLAB golden vs Python at ppm 1.5 / 2.0 / 3.0).

Usage:
    python tests/artifacts/bdc_regression_viz/generate_comparison.py \
        [output_dir] [--method matlab|mi_ncc|oneplusone|mi|ncc] [--ecm hsv,rgb,...]

Default output_dir: tests/artifacts/bdc_regression_viz/current
Default method:     "matlab" (the package default - ITK v3 (1+1)-ES port,
                    pixel-exact vs MATLAB). Pass ``--method mi_ncc`` etc. to
                    picture a backup method instead.

Generates one figure per (case, ecm_method) combination named
``comparison_<case>_ppm<ppm>_<method>_<ecm>.png``.
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

from pycurvelets._he_bdc_common import matlab_rgb2gray
from pycurvelets._registration_quality import (
    compute_registration_quality_metrics,
    make_checkerboard,
)
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

REGISTRATION_METHOD = "matlab"
ECM_METHODS = ["hsv"]


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
    ppm: float,
    golden_folder: str,
    out_dir: Path,
    *,
    registration_method: str = REGISTRATION_METHOD,
    ecm_method: str = "hsv",
) -> None:
    he_raw = _load_uint8(HE_DIR / "patient_001.tif")
    shg_raw = _load_uint8(SHG_DIR / "patient_001.tif")
    matlab_golden = _load_uint8(HE_DIR / golden_folder / "patient_001.tif")

    params = SHGHERegistrationParameters(
        HEfilepath=str(HE_DIR),
        HEfilename="patient_001.tif",
        pixelpermicron=ppm,
        SHGfilepath=str(SHG_DIR),
        areaThreshold=5000.0,
        registration_method=registration_method,
        ecm_method=ecm_method,
    )
    py_float, debug = shg_he_registration(params, save_output=False, return_debug=True)
    py_uint8 = np.round(np.clip(py_float, 0, 1) * 255).astype(np.uint8)  # im2uint8 rounds
    shg_align = debug.get("shg_alignment") or {}
    shg_mi = float(shg_align.get("shg_mi", float("nan")))
    shg_ncc = float(shg_align.get("shg_ncc", float("nan")))
    shg_id_mi = float(debug.get("shg_alignment_identity_mi", float("nan")))
    shg_mi_delta = shg_mi - shg_id_mi if np.isfinite(shg_mi) and np.isfinite(shg_id_mi) else float("nan")

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

    ecm_label = {
        "hsv": "HSV (BDcreation_reg2)",
        "rgb": "RGB (BDcreation_reg)",
        "lab": "LAB k-means",
        "gray": "inverted luma",
        "auto": "auto (best SHG MI)",
    }
    ecm_display = ecm_label.get(ecm_method, ecm_method)
    ecm_selected = debug.get("ecm_method_selected", ecm_method)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor="white")
    fig.suptitle(
        f"{case_id}  |  pixelpermicron = {ppm}  |  method={registration_method}  |  ecm={ecm_display}\n"
        f"MAE={mae:.1f}  |  RMSE={rmse:.1f}  |  Exact={exact_pct:.1f}%  |  "
        f"PSNR={psnr:.1f} dB  |  SSIM={ssim:.4f}  |  "
        f"SHG MI={shg_mi:.4f} (Δid={shg_mi_delta:+.4f})  |  SHG NCC={shg_ncc:.4f}",
        fontsize=13,
        fontweight="bold",
    )

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

    ax5 = axes[1, 0]
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
    ax7.set_title("50/50 Blend (MATLAB / Python)", fontsize=10)
    ax7.axis("off")

    ax_shg = axes[1, 3]
    shg_f = shg_raw.astype(np.float64)
    if shg_f.max() > 1.0:
        shg_f = shg_f / 255.0
    if shg_f.ndim == 3:
        shg_f = matlab_rgb2gray(shg_f)
    shg_rgb = np.stack([shg_f, shg_f, shg_f], axis=-1)
    py_rgb01 = py_uint8.astype(np.float64) / 255.0
    if shg_rgb.shape[:2] != py_rgb01.shape[:2]:
        from pycurvelets._he_bdc_common import resize_like

        shg_rgb = resize_like(shg_rgb, py_rgb01.shape[:2])
    he_shg_checker = make_checkerboard(py_rgb01, shg_rgb, block=64)
    ax_shg.imshow(np.clip(he_shg_checker, 0, 1))
    ax_shg.set_title("Python HE vs SHG checkerboard", fontsize=10)
    ax_shg.axis("off")

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
        f"registration: {registration_method}\n"
        f"ecm requested: {ecm_method}\n"
        f"ecm selected:  {ecm_selected}\n"
        f"ppm:           {ppm}\n"
        f"\n"
        f"vs MATLAB golden\n"
        f"  Shape: {py_uint8.shape}\n"
        f"  MAE:  {mae:.2f} / 255\n"
        f"  RMSE: {rmse:.2f} / 255\n"
        f"  PSNR: {psnr:.2f} dB\n"
        f"  SSIM: {ssim:.4f}\n"
        f"  Max |Diff|: {max_diff}\n"
        f"  Exact match: {exact_pct:.1f}%\n"
        f"  Within  5: {w5:.1f}%\n"
        f"  Within 10: {w10:.1f}%\n"
        f"  Within 20: {w20:.1f}%\n"
        f"\n"
        f"vs SHG (ECM moving image)\n"
        f"  SHG MI:     {shg_mi:.4f}\n"
        f"  Identity MI:{shg_id_mi:.4f}\n"
        f"  MI Δid:     {shg_mi_delta:+.4f}\n"
        f"  SHG NCC:    {shg_ncc:.4f}"
    )
    box = FancyBboxPatch(
        (0.05, 0.08), 0.9, 0.84,
        boxstyle="round,pad=0.05",
        facecolor="#f0f0f0",
        edgecolor="gray",
        transform=ax11.transAxes,
    )
    ax11.add_patch(box)
    ax11.text(
        0.5, 0.5, stats_text,
        transform=ax11.transAxes,
        fontsize=7,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"comparison_{case_id}_ppm{ppm}_{registration_method}_{ecm_method}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(
        f"  {case_id} ppm={ppm} method={registration_method} ecm={ecm_method}: "
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, "
        f"Exact={exact_pct:.1f}%, Within5={w5:.1f}%, Within10={w10:.1f}%, Within20={w20:.1f}%, "
        f"SHG_MI={shg_mi:.4f} (Δid={shg_mi_delta:+.4f}), SHG_NCC={shg_ncc:.4f}"
    )


def _parse_cli(argv: list[str]) -> tuple[Path, str, list[str]]:
    """``[out_dir] [--method M] [--ecm a,b]`` - defaults keep old behaviour."""
    out_dir = ROOT / "tests" / "artifacts" / "bdc_regression_viz" / "current"
    method = REGISTRATION_METHOD
    ecm_methods = list(ECM_METHODS)
    it = iter(argv)
    for a in it:
        if a == "--method":
            method = next(it)
        elif a == "--ecm":
            ecm_methods = next(it).split(",")
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
        else:
            out_dir = Path(a)
    return out_dir, method, ecm_methods


def main() -> None:
    out_dir, method, ecm_methods = _parse_cli(sys.argv[1:])
    out_dir.mkdir(parents=True, exist_ok=True)

    for ecm_method in ecm_methods:
        for case_id, ppm, folder in CASES:
            generate(
                case_id,
                ppm,
                folder,
                out_dir,
                registration_method=method,
                ecm_method=ecm_method,
            )


if __name__ == "__main__":
    main()
