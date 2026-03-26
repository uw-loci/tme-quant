#!/usr/bin/env python3
"""
Visualize Python vs MATLAB golden discrepancies for BDcreation regression fixtures.

Writes PNGs under ``tests/artifacts/bdc_regression_viz/`` (gitignored via tests/artifacts/).

Run from repo root: ``uv run python tests/visualize_bdc_regression_discrepancies.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from pycurvelets.SHG_HE_registration import SHGHERegistrationParameters, shg_he_registration
from pycurvelets.tumor_annotation_from_HE import TumorAnnotationFromHEParameters, tumor_annotation_from_he

_ROOT = Path(__file__).resolve().parent
_FIXTURE = _ROOT / "test_for_shg_he_registration_BDcreation"
_OUT = _ROOT / "artifacts" / "bdc_regression_viz"


def _ensure_out() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)


def _plot_registration() -> None:
    base = _FIXTURE
    cases = [
        ("test1", 1.5, "HE_registered_test1"),
        ("test2", 2.0, "HE_registered_test2"),
        ("test3", 3.0, "HE_registered_test3"),
    ]
    for cid, ppm, folder in cases:
        gpath = base / "HE" / folder / "patient_001.tif"
        golden = io.imread(gpath).astype(np.float64) / 255.0
        p = SHGHERegistrationParameters(
            HEfilepath=str(base / "HE"),
            HEfilename="patient_001.tif",
            pixelpermicron=ppm,
            SHGfilepath=str(base / "SHG"),
            areaThreshold=5000.0,
        )
        reg = shg_he_registration(p, save_output=False).astype(np.float64)
        diff = reg - golden
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))

        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        fig.suptitle(
            f"Registration {cid} ppm={ppm}  MAE={mae:.6f}  RMSE={rmse:.6f}",
            fontsize=12,
        )
        ch_names = ("R", "G", "B")
        for c in range(3):
            axes[0, c].imshow(np.clip(golden[..., c], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[0, c].set_title(f"MATLAB golden {ch_names[c]}")
            axes[0, c].axis("off")
            axes[1, c].imshow(np.clip(reg[..., c], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[1, c].set_title(f"Python {ch_names[c]}")
            axes[1, c].axis("off")
            e = np.abs(diff[..., c])
            vmax = max(0.02, float(e.max()))
            im = axes[2, c].imshow(e, cmap="magma", vmin=0, vmax=vmax)
            axes[2, c].set_title(f"|Δ| {ch_names[c]}")
            axes[2, c].axis("off")
            plt.colorbar(im, ax=axes[2, c], fraction=0.046)
        fig.tight_layout()
        out = _OUT / f"registration_{cid}_ppm{ppm}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def _plot_annotation() -> None:
    base = _FIXTURE
    cases = [
        ("test1", 1.5, "HE_registered_test1", "BDcreationHE_test1results_mask for patient_001.tif.tif"),
        ("test2", 2.0, "HE_registered_test2", "BDcreationHE_test2results_mask for patient_001.tif.tif"),
        ("test3", 3.0, "HE_registered_test3", "BDcreationHE_test3results_mask for patient_001.tif.tif"),
    ]
    for cid, ppm, hef, mf in cases:
        he_path = base / "HE" / hef / "patient_001.tif"
        he = io.imread(he_path).astype(np.float64)
        if he.max() > 1.0:
            he = he / 255.0
        he_gray = 0.2989 * he[..., 0] + 0.5870 * he[..., 1] + 0.1140 * he[..., 2]

        params = TumorAnnotationFromHEParameters(
            HEfilepath=str(base / "HE" / hef),
            HEfilename="patient_001.tif",
            pixelpermicron=ppm,
            areaThreshold=5000.0,
            SHGfilepath=str(base / "SHG"),
        )
        pred = tumor_annotation_from_he(params, save_output=False)
        gold = io.imread(base / "SHG" / "CA_Boundary" / mf) > 127

        tp = pred & gold
        fp = pred & (~gold)
        fn = (~pred) & gold

        rgb = np.zeros((*pred.shape, 3), dtype=np.float64)
        rgb[..., 0] = fp  # red: false positive
        rgb[..., 1] = tp  # green: true positive
        rgb[..., 2] = fn  # blue: false negative

        p, g = pred.ravel(), gold.ravel()
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        iou = inter / union if union else 1.0
        dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
        acc = float((p == g).mean())

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(
            f"Annotation {cid} ppm={ppm}  IoU={iou:.4f}  Dice={dice:.4f}  Acc={acc:.4f}\n"
            f"Overlay: green=TP, red=FP, blue=FN  |  pred_fg={int(pred.sum())}  gold_fg={int(gold.sum())}",
            fontsize=11,
        )
        axes[0].imshow(np.clip(he_gray, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("H&E (MATLAB-registered, luma)")
        axes[0].axis("off")
        axes[1].imshow(gold, cmap="gray")
        axes[1].set_title("MATLAB golden mask")
        axes[1].axis("off")
        axes[2].imshow(np.clip(rgb, 0, 1))
        axes[2].set_title("Python vs golden (RGB)")
        axes[2].axis("off")
        fig.tight_layout()
        out = _OUT / f"annotation_{cid}_ppm{ppm}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def main() -> None:
    _ensure_out()
    print("Generating registration figures (runs SimpleITK; may take ~3 min)...")
    _plot_registration()
    print("Generating annotation figures...")
    _plot_annotation()
    print(f"Done. Figures in {_OUT}")


if __name__ == "__main__":
    main()
