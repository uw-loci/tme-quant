"""Dev-only check of patient_003 and patient_004 annotation against MATLAB.

Not part of the pytest suite and not shipped. ``archive/`` is pruned from
the sdist. The H&E files and MATLAB ``BDmask.mat`` references live next to
this script. The figures it writes stay local and are gitignored.

From the repository root::

    python archive/visualization/validate_annotation_patient003_004.py

That reads ``patient003_004_inputs/`` and writes the PNGs and ``SUMMARY.md``
under ``archive/visualization/patient003_004/``. Pass another directory as
the first argument to write the figures somewhere else.

Exits 1 when a mask disagrees with the recorded pixel counts. HSV is
expected to match every pixel. RGB may differ by the recorded edge pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import ndimage
from skimage import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_annotation_comparison import (  # noqa: E402
    _boundary,
    _contour_overlay,
    _he01,
    _overlay_mask,
    _tpfpfn_overlay,
)
from pycurvelets._registration_quality import compute_mask_boundary_metrics  # noqa: E402
from pycurvelets.tumor_annotation_from_HE import (  # noqa: E402
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

_HERE = Path(__file__).resolve().parent
_INPUTS = _HERE / "patient003_004_inputs"
HE_DIR = _INPUTS / "HE"
MATLAB_ROOT = _INPUTS / "matlab"
DEFAULT_OUT = _HERE / "patient003_004"

# ppm tag in the MATLAB folder name, and the pixel count recorded against
# BDcreationHE2 / seeded BDcreationHE. RGB leftovers are edge pixels.
CASES = (
    ("patient_003", 1.5, "1p5"),
    ("patient_003", 2.0, "2p0"),
    ("patient_003", 3.0, "3p0"),
    ("patient_004", 1.5, "1p5"),
    ("patient_004", 2.0, "2p0"),
    ("patient_004", 3.0, "3p0"),
)
METHODS = (
    ("hsv", "hsv", "HE2 / HSV (BDcreationHE2.m)", 0),
    ("rgb_kmeans", "rgb", "HE / RGB k-means (BDcreationHE.m, seed=28)", None),
)
RECORDED_RGB_N_DIFF = {
    ("patient_003", 1.5): 1,
    ("patient_003", 2.0): 0,
    ("patient_003", 3.0): 0,
    ("patient_004", 1.5): 1,
    ("patient_004", 2.0): 0,
    ("patient_004", 3.0): 7,
}


def _as_bool(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).squeeze() > 0


def _matlab_mask(patient: str, ppm_tag: str, matlab_method: str) -> np.ndarray:
    path = MATLAB_ROOT / f"{patient}_ppm{ppm_tag}_{matlab_method}" / "BDmask.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _as_bool(sio.loadmat(str(path), squeeze_me=True)["BDmask"])


def _zoom(he: np.ndarray, matlab: np.ndarray, python: np.ndarray) -> np.ndarray:
    """Crop the H&E around differing pixels and mark them.

    An exact match returns the full H&E. A difference returns a padded crop
    with yellow marks on pixels that do not agree.
    """
    diff = np.asarray(matlab, dtype=bool) != np.asarray(python, dtype=bool)
    base = _tpfpfn_overlay(he, matlab, python)
    if not diff.any():
        return base
    ys, xs = np.nonzero(diff)
    pad = 50
    r0 = max(int(ys.min()) - pad, 0)
    r1 = min(int(ys.max()) + pad + 1, diff.shape[0])
    c0 = max(int(xs.min()) - pad, 0)
    c1 = min(int(xs.max()) + pad + 1, diff.shape[1])
    crop = base[r0:r1, c0:c1].copy()
    crop[ys - r0, xs - c0] = (1.0, 1.0, 0.0)
    return crop


def _figure(
    he: np.ndarray,
    matlab: np.ndarray,
    python: np.ndarray,
    *,
    patient: str,
    ppm: float,
    method_label: str,
    metrics: dict[str, float],
    n_diff: int,
    out_path: Path,
) -> None:
    n_pix = int(python.size)
    exact_pct = 100.0 * (1.0 - n_diff / n_pix)
    tp = int((python & matlab).sum())
    fp = int((python & ~matlab).sum())
    fn = int((~python & matlab).sum())
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor="white")
    fig.suptitle(
        f"{patient}  |  pixelpermicron = {ppm}  |  {method_label}\n"
        f"IoU={metrics['iou']:.4f}  |  Dice={metrics['dice']:.4f}  |  "
        f"n_diff={n_diff} ({exact_pct:.4f}% exact)  |  "
        f"Hausdorff={metrics['hausdorff_px']:.1f} px",
        fontsize=13,
        fontweight="bold",
    )
    panels = (
        (he, "H&E input", None),
        (matlab, "MATLAB mask", "gray"),
        (python, "Python mask", "gray"),
        (_contour_overlay(he, matlab, python), "Boundaries (cyan MATLAB / magenta Python / yellow both)", None),
        (_overlay_mask(he, matlab, (0.15, 0.75, 1.0)), "MATLAB mask on H&E", None),
        (_overlay_mask(he, python, (1.0, 0.25, 0.75)), "Python mask on H&E", None),
        (_tpfpfn_overlay(he, matlab, python), "TP green / Python-only red / MATLAB-only blue", None),
        (_zoom(he, matlab, python), "Zoom on disagreement (yellow)", None),
    )
    for ax, (img, title, cmap) in zip(axes.ravel()[:8], panels, strict=True):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    xor = np.logical_xor(python, matlab)
    ml_b = _boundary(matlab)
    if ml_b.any() and xor.any():
        dist = ndimage.distance_transform_edt(~ml_b)
        heat = np.zeros(python.shape, dtype=np.float64)
        heat[xor] = dist[xor]
        im = axes[2, 0].imshow(heat, cmap="hot", vmin=0)
        fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    else:
        axes[2, 0].imshow(np.zeros(python.shape), cmap="hot", vmin=0, vmax=1)
    axes[2, 0].set_title("Disagreement distance to MATLAB boundary (px)", fontsize=9)
    axes[2, 0].axis("off")

    axes[2, 1].bar(["TP", "Python only", "MATLAB only"], [tp, fp, fn], color=["#26d133", "#f21e1a", "#2666ff"], width=0.6)
    axes[2, 1].set_title("Pixel counts", fontsize=10)
    axes[2, 1].tick_params(axis="x", labelsize=8)

    axes[2, 2].axis("off")
    axes[2, 3].axis("off")
    stats = (
        f"ppm:        {ppm}\n"
        f"shape:      {python.shape[0]} x {python.shape[1]}\n"
        f"IoU:        {metrics['iou']:.6f}\n"
        f"Dice:       {metrics['dice']:.6f}\n"
        f"n_diff:     {n_diff}\n"
        f"exact:      {exact_pct:.4f}%\n"
        f"Hausdorff:  {metrics['hausdorff_px']:.2f} px\n"
        f"boundary F1:{metrics['boundary_f1']:.4f}\n"
        f"MATLAB cov: {100.0 * float(matlab.mean()):.2f}%\n"
        f"Python cov: {100.0 * float(python.mean()):.2f}%"
    )
    axes[2, 3].text(0.05, 0.5, stats, fontsize=11, fontfamily="monospace", verticalalignment="center")
    axes[2, 3].set_title("Counts", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_summary(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    lines = [
        "# Patient 003 and 004 annotation check",
        "",
        "Local development check only. These slides and this write-up are not part of the pytest suite and are not shipped.",
        "",
        "Python `tumor_annotation_from_he` compared with MATLAB `BDcreationHE2.m` (HSV) and `BDcreationHE.m` (RGB k-means). The RGB MATLAB run called `rng(28, 'twister')` immediately before `BDcreationHE`. The scales 1.5, 2.0, and 3.0 px/um are the patient_001 test values, not a measured resolution of these files.",
        "",
        "| Image | px/um | Method | n_diff | Python only | MATLAB only | IoU | MATLAB coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['patient']} | {row['ppm']} | {row['method']} | {row['n_diff']} | "
            f"{row['python_only']} | {row['matlab_only']} | {row['iou']:.6f} | {row['coverage']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "HSV matches MATLAB on every pixel, including the empty `patient_004` mask at 3.0 px/um. RGB differs only by the edge pixels in the table: one pixel on `patient_003` at 1.5, one pixel on `patient_004` at 1.5, and a 7-pixel step on `patient_004` at 3.0. IoU stays above 0.99997. The single-pixel miss on `patient_003` shows this is not a failure of the `patient_004` slide.",
            "",
            "`patient_004` HSV is a weak check of tumor shape. Both programs cover 0.22% and 0.27% at 1.5 and 2.0 px/um, and both return an empty mask at 3.0 px/um. The RGB masks on that slide cover about 24% to 26%.",
            "",
            "Figures in this folder are the H&E, both masks, the overlay, and a zoom on any yellow disagreement pixels.",
            "",
        ]
    )
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    out_dir = DEFAULT_OUT if len(sys.argv) < 2 else Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for patient, ppm, ppm_tag in CASES:
        he_path = HE_DIR / f"{patient}.tif"
        if not he_path.is_file():
            raise SystemExit(f"missing {he_path}")
        he = io.imread(str(he_path))
        if he.ndim == 2:
            he = np.dstack([he, he, he])
        he = he[..., :3]
        for method, matlab_method, method_label, hsv_expected in METHODS:
            matlab = _matlab_mask(patient, ppm_tag, matlab_method)
            params = TumorAnnotationFromHEParameters(
                HEfilepath=str(he_path.parent),
                HEfilename=he_path.name,
                pixelpermicron=ppm,
                SHGfilepath=str(out_dir),
                annotation_method=method,
            )
            python = np.asarray(
                tumor_annotation_from_he(params, save_output=False, return_debug=False),
                dtype=bool,
            )
            if python.shape != matlab.shape:
                failures.append(f"{patient} {method} ppm {ppm}: shape {python.shape} vs {matlab.shape}")
                continue
            metrics = compute_mask_boundary_metrics(python, matlab)
            n_diff = int(np.count_nonzero(python != matlab))
            expected = hsv_expected if method == "hsv" else RECORDED_RGB_N_DIFF[(patient, ppm)]
            stem = f"{patient}_ppm{ppm_tag}_{'hsv' if method == 'hsv' else 'rgb'}"
            _figure(
                he, matlab, python,
                patient=patient, ppm=ppm, method_label=method_label,
                metrics=metrics, n_diff=n_diff, out_path=out_dir / f"{stem}.png",
            )
            rows.append(
                {
                    "patient": patient,
                    "ppm": ppm,
                    "method": "hsv" if method == "hsv" else "rgb",
                    "n_diff": n_diff,
                    "python_only": int((python & ~matlab).sum()),
                    "matlab_only": int((matlab & ~python).sum()),
                    "iou": metrics["iou"],
                    "coverage": 100.0 * float(matlab.mean()),
                }
            )
            print(f"{stem}: n_diff={n_diff} expected={expected} IoU={metrics['iou']:.6f}")
            if n_diff != expected:
                failures.append(f"{stem}: n_diff {n_diff} != recorded {expected}")
    summary = _write_summary(out_dir, rows)
    print(f"wrote {summary}")
    if failures:
        print("FAILED")
        for line in failures:
            print(line)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
