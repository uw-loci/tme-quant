"""
Regression tests for ``tumor_annotation_from_he`` vs MATLAB ``BDcreationHE2.m``.

Uses MATLAB-registered HE images and golden tumor masks from
``tests/test_for_shg_he_registration_BDcreation/`` (Yuming's fixtures: three
``pixelpermicron`` values; parameters match ``BDCparameters_for_seg1_test*.mat``).

Registered HE inputs are MATLAB outputs from ``BDcreation_reg2`` so this module tests
tumor annotation in isolation. IoU / Dice / pixel accuracy are regression guards vs
committed golden masks (not pixel equality).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets.tumor_annotation_from_HE import (
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_SHG_DIR = _FIXTURE_ROOT / "SHG"

# Matches BDCparameters_for_seg1_test{1,2,3}.mat: ppm 1.5, 2.0, 3.0; HE from HE_registered_testN.
ANNOTATION_CASES: tuple[tuple[str, float, str, str], ...] = (
    ("test1", 1.5, "HE_registered_test1", "BDcreationHE_test1results_mask for patient_001.tif.tif"),
    ("test2", 2.0, "HE_registered_test2", "BDcreationHE_test2results_mask for patient_001.tif.tif"),
    ("test3", 3.0, "HE_registered_test3", "BDcreationHE_test3results_mask for patient_001.tif.tif"),
)

# Per-case minimum IoU, Dice, pixel accuracy (Python vs MATLAB golden mask, empirical run).
# Decrease these only when golden masks or algorithm intentionally change.
_REGRESSION_MIN_METRICS: dict[str, tuple[float, float, float]] = {
    # Observed ~ IoU 0.40 Dice 0.57 Acc 0.56
    "test1": (0.35, 0.50, 0.50),
    # Observed ~ IoU 0.38 Dice 0.55 Acc 0.52
    "test2": (0.32, 0.48, 0.47),
    # Observed ~ IoU 0.44 Dice 0.61 Acc 0.57
    "test3": (0.38, 0.54, 0.51),
}


def _registered_he_dir(folder: str) -> Path:
    return _FIXTURE_ROOT / "HE" / folder


def _golden_mask_path(name: str) -> Path:
    return _SHG_DIR / "CA_Boundary" / name


def _load_mask_as_bool(path: Path) -> np.ndarray:
    """Load binary mask as bool (uint8 TIFF: nonzero foreground)."""
    im = io.imread(str(path))
    if im.dtype == np.bool_:
        return im
    if im.dtype == np.uint8:
        return im > 127
    arr = im.astype(np.float64)
    return arr > 0.5


def _iou_dice_pixel_accuracy(
    pred: np.ndarray,
    golden: np.ndarray,
) -> tuple[float, float, float]:
    """Return (IoU, Dice, pixel accuracy) for aligned boolean masks."""
    p = np.asarray(pred, dtype=bool).ravel()
    g = np.asarray(golden, dtype=bool).ravel()
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    iou = float(inter / union) if union else 1.0
    denom = int(p.sum()) + int(g.sum())
    dice = float(2.0 * inter / denom) if denom else 1.0
    acc = float((p == g).sum() / p.size)
    return iou, dice, acc


def _require_annotation_fixtures(he_folder: str, golden_mask_name: str) -> tuple[Path, Path]:
    he_dir = _registered_he_dir(he_folder)
    he_file = he_dir / "patient_001.tif"
    mask_path = _golden_mask_path(golden_mask_name)
    missing = [p for p in (he_file, mask_path) if not p.is_file()]
    if missing:
        pytest.skip(
            "BDcreationHE2 regression needs fixture tree:\n"
            + "\n".join(f"  missing: {m}" for m in missing)
        )
    return he_dir, mask_path


@pytest.mark.parametrize(
    "case_id,pixelpermicron,he_registered_folder,golden_mask_filename",
    ANNOTATION_CASES,
    ids=[c[0] for c in ANNOTATION_CASES],
)
def test_tumor_annotation_from_he_matches_matlab_golden_mask_patient001(
    case_id: str,
    pixelpermicron: float,
    he_registered_folder: str,
    golden_mask_filename: str,
) -> None:
    """
    Compare Python mask to MATLAB ``BDcreationHE2.m`` golden under ``SHG/CA_Boundary/``.

    Input HE: MATLAB-registered ``HE/<HE_registered_testN>/patient_001.tif``.
    Golden: ``BDcreationHE_testNresults_mask for patient_001.tif.tif``.
    """
    he_dir, golden_path = _require_annotation_fixtures(he_registered_folder, golden_mask_filename)

    golden_mask = _load_mask_as_bool(golden_path)
    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_dir),
        HEfilename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        areaThreshold=5000.0,
        SHGfilepath=str(_SHG_DIR),
    )
    python_mask = tumor_annotation_from_he(params, save_output=False, return_debug=False)

    assert python_mask.shape == golden_mask.shape, (
        f"[{case_id}] shape mismatch: python {python_mask.shape} vs golden {golden_mask.shape}"
    )
    assert python_mask.dtype == np.bool_

    iou, dice, acc = _iou_dice_pixel_accuracy(python_mask, golden_mask)

    if os.environ.get("TMEQ_DEBUG_BDC_ANNOTATION") == "1":
        print(  # noqa: T201 — intentional debug aid for threshold tuning
            f"[{case_id}] ppm={pixelpermicron} IoU={iou:.6f} Dice={dice:.6f} Acc={acc:.6f}"
        )

    min_iou, min_dice, min_acc = _REGRESSION_MIN_METRICS[case_id]

    assert iou >= min_iou, (
        f"[{case_id}] IoU {iou:.6f} < {min_iou} (MATLAB golden vs Python)."
    )
    assert dice >= min_dice, (
        f"[{case_id}] Dice {dice:.6f} < {min_dice} (MATLAB golden vs Python)."
    )
    assert acc >= min_acc, (
        f"[{case_id}] pixel accuracy {acc:.6f} < {min_acc} (MATLAB golden vs Python)."
    )
