"""
Regression tests for ``tumor_annotation_from_he`` vs MATLAB ``BDcreationHE2.m``.

Uses MATLAB-registered HE images and golden tumor masks from
``tests/test_for_shg_he_registration_BDcreation/``.

Also covers boundary metrics (F6), intermediate-mask debug (F1), and a
lightweight end-to-end register→annotate smoke (F5).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._registration_quality import compute_mask_boundary_metrics
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    has_simpleitk,
    shg_he_registration,
)
from pycurvelets.tumor_annotation_from_HE import (
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_SHG_DIR = _FIXTURE_ROOT / "SHG"
_HE_RAW = _FIXTURE_ROOT / "HE" / "patient_001.tif"

# Matches BDCparameters_for_seg1_test{1,2,3}.mat: ppm 1.5, 2.0, 3.0; HE from HE_registered_testN.
ANNOTATION_CASES: tuple[tuple[str, float, str, str], ...] = (
    ("test1", 1.5, "HE_registered_test1", "BDcreationHE_test1results_mask for patient_001.tif.tif"),
    ("test2", 2.0, "HE_registered_test2", "BDcreationHE_test2results_mask for patient_001.tif.tif"),
    ("test3", 3.0, "HE_registered_test3", "BDcreationHE_test3results_mask for patient_001.tif.tif"),
)

_REGRESSION_MIN_METRICS: dict[str, tuple[float, float, float]] = {
    "test1": (0.35, 0.50, 0.50),
    "test2": (0.32, 0.48, 0.47),
    "test3": (0.38, 0.54, 0.51),
}

# Boundary F1 floors (F6). Boundaries are thin; observed values on patient_001
# goldens are low (~0.01–0.11) while region IoU stays ~0.38–0.44. Floors catch
# total collapse (empty mask) rather than require MATLAB edge identity.
_MIN_BOUNDARY_F1: dict[str, float] = {
    "test1": 0.05,
    "test2": 0.01,
    "test3": 0.01,
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
    python_mask, debug = tumor_annotation_from_he(
        params, save_output=False, return_debug=True
    )

    assert python_mask.shape == golden_mask.shape, (
        f"[{case_id}] shape mismatch: python {python_mask.shape} vs golden {golden_mask.shape}"
    )
    assert python_mask.dtype == np.bool_

    # F1: intermediate-mask dump must expose the HE2 pipeline stages.
    for key in (
        "he_adjusted",
        "nuclei_opened",
        "bw_collagen1",
        "epith_cell_bw",
        "mask_image1",
        "mask_temp",
        "bd_mask",
    ):
        assert key in debug, f"[{case_id}] missing intermediate debug key {key!r}"

    metrics = compute_mask_boundary_metrics(python_mask, golden_mask)
    iou = metrics["iou"]
    dice = metrics["dice"]
    acc = metrics["pixel_accuracy"]
    boundary_f1 = metrics["boundary_f1"]

    if os.environ.get("TMEQ_DEBUG_BDC_ANNOTATION") == "1":
        print(  # noqa: T201
            f"[{case_id}] ppm={pixelpermicron} IoU={iou:.6f} Dice={dice:.6f} "
            f"Acc={acc:.6f} BoundF1={boundary_f1:.6f} "
            f"Hausdorff={metrics['hausdorff_px']:.2f}"
        )

    min_iou, min_dice, min_acc = _REGRESSION_MIN_METRICS[case_id]
    assert iou >= min_iou, f"[{case_id}] IoU {iou:.6f} < {min_iou}"
    assert dice >= min_dice, f"[{case_id}] Dice {dice:.6f} < {min_dice}"
    assert acc >= min_acc, f"[{case_id}] pixel accuracy {acc:.6f} < {min_acc}"
    assert boundary_f1 >= _MIN_BOUNDARY_F1[case_id], (
        f"[{case_id}] boundary F1 {boundary_f1:.6f} < {_MIN_BOUNDARY_F1[case_id]}"
    )


def test_tumor_annotation_rgb_kmeans_runs() -> None:
    """F4: BDcreationHE-style rgb_kmeans path produces a boolean mask."""
    he_dir, _ = _require_annotation_fixtures(
        "HE_registered_test2",
        "BDcreationHE_test2results_mask for patient_001.tif.tif",
    )
    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_dir),
        HEfilename="patient_001.tif",
        pixelpermicron=2.0,
        areaThreshold=5000.0,
        SHGfilepath=str(_SHG_DIR),
        annotation_method="rgb_kmeans",
    )
    mask, debug = tumor_annotation_from_he(params, save_output=False, return_debug=True)
    assert mask.dtype == np.bool_
    assert mask.ndim == 2
    assert "labels" in debug
    assert str(debug["annotation_method"][0]) == "rgb_kmeans"


@pytest.mark.skipif(not has_simpleitk(), reason="e2e registration needs SimpleITK")
def test_register_then_annotate_end_to_end_smoke() -> None:
    """
    F5: Python register → Python annotate on patient_001 ppm=2.0.

    Does not assert MATLAB mask identity (registration basins differ); checks
    that the pipeline produces a non-empty mask of the SHG shape.
    """
    if not _HE_RAW.is_file() or not (_SHG_DIR / "patient_001.tif").is_file():
        pytest.skip("patient_001 HE/SHG fixtures missing")

    reg_params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_RAW.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=2.0,
        SHGfilepath=str(_SHG_DIR),
        areaThreshold=5000.0,
    )
    registered = shg_he_registration(reg_params, save_output=False, return_debug=False)
    assert registered.ndim == 3

    # Write to a temp-like path under fixtures artifacts is avoided; annotate
    # in-memory by saving a temp file next to the fixture root.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        he_name = "patient_001_pyreg.tif"
        out = Path(tmp) / he_name
        io.imsave(
            str(out),
            (np.clip(registered, 0, 1) * 255).astype(np.uint8),
            check_contrast=False,
        )
        ann_params = TumorAnnotationFromHEParameters(
            HEfilepath=str(tmp),
            HEfilename=he_name,
            pixelpermicron=2.0,
            areaThreshold=5000.0,
            SHGfilepath=str(_SHG_DIR),
        )
        mask = tumor_annotation_from_he(ann_params, save_output=False, return_debug=False)

    assert mask.shape == registered.shape[:2]
    assert mask.dtype == np.bool_
    assert int(mask.sum()) > 0
