"""
Regression tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses golden registered H&E TIFFs from ``tests/test_for_shg_he_registration_BDcreation/``.

The only registrar is the ITK v3 (1+1)-ES port. On the reference machine it is
pixel-exact (asserted in ``test_shg_he_registration_matlab_parity.py``). Here
we keep a last-ulp MAE margin so another ITK build does not fail CI.

Tests skip if fixtures or ``itk`` are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._itk_v3_matlab_engine import has_itk
from pycurvelets._registration_quality import compute_registration_quality_metrics
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"

_HE_INPUT = _FIXTURE_ROOT / "HE" / "patient_001.tif"
_SHG_INPUT = _FIXTURE_ROOT / "SHG" / "patient_001.tif"

# (case_id, pixelpermicron, MATLAB golden subfolder under HE/)
REGRESSION_CASES: tuple[tuple[str, float, str], ...] = (
    ("test1", 1.5, "HE_registered_test1"),
    ("test2", 2.0, "HE_registered_test2"),
    ("test3", 3.0, "HE_registered_test3"),
)

# Last-ulp margin for a different platform/ITK build. Anything larger means
# the ES trajectory diverged from MATLAB.
_MAX_MAE_UINT8_MATLAB: float = 1.0


def _golden_registered_path(case_folder: str) -> Path:
    return _FIXTURE_ROOT / "HE" / case_folder / "patient_001.tif"


def _load_tif_uint8(path: Path) -> np.ndarray:
    im = io.imread(str(path))
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def _require_registration_fixtures(case_folder: str) -> Path:
    golden = _golden_registered_path(case_folder)
    missing = [p for p in (_HE_INPUT, _SHG_INPUT, golden) if not p.is_file()]
    if missing:
        pytest.skip(
            "BDcreation_reg2 regression needs fixture tree:\n"
            + "\n".join(f"  missing: {m}" for m in missing)
        )
    return golden


@pytest.mark.parametrize(
    "case_id,pixelpermicron,he_registered_folder",
    REGRESSION_CASES,
    ids=[c[0] for c in REGRESSION_CASES],
)
@pytest.mark.skipif(not has_itk(), reason="ITK v3 (1+1)-ES path requires itk")
def test_shg_he_registration_matches_matlab_golden_patient001(
    case_id: str,
    pixelpermicron: float,
    he_registered_folder: str,
) -> None:
    """Default path must reproduce the MATLAB ``BDcreation_reg2`` golden TIFF."""
    golden_path = _require_registration_fixtures(he_registered_folder)
    matlab_golden = _load_tif_uint8(golden_path)

    params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_INPUT.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        SHGfilepath=str(_SHG_INPUT.parent),
        areaThreshold=5000.0,
    )
    python_float, debug = shg_he_registration(
        params, save_output=False, return_debug=True
    )
    python_uint8 = np.round(np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert debug.get("registration_backend") == "itk_v3_matlab_parity", (
        f"[{case_id}] did not dispatch to the ITK v3 engine: "
        f"{debug.get('registration_backend')}"
    )
    assert python_uint8.shape == matlab_golden.shape, (
        f"[{case_id}] Shape mismatch: python {python_uint8.shape} vs golden {matlab_golden.shape}"
    )
    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    assert mae <= _MAX_MAE_UINT8_MATLAB, (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.3f} "
        f"exceeds {_MAX_MAE_UINT8_MATLAB} (exact={metrics['exact_frac']*100:.1f}%, "
        f"SSIM={metrics['ssim']:.4f}). Final tform.A: {debug.get('aff_matlab_tform_A')}"
    )


_P02_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation" / "new_test_datasets_tests4-5-6-7"
_P02_HE = _P02_ROOT / "HE"
_P02_SHG = _P02_ROOT / "SHG"

PATIENT02_CASES: tuple[tuple[str, str, float, str, str], ...] = (
    ("test4", "patient_02_roi2.tif", 2.6, "HE_registered_for_reg2_test4_roi2_ppm2p6", "reg2"),
    ("test5", "patient_02_roi4.tif", 1.5, "HE_registered_for_reg2_test5_roi4_ppm1p5", "reg2"),
    ("test6", "patient_02_roi4.tif", 2.6, "HE_registered_for_reg2_test6_roi4_ppm2p6", "reg2"),
    ("test7", "patient_02_roi5.tif", 2.6, "HE_registered_for_reg2_test7_roi5_ppm2p6", "reg2"),
    ("test8", "patient_02_roi4.tif", 3.0, "HE_registered_for_reg1_test6b_ppm3", "reg1"),
    ("test9", "patient_02_roi4.tif", 2.6, "HE_registered_for_reg1_test9_ppm2p6", "reg1"),
)


def _require_p02_fixtures(he_filename: str, golden_folder: str) -> Path:
    he_in = _P02_HE / he_filename
    shg_in = _P02_SHG / he_filename
    golden = _P02_HE / golden_folder / he_filename
    missing = [p for p in (he_in, shg_in, golden) if not p.is_file()]
    if missing:
        pytest.skip(
            "Patient_02 regression needs fixture files:\n"
            + "\n".join(f"  missing: {m}" for m in missing)
        )
    return golden


@pytest.mark.parametrize(
    "case_id,he_filename,pixelpermicron,golden_folder,matlab_reg",
    PATIENT02_CASES,
    ids=[c[0] for c in PATIENT02_CASES],
)
@pytest.mark.skipif(not has_itk(), reason="ITK v3 (1+1)-ES path requires itk")
def test_shg_he_registration_patient02(
    case_id: str,
    he_filename: str,
    pixelpermicron: float,
    golden_folder: str,
    matlab_reg: str,
) -> None:
    """
    Patient_02: tests 4-7 are ``BDcreation_reg2``; tests 8-9 are
    ``BDcreation_reg``. Same last-ulp MAE bound as patient_001.
    """
    golden_path = _require_p02_fixtures(he_filename, golden_folder)
    matlab_golden = _load_tif_uint8(golden_path)

    params = SHGHERegistrationParameters(
        HEfilepath=str(_P02_HE),
        HEfilename=he_filename,
        pixelpermicron=pixelpermicron,
        SHGfilepath=str(_P02_SHG),
        areaThreshold=5000.0,
        pipeline=matlab_reg,
    )
    python_float, debug = shg_he_registration(
        params, save_output=False, return_debug=True
    )
    python_uint8 = np.round(np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert debug.get("registration_backend") == "itk_v3_matlab_parity", (
        f"[{case_id}] did not dispatch to the ITK v3 engine: "
        f"{debug.get('registration_backend')}"
    )
    assert python_uint8.shape == matlab_golden.shape, (
        f"[{case_id}] Shape mismatch: python {python_uint8.shape} "
        f"vs golden {matlab_golden.shape}"
    )

    shg_align = debug.get("shg_alignment")
    assert shg_align is not None, f"[{case_id}] missing shg_alignment"
    identity_mi = float(debug.get("shg_alignment_identity_mi", 0.0))
    improvement = float(shg_align["shg_mi"]) - identity_mi
    assert improvement >= -0.05, (
        f"[{case_id}] SHG MI worsened vs identity by {-improvement:.4f} "
        f"(pipeline={matlab_reg})"
    )

    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    assert mae <= _MAX_MAE_UINT8_MATLAB, (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.2f} exceeds {_MAX_MAE_UINT8_MATLAB} "
        f"(PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, pipeline={matlab_reg})"
    )
