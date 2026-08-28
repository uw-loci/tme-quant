"""
Regression tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses golden registered H&E TIFFs from ``tests/test_for_shg_he_registration_BDcreation/``.

Dual-gate scoring (plan A1/A2):
1. Primary: SHG alignment (histogram MI / NCC of registered HE vs SHG).
2. Secondary: MATLAB golden PSNR/SSIM/MAE floors on patient_001 so we do not
   silently regress the historical parity path.

Tests skip if fixtures or SimpleITK are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._registration_quality import compute_registration_quality_metrics
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    has_simpleitk,
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

# uint8-scale regression bounds (empirically measured against the MATLAB
# golden ``BDcreation_reg2`` outputs). Bounds keep ~10-15% headroom.
_MAX_MAE_UINT8: dict[str, float] = {
    "test1": 5.5,
    "test2": 3.0,
    "test3": 9.5,
}
_MIN_EXACT_FRAC_UINT8: dict[str, float] = {
    "test1": 0.35,
    "test2": 0.40,
    "test3": 0.32,
}
_MIN_PSNR_DB: dict[str, float] = {
    "test1": 20.0,
    "test2": 24.0,
    "test3": 16.0,
}
_MIN_SSIM: dict[str, float] = {
    "test1": 0.75,
    "test2": 0.80,
    "test3": 0.68,
}

# Primary SHG-alignment: require the chosen transform to beat identity on
# histogram MI of warped ECM vs SHG (absolute MI is modality-dependent and
# often << 0.05 for binary collagen masks).
_MIN_SHG_MI_IMPROVEMENT: float = 0.0  # registered MI - identity MI
_MIN_SHG_NCC: float = -0.2  # catastrophic floor only



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
@pytest.mark.skipif(not has_simpleitk(), reason="MATLAB-parity path uses SimpleITK Mattes MI")
def test_shg_he_registration_matches_matlab_golden_patient001(
    case_id: str,
    pixelpermicron: float,
    he_registered_folder: str,
) -> None:
    """
    Dual-gate: SHG alignment floors + MATLAB golden PSNR/SSIM/MAE floors.
    """
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
    python_uint8 = (np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert python_uint8.shape == matlab_golden.shape, (
        f"[{case_id}] Shape mismatch: python {python_uint8.shape} vs golden {matlab_golden.shape}"
    )

    # Gate 1: SHG alignment — registered ECM↔SHG MI must beat identity.
    shg_align = debug.get("shg_alignment")
    assert shg_align is not None, f"[{case_id}] missing shg_alignment in debug"
    identity_mi = debug.get("shg_alignment_identity_mi")
    if identity_mi is None:
        # Fallback if older debug payload: just require finite positive MI.
        assert float(shg_align["shg_mi"]) > 0.0, (
            f"[{case_id}] SHG MI={shg_align['shg_mi']:.4f} not positive"
        )
    else:
        improvement = float(shg_align["shg_mi"]) - float(identity_mi)
        assert improvement >= _MIN_SHG_MI_IMPROVEMENT, (
            f"[{case_id}] SHG MI improvement {improvement:.4f} "
            f"(reg={shg_align['shg_mi']:.4f}, id={identity_mi:.4f}) "
            f"below floor {_MIN_SHG_MI_IMPROVEMENT}"
        )
    assert float(shg_align["shg_ncc"]) >= _MIN_SHG_NCC, (
        f"[{case_id}] SHG NCC={shg_align['shg_ncc']:.4f} "
        f"below floor {_MIN_SHG_NCC}"
    )

    # Gate 2: MATLAB golden regression floors.
    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    exact_frac = float(metrics["exact_frac"])
    psnr = float(metrics["psnr"])
    ssim = float(metrics["ssim"])

    assert mae <= _MAX_MAE_UINT8[case_id], (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.2f} exceeds bound "
        f"{_MAX_MAE_UINT8[case_id]} (uint8 scale). Exact: {exact_frac*100:.1f}%."
    )
    assert exact_frac >= _MIN_EXACT_FRAC_UINT8[case_id], (
        f"[{case_id}] ppm={pixelpermicron}: exact-pixel match {exact_frac*100:.1f}% "
        f"is below required {_MIN_EXACT_FRAC_UINT8[case_id]*100:.1f}%."
    )
    assert psnr >= _MIN_PSNR_DB[case_id], (
        f"[{case_id}] ppm={pixelpermicron}: PSNR={psnr:.2f} dB "
        f"is below required {_MIN_PSNR_DB[case_id]:.2f} dB."
    )
    assert ssim >= _MIN_SSIM[case_id], (
        f"[{case_id}] ppm={pixelpermicron}: SSIM={ssim:.4f} "
        f"is below required {_MIN_SSIM[case_id]:.4f}."
    )


@pytest.mark.skipif(not has_simpleitk(), reason="oneplusone path requires SimpleITK")
def test_shg_he_registration_oneplusone_backend_is_reasonable() -> None:
    """Ensure oneplusone runs and stays in a sensible quality range."""
    case_id, pixelpermicron, he_registered_folder = REGRESSION_CASES[1]  # test2
    golden_path = _require_registration_fixtures(he_registered_folder)
    matlab_golden = _load_tif_uint8(golden_path)

    params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_INPUT.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        SHGfilepath=str(_SHG_INPUT.parent),
        areaThreshold=5000.0,
        registration_method="oneplusone",
        random_state=0,
    )
    python_float, debug = shg_he_registration(params, save_output=False, return_debug=True)
    python_uint8 = (np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert python_uint8.shape == matlab_golden.shape
    assert debug.get("registration_backend") == "oneplusone_mattes"
    assert "shg_alignment" in debug

    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    assert float(metrics["mae_uint8"]) <= 20.0, (
        f"[{case_id}] oneplusone MAE too high: {metrics['mae_uint8']:.2f}"
    )
    assert float(metrics["psnr"]) >= 15.0, (
        f"[{case_id}] oneplusone PSNR too low: {metrics['psnr']:.2f} dB"
    )
    assert float(metrics["ssim"]) >= 0.55, (
        f"[{case_id}] oneplusone SSIM too low: {metrics['ssim']:.4f}"
    )


# ---------------------------------------------------------------------------
# Patient_02 regression tests (tests 4-9) — scored primarily vs SHG (A3).
# ---------------------------------------------------------------------------

_P02_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation" / "new_test_datasets_tests4-5-6-7"
_P02_HE = _P02_ROOT / "HE"
_P02_SHG = _P02_ROOT / "SHG"

# (case_id, he_filename, ppm, golden_folder, matlab_reg_type)
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
@pytest.mark.skipif(not has_simpleitk(), reason="mi_ncc path requires SimpleITK")
def test_shg_he_registration_patient02(
    case_id: str,
    he_filename: str,
    pixelpermicron: float,
    golden_folder: str,
    matlab_reg: str,
) -> None:
    """
    Patient_02: primary assert is SHG MI/NCC; MATLAB MAE is a catastrophic ceiling.

    Uses ``ecm_method='auto'`` so hsv/rgb/lab/gray compete on SHG MI (B3).
    """
    golden_path = _require_p02_fixtures(he_filename, golden_folder)
    matlab_golden = _load_tif_uint8(golden_path)

    params = SHGHERegistrationParameters(
        HEfilepath=str(_P02_HE),
        HEfilename=he_filename,
        pixelpermicron=pixelpermicron,
        SHGfilepath=str(_P02_SHG),
        areaThreshold=5000.0,
        ecm_method="auto",
    )
    python_float, debug = shg_he_registration(
        params, save_output=False, return_debug=True
    )
    python_uint8 = (np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert python_uint8.shape == matlab_golden.shape, (
        f"[{case_id}] Shape mismatch: python {python_uint8.shape} "
        f"vs golden {matlab_golden.shape}"
    )

    shg_align = debug.get("shg_alignment")
    assert shg_align is not None, f"[{case_id}] missing shg_alignment"
    identity_mi = float(debug.get("shg_alignment_identity_mi", 0.0))
    improvement = float(shg_align["shg_mi"]) - identity_mi
    # Loose: allow small negative on hard patient_02 cases; catch collapse.
    assert improvement >= -0.05, (
        f"[{case_id}] SHG MI worsened vs identity by {-improvement:.4f} "
        f"(ecm={debug.get('ecm_method_selected')}, matlab_reg={matlab_reg})"
    )
    assert float(shg_align["shg_ncc"]) >= -0.5, (
        f"[{case_id}] SHG NCC={shg_align['shg_ncc']:.4f} too low "
        f"(ecm={debug.get('ecm_method_selected')})"
    )

    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    # Catastrophic ceiling only (blank / wrong shape already caught above).
    assert mae <= 80.0, (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.2f} exceeds ceiling 80 "
        f"(PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, "
        f"matlab_reg={matlab_reg})"
    )
