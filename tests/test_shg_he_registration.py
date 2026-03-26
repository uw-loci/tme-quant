"""
Regression tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses golden registered H&E TIFFs from ``tests/test_for_shg_he_registration_BDcreation/``
(Yuming's fixtures: patient_001 HE/SHG pair, three ``pixelpermicron`` values).

SimpleITK Mattes MI differs from MathWorks ``imregtform``; MAE/RMSE/NCC bounds are
regression guards, not pixel equality. Tests skip if fixtures or SimpleITK are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._shg_he_registration_sitk import has_simpleitk
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"

_HE_INPUT = _FIXTURE_ROOT / "HE" / "patient_001.tif"
_SHG_INPUT = _FIXTURE_ROOT / "SHG" / "patient_001.tif"

# Matches BDCparameters_for_reg2_test{1,2,3}.mat: ppm 1.5, 2.0, 3.0; areaThreshold 5000.
REGRESSION_CASES: tuple[tuple[str, float, str], ...] = (
    ("test1", 1.5, "HE_registered_test1"),
    ("test2", 2.0, "HE_registered_test2"),
    ("test3", 3.0, "HE_registered_test3"),
)

# Per-case regression bounds (Python vs MATLAB golden, uv run on patient_001 fixtures).
# SimpleITK Mattes MI != MathWorks imregtform; NCC can be near zero or negative on some channels.
# Keys: case id -> (max_mae, max_rmse, min_ncc per channel). Increase max_* / decrease min_ncc only
# when intentionally changing registration; tighten to catch regressions.
_REGRESSION_BOUNDS: dict[str, tuple[float, float, tuple[float, float, float]]] = {
    # OnePlusOneEvolutionary + MATLAB-compat preprocess; uv run patient_001 fixtures:
    # ~ MAE 0.164 RMSE 0.283 NCC [0.005, 0.008, -0.010]
    "test1": (0.19, 0.33, (-0.02, -0.02, -0.02)),
    # ~ MAE 0.156 RMSE 0.274 NCC [0.022, 0.017, 0.0]
    "test2": (0.18, 0.32, (-0.01, -0.01, -0.01)),
    # ~ MAE 0.177 RMSE 0.295 NCC [0.003, 0.008, -0.009]
    "test3": (0.21, 0.36, (-0.02, -0.02, -0.02)),
}


def _golden_registered_path(case_folder: str) -> Path:
    return _FIXTURE_ROOT / "HE" / case_folder / "patient_001.tif"


def _load_tif_unit_float(path: Path) -> np.ndarray:
    """Load image as float64 in [0, 1] (uint8 TIFF → divide by 255)."""
    im = io.imread(str(path))
    if im.dtype == np.uint8:
        return im.astype(np.float64) / 255.0
    im = im.astype(np.float64)
    mx = float(im.max()) if im.size else 1.0
    if mx > 1.0:
        return np.clip(im / 255.0 if mx <= 255.0 else im / mx, 0.0, 1.0)
    return np.clip(im, 0.0, 1.0)


def _normalized_cross_correlation_channel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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
    Compare Python output to MATLAB ``BDcreation_reg2.m`` golden registered HE.

    Inputs: ``test_for_shg_he_registration_BDcreation/HE|SHG/patient_001.tif``
    Golden: ``HE/<HE_registered_testN>/patient_001.tif`` (same ppm as ``BDCparameters_for_reg2_testN.mat``).
    """
    golden_path = _require_registration_fixtures(he_registered_folder)

    matlab_registered = _load_tif_unit_float(golden_path)
    params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_INPUT.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        SHGfilepath=str(_SHG_INPUT.parent),
        areaThreshold=5000.0,
    )
    python_registered = shg_he_registration(params, save_output=False, return_debug=False)

    assert python_registered.shape == matlab_registered.shape, (
        f"Shape mismatch: python {python_registered.shape} vs golden {matlab_registered.shape}"
    )

    diff = python_registered.astype(np.float64) - matlab_registered.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ncc_rgb = [
        _normalized_cross_correlation_channel(python_registered[..., c], matlab_registered[..., c])
        for c in range(3)
    ]

    if os.environ.get("TMEQ_DEBUG_BDC_REGISTRATION") == "1":
        print(  # noqa: T201 — intentional debug aid for threshold tuning
            f"[{case_id}] ppm={pixelpermicron} MAE={mae:.6f} RMSE={rmse:.6f} NCC={ncc_rgb}"
        )

    max_mae, max_rmse, min_ncc = _REGRESSION_BOUNDS[case_id]

    assert mae <= max_mae, (
        f"[{case_id}] MAE {mae:.6f} exceeds bound {max_mae} (MATLAB golden vs Python)."
    )
    assert rmse <= max_rmse, (
        f"[{case_id}] RMSE {rmse:.6f} exceeds bound {max_rmse} (MATLAB golden vs Python)."
    )
    for c, ncc in enumerate(ncc_rgb):
        assert ncc >= min_ncc[c], (
            f"[{case_id}] NCC channel {c} = {ncc:.6f} < {min_ncc[c]} "
            f"(MATLAB golden vs Python)."
        )
