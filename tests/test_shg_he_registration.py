"""
Parity tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses a **golden** registered H&E TIFF produced in MATLAB and committed under
``tests/test_images/``, plus the original HE/SHG pair from the repo ``utils/`` tree.

If the golden file or raw inputs are missing (e.g. minimal CI checkout), tests skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._shg_he_registration_sitk import has_simpleitk
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

# --- Golden case: same inputs as MATLAB BDcreation_reg2.m reference run ---
GOLDEN_CASE_FILENAME = "2B_D9_ROI1.tif"
GOLDEN_REGISTERED_NAME = "2B_D9_ROI1_registered_matlab.tif"
# Must match the MATLAB run used to generate the golden TIFF (see README in test_images).
GOLDEN_PIXEL_PER_MICRON = 2.0

# Empirical bounds vs committed MATLAB golden (same HE/SHG inputs, ``pixelpermicron=2.0``).
# ITK/SimpleITK Mattes MI differs from MathWorks ``imregtform``; values are regression guards, not equality.
# Re-tune if preprocessing or registration changes; see ``test_images/README_registration_2B_D9_ROI1.md``.
GOLDEN_MAX_MAE = 0.045
GOLDEN_MAX_RMSE = 0.12
GOLDEN_MIN_NCC_PER_CHANNEL = 0.28

_TESTS_DIR = Path(__file__).resolve().parent
_TME_QUANT_ROOT = _TESTS_DIR.parent
_REPO_ROOT = _TME_QUANT_ROOT.parent

_GOLDEN_TIF = _TESTS_DIR / "test_images" / GOLDEN_REGISTERED_NAME
_HE_TIF = _REPO_ROOT / "utils" / "TestimagesCA6.0_20240722" / "HE" / GOLDEN_CASE_FILENAME
_SHG_TIF = _REPO_ROOT / "utils" / "TestimagesCA6.0_20240722" / "SHG" / GOLDEN_CASE_FILENAME


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


def _require_golden_fixtures() -> None:
    missing = [p for p in (_GOLDEN_TIF, _HE_TIF, _SHG_TIF) if not p.is_file()]
    if missing:
        pytest.skip(
            "MATLAB golden parity test needs committed golden + utils images:\n"
            + "\n".join(f"  missing: {m}" for m in missing)
        )


@pytest.mark.skipif(not has_simpleitk(), reason="MATLAB-parity path uses SimpleITK Mattes MI")
def test_shg_he_registration_matches_matlab_golden_2b_d9_roi1() -> None:
    """
    Compare Python output to the reference from ``curvelets/.../BDcreation_reg2.m``.

    Golden TIFF: ``tests/test_images/2B_D9_ROI1_registered_matlab.tif``
    Inputs: ``utils/TestimagesCA6.0_20240722/HE|SHG/2B_D9_ROI1.tif``
    """
    _require_golden_fixtures()

    matlab_registered = _load_tif_unit_float(_GOLDEN_TIF)
    params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_TIF.parent),
        HEfilename=GOLDEN_CASE_FILENAME,
        pixelpermicron=GOLDEN_PIXEL_PER_MICRON,
        SHGfilepath=str(_SHG_TIF.parent),
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

    assert mae <= GOLDEN_MAX_MAE, (
        f"MAE {mae:.6f} exceeds bound {GOLDEN_MAX_MAE} (MATLAB golden vs Python)."
    )
    assert rmse <= GOLDEN_MAX_RMSE, (
        f"RMSE {rmse:.6f} exceeds bound {GOLDEN_MAX_RMSE} (MATLAB golden vs Python)."
    )
    for c, ncc in enumerate(ncc_rgb):
        assert ncc >= GOLDEN_MIN_NCC_PER_CHANNEL, (
            f"NCC channel {c} = {ncc:.6f} < {GOLDEN_MIN_NCC_PER_CHANNEL} "
            f"(MATLAB golden vs Python)."
        )
