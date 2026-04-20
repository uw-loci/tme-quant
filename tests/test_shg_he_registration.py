"""
Regression tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses golden registered H&E TIFFs from ``tests/test_for_shg_he_registration_BDcreation/``.

The test feeds the raw unregistered ``HE/patient_001.tif`` into the Python
registration pipeline and compares the uint8 output to the MATLAB golden in
``HE/HE_registered_testN/patient_001.tif``.  Pixel-exact match is not
achievable: the Python pipeline uses a grid search + Nelder-Mead refinement
on SimpleITK's Mattes MI metric, which converges to a slightly different
local minimum than MATLAB's ``imregtform`` (1+1)-ES.  Approximate MAE bounds
guard against regressions.  Tests skip if fixtures or SimpleITK are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

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

# uint8-scale regression bounds (empirically measured).
# Python and MATLAB use different optimizer implementations (SimpleITK vs
# MATLAB imregtform) so transforms differ slightly; bounds are set above
# observed values to catch actual regressions.
_MAX_MAE_UINT8: dict[str, float] = {
    "test1": 8.5,
    "test2": 8.0,
    "test3": 10.0,
}


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
    Register the raw unregistered HE to SHG with the Python conversion,
    then compare to the MATLAB golden registered HE using approximate bounds.

    Python input : ``HE/patient_001.tif`` (raw) + ``SHG/patient_001.tif``
    MATLAB golden: ``HE/<HE_registered_testN>/patient_001.tif`` (compare only)
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
    python_float = shg_he_registration(params, save_output=False, return_debug=False)
    python_uint8 = (np.clip(python_float, 0, 1) * 255).astype(np.uint8)

    assert python_uint8.shape == matlab_golden.shape, (
        f"[{case_id}] Shape mismatch: python {python_uint8.shape} vs golden {matlab_golden.shape}"
    )

    diff = np.abs(python_uint8.astype(np.float64) - matlab_golden.astype(np.float64))
    mae = float(np.mean(diff))

    max_mae = _MAX_MAE_UINT8[case_id]
    assert mae <= max_mae, (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.2f} exceeds bound {max_mae} "
        f"(uint8 scale, Python vs MATLAB golden)."
    )
