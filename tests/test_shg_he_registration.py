"""
Regression tests for ``SHG_HE_registration`` vs MATLAB ``BDcreation_reg2.m``.

Uses golden registered H&E TIFFs from ``tests/test_for_shg_he_registration_BDcreation/``.

The test feeds the raw unregistered ``HE/patient_001.tif`` into the Python
registration pipeline and compares the uint8 output to the MATLAB golden in
``HE/HE_registered_testN/patient_001.tif``.  Pixel-exact match is not
achievable: the Python pipeline uses SimpleITK's Mattes MI for basin finding
plus a bounded NCC trust-region refinement (with a raw-NCC accept gate),
which converges to a slightly different local minimum than MATLAB's
``imregtform`` (1+1)-ES.  Approximate MAE + exact-match bounds guard against
regressions.  Tests skip if fixtures or SimpleITK are missing.
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
# golden ``BDcreation_reg2`` outputs). Python and MATLAB use different
# optimizer implementations (SimpleITK Mattes MI + bounded NCC TRF refine
# with a raw-NCC accept gate, vs MATLAB's ``imregtform`` (1+1)-ES) so the
# transforms differ slightly; bounds are set ~10-15% above observed values
# to catch true regressions while absorbing minor cross-platform float drift.
#
# Observed (mi_ncc, raw-NCC accept gate):
#   test1 ppm=1.5  MAE=4.78  Exact=40.1%
#   test2 ppm=2.0  MAE=2.39  Exact=44.1%
#   test3 ppm=3.0  MAE=8.09  Exact=38.9%
_MAX_MAE_UINT8: dict[str, float] = {
    "test1": 5.5,
    "test2": 3.0,
    "test3": 9.5,
}

# Lower bound on the fraction of pixels that must match the MATLAB golden
# byte-for-byte. Pixel-exact match is impossible across optimizer + bilinear
# interpolation differences, but a sudden drop here is the most sensitive
# signal that the registration transform has drifted.
_MIN_EXACT_FRAC_UINT8: dict[str, float] = {
    "test1": 0.35,
    "test2": 0.40,
    "test3": 0.32,
}

# PSNR/SSIM are against the MATLAB golden registered RGB image. These are
# intentionally lenient floor checks: optimizer stochasticity and cross-platform
# float differences can move them slightly, but sharp drops flag real drift.
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

    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    mae = float(metrics["mae_uint8"])
    exact_frac = float(metrics["exact_frac"])
    psnr = float(metrics["psnr"])
    ssim = float(metrics["ssim"])

    max_mae = _MAX_MAE_UINT8[case_id]
    min_exact = _MIN_EXACT_FRAC_UINT8[case_id]
    min_psnr = _MIN_PSNR_DB[case_id]
    min_ssim = _MIN_SSIM[case_id]
    assert mae <= max_mae, (
        f"[{case_id}] ppm={pixelpermicron}: MAE={mae:.2f} exceeds bound {max_mae} "
        f"(uint8 scale, Python vs MATLAB golden). Exact match: {exact_frac*100:.1f}%."
    )
    assert exact_frac >= min_exact, (
        f"[{case_id}] ppm={pixelpermicron}: exact-pixel match {exact_frac*100:.1f}% "
        f"is below required {min_exact*100:.1f}% (Python vs MATLAB golden, MAE={mae:.2f})."
    )
    assert psnr >= min_psnr, (
        f"[{case_id}] ppm={pixelpermicron}: PSNR={psnr:.2f} dB "
        f"is below required {min_psnr:.2f} dB."
    )
    assert ssim >= min_ssim, (
        f"[{case_id}] ppm={pixelpermicron}: SSIM={ssim:.4f} "
        f"is below required {min_ssim:.4f}."
    )


@pytest.mark.skipif(not has_simpleitk(), reason="oneplusone path requires SimpleITK")
def test_shg_he_registration_oneplusone_backend_is_reasonable() -> None:
    """
    Ensure the oneplusone backend runs end-to-end and stays in a sensible
    quality range against the MATLAB golden output.
    """
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

    metrics = compute_registration_quality_metrics(python_uint8, matlab_golden)
    # Loose guardrails: we only require "reasonable quality", not parity with
    # the default mi_ncc backend.
    assert float(metrics["mae_uint8"]) <= 20.0, (
        f"[{case_id}] oneplusone MAE too high: {metrics['mae_uint8']:.2f}"
    )
    assert float(metrics["psnr"]) >= 15.0, (
        f"[{case_id}] oneplusone PSNR too low: {metrics['psnr']:.2f} dB"
    )
    assert float(metrics["ssim"]) >= 0.55, (
        f"[{case_id}] oneplusone SSIM too low: {metrics['ssim']:.4f}"
    )
