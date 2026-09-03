"""
Exact-parity tests for ``registration_method="matlab"`` vs MATLAB ``BDcreation_reg2.m``.

Three layers, so a regression points at one stage:

1. Primitive ports (``graythresh``, even-kernel ``imfilter``, ``imwarp`` edge
   rule) against values/probes captured from MATLAB.
2. Preprocessing: Python ``fixedSHG`` / ``HEmoving`` vs MATLAB's exact doubles
   in ``tests/matlab_parity/dumps/<case>/images.mat``.
3. Registration engine: ITK-v3 (1+1)-ES on MATLAB's own inputs reproduces
   ``tform_affine.txt``; and the full pipeline reproduces the golden TIFF
   pixel-for-pixel.

All MATLAB artefacts were generated offline (``tests/matlab_parity/*.m``);
nothing here needs MATLAB. Tests skip when ``itk`` or fixtures are missing.

This is a developer validation suite (bit-exactness against MATLAB dumps,
tens of seconds of ITK per case), not a CI gate. It is skipped unless::

    TMEQ_RUN_MATLAB_PARITY=1 pytest -q tests/test_shg_he_registration_matlab_parity.py

The CI-facing regression for the default path lives in
``tests/test_shg_he_registration.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from skimage import io

from pycurvelets._he_bdc_common import (
    adjust_rgb_mean_std,
    matlab_fspecial_gaussian,
    matlab_graythresh,
    matlab_imfilter,
    matlab_imwarp_bilinear,
    matlab_rgb2gray,
    prepare_registration_pair,
)
from pycurvelets._itk_v3_matlab_engine import (
    DEFAULT_SEED,
    has_itk,
    matlab_T_to_A,
    register_bdcreation_reg2_matlab,
)
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    _build_he_moving,
    shg_he_registration,
)

# Developer-only validation; off by default (CI). Enable locally with:
#   TMEQ_RUN_MATLAB_PARITY=1 pytest -q tests/test_shg_he_registration_matlab_parity.py
if os.environ.get("TMEQ_RUN_MATLAB_PARITY") != "1":
    pytest.skip(
        "MATLAB parity tests disabled (set TMEQ_RUN_MATLAB_PARITY=1 to enable)",
        allow_module_level=True,
    )

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_P02_ROOT = _FIXTURE_ROOT / "new_test_datasets_tests4-5-6-7"
_DUMPS = _TESTS_DIR / "matlab_parity" / "dumps"

# case_id, HE dir, HE/SHG filename, SHG dir, ppm, golden folder (under HE dir)
PARITY_CASES = (
    ("test1", _FIXTURE_ROOT / "HE", "patient_001.tif", _FIXTURE_ROOT / "SHG", 1.5, "HE_registered_test1"),
    ("test2", _FIXTURE_ROOT / "HE", "patient_001.tif", _FIXTURE_ROOT / "SHG", 2.0, "HE_registered_test2"),
    ("test3", _FIXTURE_ROOT / "HE", "patient_001.tif", _FIXTURE_ROOT / "SHG", 3.0, "HE_registered_test3"),
    ("test4", _P02_ROOT / "HE", "patient_02_roi2.tif", _P02_ROOT / "SHG", 2.6, "HE_registered_for_reg2_test4_roi2_ppm2p6"),
    ("test5", _P02_ROOT / "HE", "patient_02_roi4.tif", _P02_ROOT / "SHG", 1.5, "HE_registered_for_reg2_test5_roi4_ppm1p5"),
    ("test6", _P02_ROOT / "HE", "patient_02_roi4.tif", _P02_ROOT / "SHG", 2.6, "HE_registered_for_reg2_test6_roi4_ppm2p6"),
    ("test7", _P02_ROOT / "HE", "patient_02_roi5.tif", _P02_ROOT / "SHG", 2.6, "HE_registered_for_reg2_test7_roi5_ppm2p6"),
)
_IDS = [c[0] for c in PARITY_CASES]

requires_itk = pytest.mark.skipif(not has_itk(), reason="registration_method='matlab' needs the itk package")


def _skip_unless_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip("missing fixture(s):\n" + "\n".join(f"  {m}" for m in missing))


def _load_inputs(he_dir: Path, fname: str, shg_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    he = io.imread(str(he_dir / fname)).astype(np.float64) / 255.0
    shg = io.imread(str(shg_dir / fname)).astype(np.float64) / 255.0
    if shg.ndim == 3:
        shg = matlab_rgb2gray(shg)
    return he, shg


def _python_he_moving_and_fixed(he_dir: Path, fname: str, shg_dir: Path, ppm: float):
    he, shg = _load_inputs(he_dir, fname, shg_dir)
    he_scaled, fixed, pix = prepare_registration_pair(he, shg, float(ppm))
    he_adj = adjust_rgb_mean_std(he_scaled)
    he_moving, mode, _extras = _build_he_moving(he_scaled, he_adj, he_adj, pix, "hsv")
    assert mode == "hsv", f"hsv mask fell back to {mode!r}; MATLAB has no fallback"
    return he_moving, fixed


# ---------------------------------------------------------------------------
# 1. primitive ports
# ---------------------------------------------------------------------------


def test_graythresh_matches_matlab_on_uniform_ramp() -> None:
    # MATLAB: graythresh((0:255)/255) == 127/255 (single arg-max at bin 128).
    level = matlab_graythresh(np.arange(256) / 255.0)
    assert level == pytest.approx(127 / 255, abs=1e-12)


def test_graythresh_matches_matlab_dumped_saturation_threshold() -> None:
    """``channel2Min`` recorded by MATLAB for test1 (see intermediates dump)."""
    mat = _DUMPS / "test1" / "images.mat"
    _skip_unless_files(mat, _FIXTURE_ROOT / "HE" / "patient_001.tif", _FIXTURE_ROOT / "SHG" / "patient_001.tif")
    he, shg = _load_inputs(_FIXTURE_ROOT / "HE", "patient_001.tif", _FIXTURE_ROOT / "SHG")
    he_scaled, _fixed, _pix = prepare_registration_pair(he, shg, 1.5)
    from pycurvelets._he_bdc_common import matlab_rgb2hsv

    hsv = matlab_rgb2hsv(adjust_rgb_mean_std(he_scaled))
    # Value captured from MATLAB's dump_bdc_reg2 run (0.0902 == 23/255).
    assert matlab_graythresh(hsv[..., 1]) == pytest.approx(23 / 255, abs=1e-12)


def test_imfilter_even_kernel_uses_matlab_centre() -> None:
    """
    MATLAB centres an even kernel at floor((size+1)/2) = its top-left element,
    so correlation gives out(p) = sum_k h(k) I(p + k) with k in {0, 1}: an
    impulse at (2, 2) lands on rows/cols 1..2 (scipy's default would give 2..3).
    """
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    out = matlab_imfilter(img, matlab_fspecial_gaussian(2, 0.5))
    expected = np.zeros((5, 5))
    expected[1:3, 1:3] = 0.25
    np.testing.assert_allclose(out, expected, atol=1e-15)


def test_imwarp_bilinear_matches_matlab_interp2d_probe() -> None:
    """Dense probe of images.internal.interp2d + a real imwarp call (probe_interp2d.m)."""
    mat = _DUMPS / "interp2d_probe.mat"
    _skip_unless_files(mat)
    m = sio.loadmat(str(mat))
    img = m["img"]
    fill = float(m["fill"].item())
    # imwarp with T = [1 0 0; 0 1 0; 0.3 -0.2 1] => inverse map x_in = x_out-0.3, y_in = y_out+0.2
    A = np.array([[1.0, 0.0, -0.3], [0.0, 1.0, 0.2]])
    warped_py = matlab_imwarp_bilinear(img, (6, 7), A, fill_value=fill)
    np.testing.assert_allclose(warped_py, m["warped"], atol=1e-12)

    # Edge rule: samples in the half-pixel band [0.5, 1) are pure fill, and
    # inside samples never blend with fill.
    X, Y = m["X"], m["Y"]
    H, W = img.shape
    x0, y0 = X - 1.0, Y - 1.0  # 0-based
    inside = (x0 >= 0) & (x0 <= W - 1) & (y0 >= 0) & (y0 <= H - 1)
    ml = m["out_linear"]
    assert np.all(ml[~inside] == fill)
    assert np.all(ml[inside] <= 1.0)


# ---------------------------------------------------------------------------
# 2. preprocessing parity vs MATLAB doubles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,he_dir,fname,shg_dir,ppm,_golden", PARITY_CASES, ids=_IDS)
def test_preprocessing_reproduces_matlab_fixed_and_moving(case_id, he_dir, fname, shg_dir, ppm, _golden) -> None:
    mat = _DUMPS / case_id / "images.mat"
    _skip_unless_files(mat, he_dir / fname, shg_dir / fname)
    m = sio.loadmat(str(mat))
    ml_moving = np.asarray(m["HEmoving"], dtype=np.float64)
    ml_fixed = np.asarray(m["fixedSHG"], dtype=np.float64)

    he_moving, fixed = _python_he_moving_and_fixed(he_dir, fname, shg_dir, ppm)

    assert fixed.shape == ml_fixed.shape, f"[{case_id}] fixedSHG grid {fixed.shape} != MATLAB {ml_fixed.shape}"
    assert np.max(np.abs(fixed - ml_fixed)) < 1e-12, f"[{case_id}] fixedSHG differs from MATLAB imresize"
    n_diff = int(np.count_nonzero((he_moving > 0) != (ml_moving > 0)))
    assert n_diff == 0, f"[{case_id}] HEmoving mask differs from MATLAB in {n_diff} px"


# ---------------------------------------------------------------------------
# 3. engine + end-to-end
# ---------------------------------------------------------------------------


@requires_itk
@pytest.mark.parametrize("case_id,he_dir,fname,shg_dir,ppm,_golden", PARITY_CASES, ids=_IDS)
def test_itk_v3_engine_reproduces_matlab_tform_on_matlab_inputs(case_id, he_dir, fname, shg_dir, ppm, _golden) -> None:
    """Feed MATLAB's exact HEmoving/fixedSHG; the affine must match tform.T."""
    mat = _DUMPS / case_id / "images.mat"
    tform_txt = _DUMPS / case_id / "tform_affine.txt"
    _skip_unless_files(mat, tform_txt)
    m = sio.loadmat(str(mat))
    _fwd, dbg = register_bdcreation_reg2_matlab(
        np.asarray(m["HEmoving"], dtype=np.float64),
        np.asarray(m["fixedSHG"], dtype=np.float64),
        seed=DEFAULT_SEED,
    )
    A_py = np.asarray(dbg["aff_matlab_tform_A"])
    A_ml = matlab_T_to_A(np.loadtxt(tform_txt))
    max_diff = float(np.max(np.abs(A_py - A_ml)))
    assert max_diff < 1e-6, (
        f"[{case_id}] tform.A differs from MATLAB by {max_diff:.3e}\n"
        f"python:\n{A_py}\nmatlab:\n{A_ml}\n"
        f"stop={dbg.get('aff_stop_condition')} n_it={dbg.get('aff_n_iterations')}"
    )


@requires_itk
@pytest.mark.parametrize("case_id,he_dir,fname,shg_dir,ppm,golden", PARITY_CASES, ids=_IDS)
def test_matlab_method_reproduces_golden_tiff_exactly(case_id, he_dir, fname, shg_dir, ppm, golden) -> None:
    golden_path = he_dir / golden / fname
    _skip_unless_files(he_dir / fname, shg_dir / fname, golden_path)
    matlab_golden = io.imread(str(golden_path))
    if matlab_golden.dtype != np.uint8:
        matlab_golden = np.clip(matlab_golden, 0, 255).astype(np.uint8)

    params = SHGHERegistrationParameters(
        HEfilepath=str(he_dir),
        HEfilename=fname,
        pixelpermicron=ppm,
        SHGfilepath=str(shg_dir),
        areaThreshold=5000.0,
        registration_method="matlab",
        ecm_method="hsv",
    )
    py_float, debug = shg_he_registration(params, save_output=False, return_debug=True)
    py_uint8 = np.round(np.clip(py_float, 0, 1) * 255).astype(np.uint8)  # im2uint8

    assert py_uint8.shape == matlab_golden.shape
    diff = np.abs(py_uint8.astype(int) - matlab_golden.astype(int))
    n_bad = int(np.count_nonzero(diff))
    assert n_bad == 0, (
        f"[{case_id}] {n_bad} / {diff.size} pixels differ from MATLAB golden "
        f"(max {diff.max()} gray levels); backend={debug.get('registration_backend')}"
    )
