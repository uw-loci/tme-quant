"""
Ground-truth evaluation of SHG<->HE registration (Plan D).

The MATLAB goldens are reference *outputs*, not truth. The patient_02 cases
are synthetic: the HE was rotated/scaled by a known transform, so
``patient_02_HE_original-<roi>.tif`` is the truly aligned HE and the affine
recovered between input HE and that ROI (``tests/matlab_parity/dumps/
gt_affine_<case>.json``, SIFT+RANSAC, inlier RMS ~0.6 px) is ground truth.

What is asserted:

* the ITK v3 port reproduces MATLAB's *own* GT error on each case
  (5.5 / 68.7 / 9.4 / 7.5 px). This pins down that MATLAB fails test5 and
  that Python matches MATLAB rather than truth there.
* cross-ppm consistency: registering the same patient_001 pair at ppm
  1.5/2.0/3.0 should give the same full-resolution transform. MATLAB (and
  hence this port) is 4.4-7.8 px apart.

Measured values are in the module constants so a regression shows up as a
number, not a boolean. Tests skip without fixtures / itk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._itk_v3_matlab_engine import has_itk
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)

from _registration_gt_eval import (
    decompose_affine,
    gt_forward_to_working_grid,
    gt_report,
    mean_corner_displacement_px,
    registration_transform_to_grid,
    ssim_vs_reference_rgb,
)

# Developer-only validation; off by default. Enable locally with:
#   TMEQ_RUN_MATLAB_PARITY=1 pytest -q tests/test_shg_he_registration_gt.py
if os.environ.get("TMEQ_RUN_MATLAB_PARITY") != "1":
    pytest.skip(
        "GT evaluation tests disabled (set TMEQ_RUN_MATLAB_PARITY=1 to enable)",
        allow_module_level=True,
    )

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_P02_ROOT = _FIXTURE_ROOT / "new_test_datasets_tests4-5-6-7"
_GT_DIR = _TESTS_DIR / "matlab_parity" / "dumps"

GT_CASES = (
    ("test4", "patient_02_roi2.tif", 2.6, "roi2"),
    ("test5", "patient_02_roi4.tif", 1.5, "roi4"),
    ("test6", "patient_02_roi4.tif", 2.6, "roi4"),
    ("test7", "patient_02_roi5.tif", 2.6, "roi5"),
)
_IDS = [c[0] for c in GT_CASES]

MATLAB_GT_CORNER_DISP_PX = {"test4": 5.53, "test5": 68.67, "test6": 9.41, "test7": 7.50}
_MATLAB_MATCH_TOL_PX = 0.5

# Cross-ppm consistency ceiling on patient_001 (full-res px). Measured
# 4.41 / 7.77 / 7.64 for (1.5,2.0)/(1.5,3.0)/(2.0,3.0).
_XPPM_CEILING_PX = 12.0


def _skip_unless_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip("missing fixture(s):\n" + "\n".join(f"  {m}" for m in missing))


def _load_gt(case_id: str) -> dict:
    p = _GT_DIR / f"gt_affine_{case_id}.json"
    _skip_unless_files(p)
    gt = json.loads(p.read_text())
    if "forward_2x3_input_grid_0based" not in gt:
        pytest.skip(f"GT recovery failed for {case_id}: {gt.get('error')}")
    return gt


def _run(he_dir: Path, fname: str, shg_dir: Path, ppm: float):
    params = SHGHERegistrationParameters(
        HEfilepath=str(he_dir),
        HEfilename=fname,
        pixelpermicron=ppm,
        SHGfilepath=str(shg_dir),
        areaThreshold=5000.0,
    )
    img, debug = shg_he_registration(params, save_output=False, return_debug=True)
    return img, debug


def _gt_eval(case_id: str, fname: str, ppm: float, roi: str) -> dict:
    gt = _load_gt(case_id)
    ref_path = _P02_ROOT / f"patient_02_HE_original-{roi}.tif"
    _skip_unless_files(_P02_ROOT / "HE" / fname, _P02_ROOT / "SHG" / fname, ref_path)
    img, debug = _run(_P02_ROOT / "HE", fname, _P02_ROOT / "SHG", ppm)
    work = tuple(debug["fixed_shape"])
    G = gt_forward_to_working_grid(
        np.asarray(gt["forward_2x3_input_grid_0based"]),
        tuple(gt["src_shape"]), tuple(gt["dst_shape"]), work,
    )
    rep = gt_report(np.asarray(debug["forward_2x3"]), G, work)
    ref = io.imread(str(ref_path))[..., :3]
    u8 = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    rep["ssim_vs_gt"] = ssim_vs_reference_rgb(u8, ref)
    return rep


def test_decompose_affine_roundtrip() -> None:
    ang, s = 12.0, 1.25
    c, si = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    F = np.array([[s * c, -s * si, 3.0], [s * si, s * c, -4.0]])
    d = decompose_affine(F)
    assert d["angle_deg"] == pytest.approx(ang, abs=1e-9)
    assert d["scale_x"] == pytest.approx(s, abs=1e-9)
    assert d["scale_y"] == pytest.approx(s, abs=1e-9)
    assert d["shear"] == pytest.approx(0.0, abs=1e-9)
    assert (d["tx"], d["ty"]) == (3.0, -4.0)


def test_corner_displacement_is_translation_norm_for_pure_shift() -> None:
    F = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]])
    assert mean_corner_displacement_px(F, np.eye(3), (100, 200)) == pytest.approx(5.0)


def test_working_grid_mapping_roundtrips() -> None:
    F_full = np.array([[1.02, 0.05, 7.0], [-0.04, 0.98, -3.0]])
    he_in, shg_in, work = (640, 640), (512, 512), (394, 394)
    F_work = gt_forward_to_working_grid(F_full, he_in, shg_in, work)
    back = registration_transform_to_grid(F_work, work, he_in, shg_in)
    np.testing.assert_allclose(back, F_full, atol=1e-10)


@pytest.mark.skipif(not has_itk(), reason="ITK v3 (1+1)-ES path requires itk")
@pytest.mark.parametrize("case_id,fname,ppm,roi", GT_CASES, ids=_IDS)
def test_matlab_method_gt_error_matches_matlab(case_id, fname, ppm, roi) -> None:
    rep = _gt_eval(case_id, fname, ppm, roi)
    expected = MATLAB_GT_CORNER_DISP_PX[case_id]
    assert rep["corner_disp_px"] == pytest.approx(expected, abs=_MATLAB_MATCH_TOL_PX), (
        f"[{case_id}] vs GT: {rep['corner_disp_px']:.2f} px, MATLAB itself: {expected} px "
        f"(identity {rep['identity_corner_disp_px']:.1f} px, dAngle {rep['d_angle_deg']:+.2f} deg, "
        f"dScale {rep['d_scale']:+.3f}, SSIM vs GT {rep['ssim_vs_gt']:.3f})"
    )


def _full_res_transforms() -> tuple[dict[float, np.ndarray], tuple[int, int]]:
    he_p, shg_p = _FIXTURE_ROOT / "HE" / "patient_001.tif", _FIXTURE_ROOT / "SHG" / "patient_001.tif"
    _skip_unless_files(he_p, shg_p)
    he_shape = io.imread(str(he_p)).shape[:2]
    shg_shape = io.imread(str(shg_p)).shape[:2]
    out: dict[float, np.ndarray] = {}
    for ppm in (1.5, 2.0, 3.0):
        _img, debug = _run(_FIXTURE_ROOT / "HE", "patient_001.tif", _FIXTURE_ROOT / "SHG", ppm)
        out[ppm] = registration_transform_to_grid(
            np.asarray(debug["forward_2x3"]), tuple(debug["fixed_shape"]), he_shape, shg_shape
        )
    return out, he_shape


@pytest.mark.skipif(not has_itk(), reason="ITK v3 (1+1)-ES path requires itk")
def test_cross_ppm_consistency_patient001() -> None:
    full, he_shape = _full_res_transforms()
    pairs = {(a, b): mean_corner_displacement_px(full[a], full[b], he_shape) for a, b in ((1.5, 2.0), (1.5, 3.0), (2.0, 3.0))}
    detail = ", ".join(f"ppm {a} vs {b}: {d:.2f} px" for (a, b), d in pairs.items())
    worst = max(pairs.values())
    assert worst <= _XPPM_CEILING_PX, f"cross-ppm inconsistency too large: {detail}"
