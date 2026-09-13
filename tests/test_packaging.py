"""Guard the wheel vs sdist file split.

Wheel
    ``pip install tme-quant`` / ``python -m build --wheel``
    Only ``src/`` (``pycurvelets``, ``napari_curvealign``) plus package data
    (``napari.yaml``, ``data/*.npz``).

sdist
    ``pip install tme-quant --no-binary tme-quant`` / ``python -m build --sdist``
    Developer source: tests (including MATLAB dumps and comparison figures),
    patient_001, and the patient_02 files that tests 4-9 read. Unused
    copies under ``new_test_datasets_tests4-5-6-7/`` stay out.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_WHEEL_REGISTRATION_MODULES = (
    "SHG_HE_registration.py",
    "_he_bdc_common.py",
    "_he_bdc_reg1.py",
    "_itk_v3_matlab_engine.py",
    "_matlab_imresize.py",
    "_registration_quality.py",
)

_SDIST_PATIENT02_FILES = (
    "HE/patient_02_roi2.tif",
    "HE/patient_02_roi4.tif",
    "HE/patient_02_roi5.tif",
    "SHG/patient_02_roi2.tif",
    "SHG/patient_02_roi4.tif",
    "SHG/patient_02_roi5.tif",
    "HE/HE_registered_for_reg2_test4_roi2_ppm2p6/patient_02_roi2.tif",
    "HE/HE_registered_for_reg2_test5_roi4_ppm1p5/patient_02_roi4.tif",
    "HE/HE_registered_for_reg2_test6_roi4_ppm2p6/patient_02_roi4.tif",
    "HE/HE_registered_for_reg2_test7_roi5_ppm2p6/patient_02_roi5.tif",
    "HE/HE_registered_for_reg1_test6b_ppm3/patient_02_roi4.tif",
    "HE/HE_registered_for_reg1_test9_ppm2p6/patient_02_roi4.tif",
    "patient_02_HE_original-roi2.tif",
    "patient_02_HE_original-roi4.tif",
    "patient_02_HE_original-roi5.tif",
)


def test_wheel_runtime_modules_live_under_src() -> None:
    for name in _WHEEL_REGISTRATION_MODULES:
        path = ROOT / "src" / "pycurvelets" / name
        assert path.is_file(), f"wheel must ship {path.relative_to(ROOT)}"
    assert (ROOT / "src" / "pycurvelets" / "data" / "matlab_srgb2lab_components.npz").is_file()


def test_gt_eval_is_not_in_the_wheel() -> None:
    assert not (ROOT / "src" / "pycurvelets" / "_registration_gt_eval.py").exists()
    assert (ROOT / "tests" / "_registration_gt_eval.py").is_file()


def test_sdist_manifest_is_the_dev_tree() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "graft tests" in manifest
    assert "prune tests/matlab_parity" not in manifest
    assert "prune tests/artifacts" not in manifest
    assert "prune tests/test_for_shg_he_registration_BDcreation/new_test_datasets_tests4-5-6-7" in manifest
    for rel in _SDIST_PATIENT02_FILES:
        assert rel in manifest, f"sdist must include patient_02 fixture {rel}"


def test_setuptools_wheel_is_src_only() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert 'where = ["src"]' in text
    assert "pycurvelets = [\"data/*.npz\"]" in text
