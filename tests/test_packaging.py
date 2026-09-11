"""Guard the wheel vs sdist vs git-dev file split for SHG–HE registration.

Wheel
    ``pip install tme-quant`` / ``python -m build --wheel``
    Only ``src/`` (``pycurvelets``, ``napari_curvealign``) plus package data
    (``napari.yaml``, ``data/*.npz``).

sdist
    ``pip install tme-quant --no-binary tme-quant`` / ``python -m build --sdist``
    Source + CI-runnable tests and the small patient_001 fixtures.
    ``MANIFEST.in`` prunes the MATLAB dump harness and comparison PNGs.

git-dev
    A clone of this repo. Adds ``tests/matlab_parity`` and
    ``tests/artifacts/bdc_regression_viz``. The optional patient_02 tree is
    local-only (gitignored).
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


def test_wheel_runtime_modules_live_under_src() -> None:
    for name in _WHEEL_REGISTRATION_MODULES:
        path = ROOT / "src" / "pycurvelets" / name
        assert path.is_file(), f"wheel must ship {path.relative_to(ROOT)}"
    assert (ROOT / "src" / "pycurvelets" / "data" / "matlab_srgb2lab_components.npz").is_file()


def test_gt_eval_is_dev_only_not_in_the_wheel() -> None:
    assert not (ROOT / "src" / "pycurvelets" / "_registration_gt_eval.py").exists()
    assert (ROOT / "tests" / "_registration_gt_eval.py").is_file()


def test_sdist_manifest_includes_ci_tests_and_prunes_dev_harness() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "graft tests" in manifest
    assert "prune tests/matlab_parity" in manifest
    assert "prune tests/artifacts" in manifest
    assert "prune tests/test_for_shg_he_registration_BDcreation/new_test_datasets_tests4-5-6-7" in manifest


def test_setuptools_wheel_is_src_only() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert 'where = ["src"]' in text
    assert "pycurvelets = [\"data/*.npz\"]" in text
