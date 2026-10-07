"""Guard the wheel vs sdist file split.

Wheel
    ``pip install tme-quant`` / ``python -m build --wheel``
    Only ``src/`` (``pycurvelets``, ``napari_curvealign``) plus package data
    (``napari.yaml``, ``data/*.npz``).

sdist
    ``pip install tme-quant --no-binary tme-quant`` / ``python -m build --sdist``
    Source needed to rebuild the wheel, plus the lightweight pytest suite.
    Tests 1-9 fixtures, MATLAB dumps, GT TIFFs, and comparison figures stay
    in git; they are not release artifacts.
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

_SDIST_PRUNED = (
    "tests/test_for_shg_he_registration_BDcreation",
    "tests/matlab_parity",
    "tests/artifacts",
)


def test_wheel_runtime_modules_live_under_src() -> None:
    for name in _WHEEL_REGISTRATION_MODULES:
        path = ROOT / "src" / "pycurvelets" / name
        assert path.is_file(), f"wheel must ship {path.relative_to(ROOT)}"
    assert (ROOT / "src" / "pycurvelets" / "data" / "matlab_srgb2lab_components.npz").is_file()


def test_gt_eval_is_not_in_the_wheel() -> None:
    assert not (ROOT / "src" / "pycurvelets" / "_registration_gt_eval.py").exists()
    assert (ROOT / "tests" / "_registration_gt_eval.py").is_file()


def test_sdist_excludes_registration_verification_assets() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "graft tests" in manifest
    for rel in _SDIST_PRUNED:
        assert f"prune {rel}" in manifest, f"sdist must prune {rel}"
    assert "patient_02" not in manifest
    assert "include tests/test_for_shg_he_registration_BDcreation" not in manifest


def test_setuptools_wheel_is_src_only() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert 'where = ["src"]' in text
    assert "pycurvelets = [\"data/*.npz\"]" in text
