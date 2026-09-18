"""
Smoke tests for ``tumor_annotation_from_he``.

Pixel-exact MATLAB dumps live in ``test_tumor_annotation_matlab_parity.py``
(``TMEQ_RUN_MATLAB_PARITY=1``). This file only checks that the public API
runs on the clone fixtures and that HE2 matches the committed CA_Boundary
goldens (refreshed from ``BDcreationHE2.m``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._itk_v3_matlab_engine import has_itk
from pycurvelets.SHG_HE_registration import (
    SHGHERegistrationParameters,
    shg_he_registration,
)
from pycurvelets.tumor_annotation_from_HE import (
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_SHG_DIR = _FIXTURE_ROOT / "SHG"
_HE_RAW = _FIXTURE_ROOT / "HE" / "patient_001.tif"

ANNOTATION_CASES: tuple[tuple[str, float, str, str], ...] = (
    ("test1", 1.5, "HE_registered_test1", "BDcreationHE_test1results_mask for patient_001.tif.tif"),
    ("test2", 2.0, "HE_registered_test2", "BDcreationHE_test2results_mask for patient_001.tif.tif"),
    ("test3", 3.0, "HE_registered_test3", "BDcreationHE_test3results_mask for patient_001.tif.tif"),
)


def _require_annotation_fixtures(he_folder: str, golden_mask_name: str) -> tuple[Path, Path]:
    he_file = _FIXTURE_ROOT / "HE" / he_folder / "patient_001.tif"
    mask_path = _SHG_DIR / "CA_Boundary" / golden_mask_name
    missing = [p for p in (he_file, mask_path) if not p.is_file()]
    if missing:
        pytest.skip(
            "BDcreationHE2 regression needs fixture tree:\n"
            + "\n".join(f"  missing: {m}" for m in missing)
        )
    return he_file.parent, mask_path


@pytest.mark.parametrize(
    "case_id,pixelpermicron,he_registered_folder,golden_mask_filename",
    ANNOTATION_CASES,
    ids=[c[0] for c in ANNOTATION_CASES],
)
def test_tumor_annotation_he2_matches_committed_golden(
    case_id: str,
    pixelpermicron: float,
    he_registered_folder: str,
    golden_mask_filename: str,
) -> None:
    he_dir, golden_path = _require_annotation_fixtures(he_registered_folder, golden_mask_filename)
    golden = io.imread(str(golden_path)) > 0
    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_dir),
        HEfilename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        areaThreshold=5000.0,
        SHGfilepath=str(_SHG_DIR),
    )
    python_mask, debug = tumor_annotation_from_he(params, save_output=False, return_debug=True)
    assert python_mask.shape == golden.shape
    assert python_mask.dtype == np.bool_
    for key in ("he_adjusted", "BW_nuclei", "BW_collagen1", "epith_cell_BW", "mask_image1", "mask_temp", "BDmask"):
        assert key in debug, f"[{case_id}] missing debug key {key!r}"
    n_diff = int(np.count_nonzero(python_mask != golden))
    assert n_diff == 0, f"[{case_id}] HE2 vs golden differs in {n_diff} pixels"


def test_tumor_annotation_rgb_kmeans_runs() -> None:
    he_dir, _ = _require_annotation_fixtures(
        "HE_registered_test2",
        "BDcreationHE_test2results_mask for patient_001.tif.tif",
    )
    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_dir),
        HEfilename="patient_001.tif",
        pixelpermicron=2.0,
        areaThreshold=5000.0,
        SHGfilepath=str(_SHG_DIR),
        annotation_method="rgb_kmeans",
    )
    mask, debug = tumor_annotation_from_he(params, save_output=False, return_debug=True)
    assert mask.dtype == np.bool_
    assert mask.ndim == 2
    assert "labels" in debug
    assert str(debug["annotation_method"][0]) == "rgb_kmeans"
    assert debug["BDmask"].dtype == np.uint8


@pytest.mark.skipif(not has_itk(), reason="e2e registration needs itk")
def test_register_then_annotate_end_to_end_smoke() -> None:
    if not _HE_RAW.is_file() or not (_SHG_DIR / "patient_001.tif").is_file():
        pytest.skip("patient_001 HE/SHG fixtures missing")

    reg_params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_RAW.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=2.0,
        SHGfilepath=str(_SHG_DIR),
        areaThreshold=5000.0,
    )
    registered = shg_he_registration(reg_params, save_output=False, return_debug=False)
    assert registered.ndim == 3

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        he_name = "patient_001_pyreg.tif"
        out = Path(tmp) / he_name
        io.imsave(
            str(out),
            (np.clip(registered, 0, 1) * 255).astype(np.uint8),
            check_contrast=False,
        )
        ann_params = TumorAnnotationFromHEParameters(
            HEfilepath=str(tmp),
            HEfilename=he_name,
            pixelpermicron=2.0,
            areaThreshold=5000.0,
            SHGfilepath=str(_SHG_DIR),
        )
        mask = tumor_annotation_from_he(ann_params, save_output=False, return_debug=False)

    assert mask.shape == registered.shape[:2]
    assert mask.dtype == np.bool_
    assert int(mask.sum()) > 0
