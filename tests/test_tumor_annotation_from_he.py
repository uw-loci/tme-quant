"""
Tests for ``tumor_annotation_from_he``.

Pixel-exact MATLAB dumps live in ``test_tumor_annotation_matlab_parity.py``
(``TMEQ_RUN_MATLAB_PARITY=1``). This file checks the public API against the
committed ``CA_Boundary`` TIFFs: HE2 for patient_001 and RGB k-means
``imwrite`` masks for patient_001.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from skimage import io

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
RGB_ANNOTATION_CASES: tuple[tuple[str, float, str, str], ...] = (
    ("test1", 1.5, "HE_registered_test1", "BDcreationHE_rgb_kmeans_test1_mask for patient_001.tif.tif"),
    ("test2", 2.0, "HE_registered_test2", "BDcreationHE_rgb_kmeans_test2_mask for patient_001.tif.tif"),
    ("test3", 3.0, "HE_registered_test3", "BDcreationHE_rgb_kmeans_test3_mask for patient_001.tif.tif"),
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
        SHGfilepath=str(_SHG_DIR),
    )
    python_mask, debug = tumor_annotation_from_he(params, save_output=False, return_debug=True)
    assert python_mask.shape == golden.shape
    assert python_mask.dtype == np.bool_
    for key in ("he_adjusted", "BW_nuclei", "BW_collagen1", "epith_cell_BW", "mask_image1", "mask_temp", "BDmask"):
        assert key in debug, f"[{case_id}] missing debug key {key!r}"
    n_diff = int(np.count_nonzero(python_mask != golden))
    assert n_diff == 0, f"[{case_id}] HE2 vs golden differs in {n_diff} pixels"


def _assert_saved_tiff_matches_golden(
    *,
    he_dir: Path,
    he_filename: str,
    pixelpermicron: float,
    annotation_method: str,
    golden_path: Path,
    max_diff: int = 0,
) -> None:
    """Run the public API with save_output and compare the uint8 TIFF."""
    if not (he_dir / he_filename).is_file() or not golden_path.is_file():
        pytest.skip(f"missing annotation fixture or golden: {he_dir / he_filename} / {golden_path}")
    golden = np.squeeze(io.imread(str(golden_path)))
    with tempfile.TemporaryDirectory() as tmp:
        params = TumorAnnotationFromHEParameters(
            HEfilepath=str(he_dir),
            HEfilename=he_filename,
            pixelpermicron=pixelpermicron,
            SHGfilepath=tmp,
            annotation_method=annotation_method,
        )
        mask = tumor_annotation_from_he(params, save_output=True, return_debug=False)
        written = Path(tmp) / "CA_Boundary" / f"mask for {he_filename.replace('HE', 'SHG')}.tif"
        saved = np.squeeze(io.imread(str(written)))
    assert saved.dtype == np.uint8
    assert set(np.unique(saved).tolist()) <= {0, 255}
    golden_arr = np.asarray(golden)
    n_diff = int(np.count_nonzero(saved != golden_arr))
    assert n_diff <= max_diff, f"{annotation_method} TIFF differs in {n_diff} pixels"
    np.testing.assert_array_equal(mask, saved > 0)


@pytest.mark.parametrize(
    "case_id,pixelpermicron,he_registered_folder,golden_mask_filename",
    RGB_ANNOTATION_CASES,
    ids=[c[0] for c in RGB_ANNOTATION_CASES],
)
def test_tumor_annotation_rgb_kmeans_matches_matlab_tiff(
    case_id: str,
    pixelpermicron: float,
    he_registered_folder: str,
    golden_mask_filename: str,
) -> None:
    """RGB wrapper and its saved 0/255 TIFF match the MATLAB imwrite golden.

    MATLAB ``kmeans`` is unseeded. These TIFFs were written with
    ``rng(28, 'twister')``, which is the API default.
    """
    he_dir = _FIXTURE_ROOT / "HE" / he_registered_folder
    _assert_saved_tiff_matches_golden(
        he_dir=he_dir,
        he_filename="patient_001.tif",
        pixelpermicron=pixelpermicron,
        annotation_method="rgb_kmeans",
        golden_path=_SHG_DIR / "CA_Boundary" / golden_mask_filename,
    )
    assert case_id in {"test1", "test2", "test3"}


def test_save_mask_tiff_round_trip_bool_and_uint8(tmp_path: Path) -> None:
    """A bool mask and a uint8 0/255 mask both come back as uint8 0/255."""
    from pycurvelets.tumor_annotation_from_HE import _save_image_uint8

    bool_mask = np.array([[False, True], [True, False]])
    u8_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    _save_image_uint8(tmp_path / "bool.tif", bool_mask)
    _save_image_uint8(tmp_path / "u8.tif", u8_mask)
    bool_back = io.imread(str(tmp_path / "bool.tif"))
    u8_back = io.imread(str(tmp_path / "u8.tif"))
    assert bool_back.dtype == np.uint8 and u8_back.dtype == np.uint8
    np.testing.assert_array_equal(bool_back, u8_mask)
    np.testing.assert_array_equal(u8_back, u8_mask)
    assert set(np.unique(bool_back).tolist()) == {0, 255}
    with pytest.raises(TypeError, match="bool or uint8"):
        _save_image_uint8(tmp_path / "float.tif", bool_mask.astype(np.float64))


def test_annotation_method_rejects_unknown_value() -> None:
    """The constructor accepts only 'hsv' and 'rgb_kmeans'."""
    with pytest.raises(ValueError, match="annotation_method"):
        TumorAnnotationFromHEParameters(
            HEfilepath=".",
            HEfilename="a.tif",
            pixelpermicron=2.0,
            SHGfilepath=".",
            annotation_method="lab",
        )


def test_register_then_annotate_end_to_end_smoke() -> None:
    if not _HE_RAW.is_file() or not (_SHG_DIR / "patient_001.tif").is_file():
        pytest.skip("patient_001 HE/SHG fixtures missing")

    reg_params = SHGHERegistrationParameters(
        HEfilepath=str(_HE_RAW.parent),
        HEfilename="patient_001.tif",
        pixelpermicron=2.0,
        SHGfilepath=str(_SHG_DIR),
    )
    registered = shg_he_registration(reg_params, save_output=False, return_debug=False)
    assert registered.ndim == 3

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
            SHGfilepath=str(_SHG_DIR),
        )
        mask = tumor_annotation_from_he(ann_params, save_output=False, return_debug=False)

    assert mask.shape == registered.shape[:2]
    assert mask.dtype == np.bool_
    assert int(mask.sum()) > 0
