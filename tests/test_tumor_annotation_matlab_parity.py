"""Exact-parity tests for BDcreationHE2 / BDcreationHE vs MATLAB dumps.

Layers (a regression points at one stage):

1. Primitive ports (``histeq``, ``fspecial('disk')``, ``strel``, ``padarray``,
   ``im2bw``) against ``dumps/annotation_primitives.mat``.
2. Preprocessing intermediates vs ``dumps/he2_<case>/images.mat`` and
   ``dumps/he_<case>/images.mat``.
3. Full ``BDmask``: 0 differing pixels on tests 1-3 (HE2 always; HE at
   ``kmeans_seed=28``).

Offline MATLAB: ``dump_annotation_primitives.m``, ``dump_bdc_he2.m``,
``dump_bdc_he.m``. Enable with::

    TMEQ_RUN_MATLAB_PARITY=1 pytest -q tests/test_tumor_annotation_matlab_parity.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from skimage import io

from pycurvelets._he_bdc_annotation import (
    annotate_he,
    annotate_he2,
    matlab_fspecial_disk,
    matlab_histeq,
    matlab_im2bw,
    matlab_padarray,
)
from pycurvelets._he_bdc_reg1 import DEFAULT_KMEANS_SEED, matlab_strel_disk
from pycurvelets.tumor_annotation_from_HE import (
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

if os.environ.get("TMEQ_RUN_MATLAB_PARITY") != "1":
    pytest.skip(
        "MATLAB parity tests disabled (set TMEQ_RUN_MATLAB_PARITY=1 to enable)",
        allow_module_level=True,
    )

_TESTS_DIR = Path(__file__).resolve().parent
_FIXTURE_ROOT = _TESTS_DIR / "test_for_shg_he_registration_BDcreation"
_DUMPS = _TESTS_DIR / "matlab_parity" / "dumps"
_GOLDEN_DIR = _FIXTURE_ROOT / "SHG" / "CA_Boundary"

HE2_CASES = (
    ("test1", 1.5, "HE_registered_test1"),
    ("test2", 2.0, "HE_registered_test2"),
    ("test3", 3.0, "HE_registered_test3"),
)
_IDS = [c[0] for c in HE2_CASES]


def _skip_unless_files(*paths: Path) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip("missing fixture(s):\n" + "\n".join(f"  {m}" for m in missing))


def _loadmat(path: Path) -> dict:
    return sio.loadmat(str(path), squeeze_me=True)


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).squeeze().astype(bool)


# ---------------------------------------------------------------------------
# 1. primitive ports
# ---------------------------------------------------------------------------


def test_primitives_match_matlab_probe() -> None:
    mat = _DUMPS / "annotation_primitives.mat"
    _skip_unless_files(mat)
    m = _loadmat(mat)
    for r in (1, 2, 3, 4, 5, 7, 11, 14, 21):
        np.testing.assert_allclose(
            matlab_fspecial_disk(float(r)), m[f"disk_r{r}"], atol=1e-15, rtol=0.0
        )
    np.testing.assert_allclose(matlab_fspecial_disk(10.5), m["disk_r10p5"], atol=1e-15, rtol=0.0)
    for r in (1, 2, 3, 4, 5, 7, 11):
        np.testing.assert_array_equal(matlab_strel_disk(r), m[f"strel_r{r}"].astype(bool))
    np.testing.assert_array_equal(matlab_padarray(m["pad_src"], 2), m["pad_sym"])
    np.testing.assert_array_equal(matlab_histeq(m["histeq_rand_in"]), m["histeq_rand_out"])
    np.testing.assert_array_equal(matlab_histeq(m["histeq_u8_in"]), m["histeq_u8_out"])
    np.testing.assert_allclose(
        matlab_histeq(np.asarray(m["histeq_dbl_in"], dtype=np.float64)),
        np.asarray(m["histeq_dbl_out"], dtype=np.float64),
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        matlab_im2bw(np.asarray(m["im2bw_in"], dtype=np.float64), 0.5),
        _as_bool(m["im2bw_out"]),
    )


# ---------------------------------------------------------------------------
# 2. HE2 preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,ppm,he_folder", HE2_CASES, ids=_IDS)
def test_he2_intermediates_match_matlab_dump(case_id: str, ppm: float, he_folder: str) -> None:
    dump = _DUMPS / f"he2_{case_id}" / "images.mat"
    he_path = _FIXTURE_ROOT / "HE" / he_folder / "patient_001.tif"
    _skip_unless_files(dump, he_path)
    ml = _loadmat(dump)
    _mask, dbg = annotate_he2(io.imread(str(he_path)), ppm)

    np.testing.assert_allclose(dbg["he_adjusted"], ml["he_adjusted"], atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(_as_bool(dbg["BW_nuclei"]), _as_bool(ml["BW_nuclei"]))
    np.testing.assert_allclose(dbg["maskednucleiImage"], ml["maskednucleiImage"], atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(_as_bool(dbg["BW_collagen1"]), _as_bool(ml["BW_collagen1"]))
    np.testing.assert_array_equal(_as_bool(dbg["BW_nobackground"]), _as_bool(ml["BW_nobackground"]))
    assert float(np.asarray(dbg["sat_thresh"]).squeeze()) == pytest.approx(
        float(np.asarray(ml["sat_thresh"]).squeeze()), abs=1e-15
    )
    np.testing.assert_array_equal(_as_bool(dbg["epith_cell_BW"]), _as_bool(ml["epith_cell_BW"]))
    np.testing.assert_array_equal(_as_bool(dbg["BWx"]), _as_bool(ml["BWx"]))
    np.testing.assert_array_equal(_as_bool(dbg["mask_image1"]), _as_bool(ml["mask_image1"]))
    np.testing.assert_allclose(dbg["B"], ml["B"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dbg["mask_temp"], ml["mask_temp"], atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# 3. HE2 full mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,ppm,he_folder", HE2_CASES, ids=_IDS)
def test_he2_bdmask_is_pixel_identical(case_id: str, ppm: float, he_folder: str) -> None:
    dump = _DUMPS / f"he2_{case_id}" / "images.mat"
    he_path = _FIXTURE_ROOT / "HE" / he_folder / "patient_001.tif"
    golden = _GOLDEN_DIR / f"BDcreationHE_{case_id}results_mask for patient_001.tif.tif"
    _skip_unless_files(dump, he_path, golden)
    ml = _loadmat(dump)
    mask, dbg = annotate_he2(io.imread(str(he_path)), ppm)
    n_dump = int(np.count_nonzero(_as_bool(mask) != _as_bool(ml["BDmask"])))
    n_golden = int(np.count_nonzero(_as_bool(mask) != (io.imread(str(golden)) > 0)))
    assert n_dump == 0, f"[{case_id}] HE2 vs dump BDmask differs in {n_dump} pixels"
    assert n_golden == 0, f"[{case_id}] HE2 vs golden differs in {n_golden} pixels"
    assert dbg["BDmask"].dtype == np.bool_ or dbg["BDmask"].dtype == bool

    params = TumorAnnotationFromHEParameters(
        HEfilepath=str(he_path.parent),
        HEfilename=he_path.name,
        pixelpermicron=ppm,
        areaThreshold=5000.0,
        SHGfilepath=str(_FIXTURE_ROOT / "SHG"),
        annotation_method="hsv",
    )
    public = tumor_annotation_from_he(params, save_output=False, return_debug=False)
    assert int(np.count_nonzero(_as_bool(public) != _as_bool(ml["BDmask"]))) == 0


# ---------------------------------------------------------------------------
# HE (BDcreationHE.m) at the pinned k-means seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,ppm,he_folder", HE2_CASES, ids=_IDS)
def test_he_bdmask_is_pixel_identical_at_seed_28(case_id: str, ppm: float, he_folder: str) -> None:
    dump = _DUMPS / f"he_{case_id}" / "images.mat"
    he_path = _FIXTURE_ROOT / "HE" / he_folder / "patient_001.tif"
    _skip_unless_files(dump, he_path)
    ml = _loadmat(dump)
    seed = int(np.asarray(ml["kmeansSeed"]).squeeze()) if "kmeansSeed" in ml else DEFAULT_KMEANS_SEED
    assert seed == DEFAULT_KMEANS_SEED
    he = io.imread(str(he_path))
    mask, dbg = annotate_he(he, ppm, kmeans_seed=seed)

    np.testing.assert_array_equal(dbg["S"], ml["S"])
    # uint8 imfilter (IPP) can differ by 1 gray level from float64 + round;
    # MATLAB's own imfilter(double) rounded also disagrees with imfilter(uint8).
    k3_abs = np.abs(dbg["k3"].astype(np.int16) - np.asarray(ml["k3"]).astype(np.int16))
    assert int(k3_abs.max()) <= 1
    ml_blue = int(np.asarray(ml["blue_cluster_num"]).squeeze())
    assert int(dbg["blue_cluster_num"][0]) + 1 == ml_blue
    np.testing.assert_array_equal(_as_bool(dbg["BWx"]), _as_bool(ml["BWx"]))
    np.testing.assert_array_equal(_as_bool(dbg["mask_image"]), _as_bool(ml["mask_image"]))
    n_diff = int(np.count_nonzero(dbg["BDmask"] != ml["BDmask"]))
    assert n_diff == 0, f"[{case_id}] HE vs dump BDmask differs in {n_diff} pixels"
    assert int(np.count_nonzero(_as_bool(mask) != (np.asarray(ml["BDmask"]) > 0))) == 0
