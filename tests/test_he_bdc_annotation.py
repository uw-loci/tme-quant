"""Fast, always-on checks for BDcreationHE / HE2 annotation primitives.

Bit-exact dumps live in ``test_tumor_annotation_matlab_parity.py`` (dev-only).
This module only covers cheap invariants so a broken port is caught in CI
without MATLAB dumps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from pycurvelets._he_bdc_annotation import (
    EPITH_BINARY_THRESHOLD,
    HE2_PPM_CAP,
    annotate_he,
    annotate_he2,
    matlab_fspecial_disk,
    matlab_histeq,
    matlab_im2bw,
    matlab_padarray,
    pick_he_epithelial_cluster,
    prepare_he2_image,
)
from pycurvelets._he_bdc_common import (
    COLLAGEN_HUE_MAX,
    COLLAGEN_HUE_MIN,
    NUCLEI_HUE_MAX,
    NUCLEI_HUE_MIN,
)
from pycurvelets._he_bdc_reg1 import DEFAULT_KMEANS_SEED, matlab_strel_disk
from pycurvelets.tumor_annotation_from_HE import TumorAnnotationFromHEParameters

_FIXTURE_ROOT = Path(__file__).resolve().parent / "test_for_shg_he_registration_BDcreation"


def test_hsv_thresholds_match_bdcreation_he2() -> None:
    assert (NUCLEI_HUE_MIN, NUCLEI_HUE_MAX) == (0.500, 0.790)
    assert (COLLAGEN_HUE_MIN, COLLAGEN_HUE_MAX) == (0.837, 0.066)
    assert EPITH_BINARY_THRESHOLD == 0.001
    assert HE2_PPM_CAP == 2.0


def test_histeq_uint8_keeps_class_and_uses_default_64_levels() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = matlab_histeq(ramp)
    assert out.dtype == np.uint8 and out.shape == ramp.shape
    # Flattening a full 0..255 histogram must use every output bucket of the
    # default N=64 mapping (T = k/63).
    assert int(out.min()) == 0 and int(out.max()) == 255


def test_histeq_double_default_n64() -> None:
    x = np.linspace(0.0, 1.0, 64)
    out = matlab_histeq(x)
    assert out.dtype == np.float64
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_fspecial_disk_is_normalized_odd_kernel() -> None:
    for r in (1, 3, 11):
        h = matlab_fspecial_disk(r)
        assert h.ndim == 2 and h.shape[0] == h.shape[1] and h.shape[0] % 2 == 1
        assert h.sum() == pytest.approx(1.0, abs=1e-15)
        assert np.all(h >= -1e-15)


def test_padarray_symmetric_duplicates_edge() -> None:
    src = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
    out = matlab_padarray(src, 2, mode="symmetric")
    assert out.shape == (8, 8) and out.dtype == np.uint8
    assert out[2, 2] == src[0, 0]
    np.testing.assert_array_equal(out[1, 2:6], src[0])
    np.testing.assert_array_equal(out[2:6, 1], src[:, 0])


def test_im2bw_double_is_strict_greater() -> None:
    x = np.array([0.0, 0.001, 0.5, 1.0])
    np.testing.assert_array_equal(matlab_im2bw(x, 0.001), np.array([False, False, True, True]))
    u8 = np.array([0, 1, 255], dtype=np.uint8)
    # im2double(uint8) then > 0.001 -> 1/255 ≈ 0.0039 is True.
    np.testing.assert_array_equal(matlab_im2bw(u8, 0.001), np.array([False, True, True]))


def test_strel_disk_r3_is_adams_5x5_not_euclidean() -> None:
    d3 = matlab_strel_disk(3)
    assert d3.shape == (5, 5) and bool(np.all(d3))


def test_prepare_he2_ppm3_uses_ceil_imresize_size() -> None:
    # MATLAB imresize(I, 2/3) on 512x512 -> ceil(512*2/3) = 342, not round()=341.
    img = np.zeros((512, 512, 3), dtype=np.float64)
    out, pix = prepare_he2_image(img, 3.0)
    assert pix == 2.0
    assert out.shape[:2] == (342, 342)


def test_he_cluster_pick_is_product_of_1based_ranks() -> None:
    # Cluster 0 darkest+dimmest -> ranks 1,1 product 1 (picked).
    centers = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    intensity = np.array([10.0, 20.0, 30.0, 40.0])
    assert pick_he_epithelial_cluster(centers, intensity) == 0
    # Swap so cluster 2 is darkest but brightest intensity: ranks 1 * 4 = 4;
    # cluster 0 is brightest-center (rank 4) and dimmest (rank 1) -> 4; cluster 1
    # ranks 2*2=4. Lowest unique is still cluster 0 if we only swap 2's intensity
    # high: use a case where cluster 3 wins.
    intensity = np.array([40.0, 30.0, 20.0, 10.0])
    # center ranks 1,2,3,4 and intensity ranks 4,3,2,1 -> products 4,6,6,4; sort
    # stable picks cluster 0 first among ties. Force cluster 3 unique min:
    intensity = np.array([50.0, 40.0, 30.0, 1.0])
    # products: (1*4)=4, (2*3)=6, (3*2)=6, (4*1)=4 -> tie 0 and 3, stable -> 0.
    assert pick_he_epithelial_cluster(centers, intensity) == 0
    centers = np.array([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    intensity = np.array([40.0, 30.0, 20.0, 10.0])
    assert pick_he_epithelial_cluster(centers, intensity) == 3


def _he_fixture(case: str) -> Path:
    path = _FIXTURE_ROOT / "HE" / f"HE_registered_{case}" / "patient_001.tif"
    if not path.is_file():
        pytest.skip(f"missing fixture {path}")
    return path


@pytest.mark.parametrize("case,ppm", [("test1", 1.5), ("test2", 2.0), ("test3", 3.0)])
def test_he2_mask_is_bool_original_shape(case: str, ppm: float) -> None:
    he = io.imread(str(_he_fixture(case)))
    mask, debug = annotate_he2(he, ppm)
    assert mask.dtype == np.bool_ and mask.shape == he.shape[:2]
    assert debug["BDmask"].shape == mask.shape
    assert str(debug["annotation_method"][0]) == "hsv"


@pytest.mark.parametrize("case,ppm", [("test2", 2.0)])
def test_he_mask_is_bool_with_uint8_bdmask(case: str, ppm: float) -> None:
    he = io.imread(str(_he_fixture(case)))
    mask, debug = annotate_he(he, ppm, kmeans_seed=DEFAULT_KMEANS_SEED)
    assert mask.dtype == np.bool_ and mask.shape == he.shape[:2]
    assert debug["BDmask"].dtype == np.uint8
    assert int(debug["BDmask"].max()) in (0, 255)
    assert TumorAnnotationFromHEParameters.kmeans_seed == DEFAULT_KMEANS_SEED
