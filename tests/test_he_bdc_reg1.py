"""Fast, always-on checks for the ``BDcreation_reg.m`` (reg1) primitives.

Bit-exact dumps and the full ITK v3 registration live in
``test_shg_he_registration_matlab_parity.py`` (dev-only). This module only
covers cheap invariants so a broken port is caught in CI without MATLAB dumps.
"""

from __future__ import annotations

import numpy as np

import pytest

from pycurvelets._he_bdc_common import (
    COLLAGEN_HUE_MAX,
    COLLAGEN_HUE_MIN,
    NUCLEI_HUE_MAX,
    NUCLEI_HUE_MIN,
    RGB_EOSIN_B_MIN,
    RGB_EOSIN_G_MAX,
    RGB_EOSIN_R_MIN,
    RGB_NUCLEI_B_MAX,
    RGB_NUCLEI_G_MIN,
    RGB_NUCLEI_R_MAX,
    rgb_threshold_masks_uint8,
)
from pycurvelets._he_bdc_reg1 import (
    DEFAULT_KMEANS_SEED,
    matlab_im2uint8,
    matlab_srgb2lab_uint8,
    matlab_strel_disk,
    matlab_stretchlim,
)
from pycurvelets.SHG_HE_registration import (
    _require_ecm_method,
    _require_matlab_method,
)


def test_default_kmeans_seed_is_the_golden_optimum() -> None:
    # BDcreation_reg.m never seeds kmeans; seed 28 is the rare optimum that
    # matches the committed test8/test9 goldens.
    assert DEFAULT_KMEANS_SEED == 28


def test_im2uint8_rounds_half_up() -> None:
    x = np.array([0.0, 1.0 / 255.0, 1.5 / 255.0, 1.0])
    np.testing.assert_array_equal(matlab_im2uint8(x), np.array([0, 1, 2, 255], dtype=np.uint8))


def test_stretchlim_uint8_ramp() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    lim = matlab_stretchlim(ramp, (0.01, 0.99))
    # First bin whose CDF exceeds 0.01 / reaches 0.99, converted to [0, 1].
    assert lim.shape == (2, 1)
    assert 0.0 <= lim[0, 0] < lim[1, 0] <= 1.0


def test_strel_disk_matches_matlab_small_and_decomposed() -> None:
    # r < 3: Euclidean disk. r = 3, n = 4: Adams decomposition -> 5x5 square.
    d1 = matlab_strel_disk(1)
    assert d1.shape == (3, 3) and bool(d1[1, 1]) and int(d1.sum()) == 5
    d3 = matlab_strel_disk(3)
    assert d3.shape == (5, 5) and bool(np.all(d3))


def test_srgb2lab_uint8_black_white_and_red_cut() -> None:
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 1] = 255
    rgb[0, 2] = (201, 99, 101)  # just inside the eosin RGB cut
    lab = matlab_srgb2lab_uint8(rgb)
    assert lab.dtype == np.uint8 and lab.shape == rgb.shape
    # L of black is 0; L of white is 255 in MATLAB's uint8 Lab encoding.
    assert lab[0, 0, 0] == 0
    assert lab[0, 1, 0] == 255
    # a* of a red-ish pixel is above the neutral 128.
    assert lab[0, 2, 1] > 128


def test_hsv_and_rgb_thresholds_match_matlab() -> None:
    # BDcreation_reg2.m HSV bands.
    assert (NUCLEI_HUE_MIN, NUCLEI_HUE_MAX) == (0.500, 0.790)
    assert (COLLAGEN_HUE_MIN, COLLAGEN_HUE_MAX) == (0.837, 0.066)
    # BDcreation_reg.m RGB cuts on decorrstretched uint8.
    assert (RGB_NUCLEI_R_MAX, RGB_NUCLEI_G_MIN, RGB_NUCLEI_B_MAX) == (120, 150, 120)
    assert (RGB_EOSIN_R_MIN, RGB_EOSIN_G_MAX, RGB_EOSIN_B_MIN) == (200, 100, 100)

    s = np.zeros((1, 3, 3), dtype=np.uint8)
    s[0, 0] = (119, 151, 119)  # nuclei
    s[0, 1] = (201, 99, 101)  # eosin
    s[0, 2] = (128, 128, 128)  # neither
    nuclei, eosin = rgb_threshold_masks_uint8(s)
    assert bool(nuclei[0, 0]) and not bool(eosin[0, 0])
    assert bool(eosin[0, 1]) and not bool(nuclei[0, 1])
    assert not bool(nuclei[0, 2]) and not bool(eosin[0, 2])


def test_only_itk_v3_registrar_is_supported() -> None:
    assert _require_matlab_method(None) == "matlab"
    assert _require_matlab_method("matlab") == "matlab"
    assert _require_ecm_method("hsv") == "hsv"
    assert _require_ecm_method("rgb") == "rgb"
    with pytest.raises(ValueError, match="not supported"):
        _require_matlab_method("mi_ncc")
    with pytest.raises(ValueError, match="not supported"):
        _require_ecm_method("auto")
