"""Fast, always-on checks for the ``BDcreation_reg.m`` (reg1) primitives.

Bit-exact dumps and the full ITK registration live in
``test_shg_he_registration_matlab_parity.py`` (dev-only). This module only
covers cheap invariants so a broken port is caught in CI without MATLAB dumps.
"""

from __future__ import annotations

import numpy as np

from pycurvelets._he_bdc_reg1 import (
    DEFAULT_KMEANS_SEED,
    matlab_im2uint8,
    matlab_srgb2lab_uint8,
    matlab_strel_disk,
    matlab_stretchlim,
)


def test_default_kmeans_seed_is_the_golden_optimum() -> None:
    # BDcreation_reg.m never seeds kmeans; seed 28 is the rare optimum that
    # matches the committed test8/test9 goldens (see probe_reg1_kmeans.m).
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
