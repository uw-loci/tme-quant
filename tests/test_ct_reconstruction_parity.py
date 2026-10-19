"""MATLAB coefficient selection and default transform parameter regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctfire_py.ct_reconstruction import ct_reconstruction


@pytest.mark.parametrize("fraction,expected_sum", [(0, 0), (1 / 3, 17), (1, 42)])
def test_matlab_threshold_rank_and_non_power_of_two_scale_count(monkeypatch, fraction, expected_sum):
    # Nine distinct magnitudes: MATLAB's third-ranked cutoff is 7, so strict
    # thresholding retains 9 and 8. The finest scale is excluded independently.
    def forward(scales, angles, ac, image):
        assert scales == 4  # ceil(log2(96)-3), rather than floor(...)=3.
        assert (angles, ac) == (16, 0)
        assert image.dtype == np.float64
        return [[np.array(values, dtype=np.complex128)] for values in (
            [9, 8, 7], [6, 5], [4, 3], [2, 1],
        )]

    def inverse(height, width, scales, angles, ac, coefficients):
        assert (height, width, scales, angles, ac) == (96, 128, 4, 16, 0)
        assert not np.any(coefficients[-1][0])
        total = sum(wedge.real.sum() for scale in coefficients for wedge in scale)
        return np.full((height, width), total)

    wrapper = SimpleNamespace(fdct2d_forward_wrap=forward, fdct2d_inverse_wrap=inverse)
    monkeypatch.setitem(sys.modules, "curvelops", SimpleNamespace(fdct2d_wrapper=wrapper))
    result = ct_reconstruction(np.ones((96, 128), dtype=np.uint8), "unused.mat", fraction, 3)
    np.testing.assert_array_equal(result, np.full((96, 128), expected_sum))
