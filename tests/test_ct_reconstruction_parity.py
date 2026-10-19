"""MATLAB coefficient selection and default transform parameter regressions."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctfire_py.ct_reconstruction import ct_reconstruction
from ctfire_py import matlab_wrapping


@pytest.mark.parametrize("fraction,expected_sum", [(0, 0), (1 / 3, 17), (1, 42)])
def test_matlab_threshold_rank(monkeypatch, fraction, expected_sum):
    # Nine distinct magnitudes: MATLAB's third-ranked cutoff is 7, so strict
    # thresholding retains 9 and 8. The finest scale is excluded independently.
    def forward(image):
        assert image.dtype == np.float64
        return [
            [np.array(values, dtype=np.complex128)]
            for values in (
                [9, 8, 7],
                [6, 5],
                [4, 3],
                [2, 1],
            )
        ]

    def inverse(coefficients):
        assert not np.any(coefficients[-1][0])
        total = sum(wedge.real.sum() for scale in coefficients for wedge in scale)
        return np.full((96, 128), total)

    monkeypatch.setattr(matlab_wrapping, "forward", forward)
    monkeypatch.setattr(matlab_wrapping, "inverse", inverse)
    result = ct_reconstruction(
        np.ones((96, 128), dtype=np.uint8), "unused.mat", fraction, 3
    )
    np.testing.assert_array_equal(result, np.full((96, 128), expected_sum))
