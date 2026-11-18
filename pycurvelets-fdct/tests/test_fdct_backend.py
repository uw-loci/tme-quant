"""FDCT backend: registration, Protocol conformance and newCurv.m geometry."""

import numpy as np
import pytest

pytest.importorskip("curvelops", reason="curvelops (CurveLab) is not installed")

from pycurvelets_fdct.backend import (
    FDCTBackend,
    number_of_scales,
    wedge_angle_deg,
)

from pycurvelets.backends import available_backends, load_backend
from pycurvelets.backends.conformance import check_backend


def test_registered_under_fdct_entry_point():
    assert "fdct" in available_backends()
    assert isinstance(load_backend("fdct"), FDCTBackend)


@pytest.mark.parametrize("shape", [(256, 256), (240, 320), (512, 512)])
def test_conforms_to_backend_protocol(shape):
    check_backend(FDCTBackend(), shape=shape)


def test_number_of_scales_matches_newcurv():
    # newCurv.m: nbscales = floor(log2(min(M, N)) - 3)
    assert number_of_scales((512, 512)) == 6
    assert number_of_scales((300, 420)) == 5


def test_wedge_angles_follow_curvelab_numbering():
    # 64 wedges -> 5.625 degree sectors, numbered clockwise from 225 degrees;
    # the angle is the centre of the sector.
    step = 360 / 64
    assert wedge_angle_deg(0, step) == pytest.approx(222.1875)
    assert wedge_angle_deg(16, step) == pytest.approx(132.1875)
    assert wedge_angle_deg(31, step) == pytest.approx(47.8125)
    # Sector centres below 45 degrees are folded up by 180.
    assert wedge_angle_deg(32, step) == pytest.approx(42.1875 + 180)


def test_scale_geometry_lists_half_the_wedges():
    backend = FDCTBackend()
    shape = (256, 256)
    coefficients = backend.forward(np.zeros(shape))
    scale = len(coefficients) - 2
    geometry = backend.scale_geometry(shape, scale)
    assert len(geometry.wedges) == len(coefficients[scale]) // 2
    assert geometry.angle_step_deg == pytest.approx(360 / len(coefficients[scale]))
