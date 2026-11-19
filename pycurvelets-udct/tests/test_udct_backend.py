from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("curvelets")

from pycurvelets.backends import available_backends, load_backend
from pycurvelets.backends.conformance import _line_image, check_backend
from pycurvelets.models import CurveletControlParameters
from pycurvelets.new_curv import new_curv
from pycurvelets_udct.backend import (
    UDCTBackend,
    angular_wedges_config,
    number_of_scales,
    padded_shape,
    wedges_per_direction,
)

CORE_TEST_IMAGES = Path(__file__).resolve().parents[2] / "pycurvelets-core" / "tests" / "test_images"


def test_registered_as_udct_entry_point():
    assert "udct" in available_backends()
    assert isinstance(load_backend("udct"), UDCTBackend)


# (300, 300) and (250, 333) are not multiples of the decimation, so they
# exercise the padding; (64, 64) is the smallest supported size.
@pytest.mark.parametrize("shape", [(256, 256), (240, 320), (300, 300), (250, 333), (512, 512), (64, 64)])
def test_conforms_to_backend_protocol(shape):
    check_backend(UDCTBackend(), shape=shape)


def test_wedge_counts_double_every_other_scale():
    assert [wedges_per_direction(j) for j in range(1, 7)] == [3, 6, 6, 12, 12, 24]
    assert angular_wedges_config(5).tolist() == [[3, 3], [6, 6], [6, 6], [12, 12]]


def test_scale_count_matches_fdct_formula_and_rejects_tiny_images():
    assert number_of_scales((256, 256)) == 5
    assert number_of_scales((512, 300)) == 5
    with pytest.raises(ValueError, match="too small"):
        number_of_scales((32, 128))


def test_padded_shape_is_a_multiple_of_the_decimation():
    assert padded_shape((256, 256), 5) == (256, 256)
    assert padded_shape((300, 250), 5) == (304, 256)


def test_every_wedge_is_listed_with_angles_in_fdct_range():
    backend = UDCTBackend()
    shape = (256, 256)
    coefficients = backend.forward(np.zeros(shape))
    scale = len(coefficients) - 2
    geometry = backend.scale_geometry(shape, scale)
    # One-sided windows: no mirrored pairs, so every wedge is described.
    assert len(geometry.wedges) == len(coefficients[scale])
    angles = np.array([w.angle_deg for w in geometry.wedges])
    assert ((angles > 45) & (angles <= 225)).all()
    assert geometry.angle_step_deg == pytest.approx(180 / len(geometry.wedges))
    for wedge in geometry.wedges:
        assert wedge.rows.min() >= 1 and wedge.cols.min() >= 1


def test_inverse_crops_padded_images_back_to_shape():
    backend = UDCTBackend()
    img = np.random.default_rng(1).standard_normal((300, 250))
    reconstructed = backend.inverse(backend.forward(img), img.shape)
    assert reconstructed.shape == img.shape
    np.testing.assert_allclose(reconstructed, img, atol=1e-9)


@pytest.mark.parametrize("theta", [20.0, 70.0, 110.0, 160.0])
def test_new_curv_recovers_line_orientation(theta):
    params = CurveletControlParameters(keep=0.01, scale=1, radius=3, backend="udct")
    curves, _, inc = new_curv(_line_image((256, 256), theta), params)
    assert len(curves) > 0
    mean = np.degrees(np.angle(np.mean(np.exp(2j * np.radians(curves["angle"])))) / 2) % 180
    error = min((mean - theta) % 180, (theta - mean) % 180)
    assert error <= inc, f"line at {theta} deg reported at {mean:.2f} deg (allowed {inc})"


@pytest.mark.skipif("fdct" not in available_backends(), reason="pycurvelets-fdct not installed")
@pytest.mark.skipif(not (CORE_TEST_IMAGES / "real1.tif").is_file(), reason="needs a tme-quant checkout")
def test_fiber_orientation_statistics_agree_with_fdct():
    import matplotlib.pyplot as plt

    img = plt.imread(CORE_TEST_IMAGES / "real1.tif", format="TIF")
    stats = {}
    for backend in ("fdct", "udct"):
        curves, _, _ = new_curv(img, CurveletControlParameters(keep=0.05, scale=1, radius=3, backend=backend))
        z = np.mean(np.exp(2j * np.radians(curves["angle"])))
        stats[backend] = (np.degrees(np.angle(z) / 2) % 180, abs(z))
    (fdct_mean, fdct_r), (udct_mean, udct_r) = stats["fdct"], stats["udct"]
    assert min((fdct_mean - udct_mean) % 180, (udct_mean - fdct_mean) % 180) < 5.0, stats
    assert abs(fdct_r - udct_r) < 0.05, stats
