"""Regression coverage for DSM ownership and MATLAB singleton-depth smoothing."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
fire = importlib.import_module("ctfire_py.fire_2d_angle")


@pytest.mark.parametrize("readonly", [False, True])
def test_nucleation_preserves_input_and_is_repeatable(readonly):
    if fire.fiber_backend is None:
        pytest.skip("C++ backend is not available")
    # A rectangular plateau exercises random tie-breaking and makes ceil(1)
    # sensitive to any noise accidentally retained in the caller's image.
    image = np.ones(7 * 11, dtype=np.float32)
    original = image.copy()
    image.flags.writeable = not readonly
    first = fire.fiber_backend.find_local_max(1, 7, 11, image, 2, 0.2)
    second = fire.fiber_backend.find_local_max(1, 7, 11, image, 2, 0.2)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(np.ceil(image), np.ceil(original))
    assert len(first) > 0
    assert set(map(tuple, first)) == set(map(tuple, second))


def test_initial_lmp_pruning_uses_discarded_candidates():
    if fire.fiber_backend is None:
        pytest.skip("C++ backend is not available")
    image = np.zeros((9, 11), dtype=np.float32)
    image[3:6, 4:7] = 1
    image[4, 5] = 2  # Initial radius is two.
    image[3:6, 3] = 3  # Three tied LMPs along the left face.
    points = np.array([[4, 5, 0]], dtype=np.int32)
    vertices, fibers, _, _ = fire.fiber_backend.extend_xlink(
        1, 9, 11, image.ravel(), points, {
            "thresh_LMPdist": 2, "thresh_LMP": 0.2,
            "thresh_ext": 1.1,  # Stop after the initial branch for this fixture.
            "lam_dirdecay": 0.5, "thresh_linkd": 0,
            "thresh_linka": -0.866, "s_fiberdir": 4,
        },
    )
    # A removes B; B must still remove C even though A-C distance equals two.
    assert len(fibers) == 1
    np.testing.assert_array_equal(vertices[fibers[0]["v"][-1], :2], [3, 3])


def test_lmp_face_does_not_compare_diagonally_across_corner():
    if fire.fiber_backend is None:
        pytest.skip("C++ backend is not available")
    image = np.zeros((9, 11), dtype=np.float32)
    image[3:6, 4:7] = 1
    image[4, 5] = 2
    image[2, 4] = 2  # Top-face maximum next to the top-left corner.
    image[3, 3] = 3  # Higher diagonal pixel on the left face.
    vertices, fibers, _, _ = fire.fiber_backend.extend_xlink(
        1, 9, 11, image.ravel(), np.array([[4, 5, 0]], dtype=np.int32), {
            "thresh_LMPdist": 0, "thresh_LMP": 0.2, "thresh_ext": 1.1,
            "lam_dirdecay": 0.5, "thresh_linkd": 0,
            "thresh_linka": -0.866, "s_fiberdir": 4,
        },
    )
    endpoints = {tuple(vertices[fiber["v"][-1], :2]) for fiber in fibers}
    assert (2, 4) in endpoints
    assert (3, 3) in endpoints


def test_final_endpoint_direction_uses_matlab_sample_position():
    if fire.fiber_backend is None:
        pytest.skip("C++ backend is not available")
    # At the shared endpoint, MATLAB samples [2,1] on the bent fiber,
    # giving cosine -1/sqrt(2). Sampling [2,2] instead gives -1 and
    # incorrectly merges it with the straight fiber at threshold -0.8.
    vertices = np.array([
        [1, 1, 1], [2, 1, 1], [2, 2, 1], [3, 2, 1],
        [4, 2, 1], [5, 2, 1], [6, 2, 1],
    ], dtype=np.int32)
    _, fibers, _, _, _ = fire.fiber_backend.process_fibers(
        1, 9, 11, np.ones(99, dtype=np.float32), vertices,
        [{"v": [0, 1, 2, 3]}, {"v": [3, 4, 5, 6]}],
        np.ones(7, dtype=np.float32), {
            "s_fiberdir": 2, "thresh_linka": -0.8,
            "thresh_linkd": 0, "thresh_flen": 0, "thresh_numv": 3,
        },
    )
    assert len(fibers) == 2


@pytest.mark.parametrize("sigma", [0.0, 0.3, 1.0])
def test_pipeline_distance_smoothing_includes_depth_axis(monkeypatch, sigma):
    image = np.zeros((9, 13), dtype=np.uint8)
    image[2:7, 3:10] = 255
    distance = fire.bw_dist(~(image > 5), "cityblock")
    planar = fire.smooth(distance, sigma)
    if sigma == 0:
        attenuation = 1.0
    else:
        radius = int(np.ceil(2 * sigma))
        attenuation = 1.0 / np.exp(
            -np.arange(-radius, radius + 1) ** 2 / (2 * sigma**2)
        ).sum()
    expected = (planar * attenuation).astype(np.float32)

    class ReachedNucleation(Exception):
        pass

    def inspect_dsm(depth, height, width, dsm, radius, minimum):
        assert (depth, height, width) == (1, 9, 13)
        np.testing.assert_allclose(dsm.reshape(height, width), expected, rtol=1e-6, atol=1e-7)
        raise ReachedNucleation

    monkeypatch.setattr(fire, "fiber_backend", SimpleNamespace(find_local_max=inspect_dsm))
    with pytest.raises(ReachedNucleation):
        fire.fire_2d_angle({
            "sigma_im": 0, "sigma_d": sigma, "dtype": "cityblock", "thresh_im2": 5,
        }, image, plotflag=0)
