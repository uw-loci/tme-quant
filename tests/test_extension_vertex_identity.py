"""Extension must preserve vertex identity until MATLAB's remove_repeat stage."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
backend = importlib.import_module("ctfire_py.fire_2d_angle").fiber_backend
pytestmark = pytest.mark.skipif(backend is None, reason="C++ backend is not available")


def parameters(**overrides):
    result = {
        "thresh_LMPdist": 0, "thresh_LMP": 0.2, "thresh_ext": 1.1,
        "lam_dirdecay": 0.5, "thresh_linkd": 0,
        "thresh_linka": -0.866, "s_fiberdir": 4,
    }
    result.update(overrides)
    return result


def test_coincident_free_tips_keep_distinct_vertex_ids():
    image = np.ones((9, 11), dtype=np.float32)
    points = np.array([[4, 3, 0], [4, 7, 0]], dtype=np.int32)
    vertices, fibers, info, radii = backend.extend_xlink(
        1, 9, 11, image.ravel(), points, parameters()
    )
    assert len(fibers) == 32
    assert len(vertices) == 34  # Two roots, plus a separate tip for every branch.
    assert {fiber["v"][0] for fiber in fibers} == {0, 1}
    endpoints = [fiber["v"][-1] for fiber in fibers]
    assert len(set(endpoints)) == 32
    coincident = [i for i in endpoints if tuple(vertices[i, :2]) == (4, 5)]
    assert len(coincident) == 2
    assert all(len(info[i]["f"]) == 1 for i in coincident)
    np.testing.assert_array_equal(radii, np.ones(34))


def test_different_routes_with_same_endpoint_pixels_both_survive():
    image = np.zeros((11, 11), dtype=np.float32)
    image[6, 6] = 2
    image[6, 5] = 0.5
    image[5, 4] = image[7, 4] = image[6, 3] = 1
    vertices, fibers, info, _ = backend.extend_xlink(
        1, 11, 11, image.ravel(), np.array([[6, 6, 0]], dtype=np.int32),
        parameters(thresh_LMPdist=2, thresh_ext=0.2),
    )
    paths = {tuple(map(tuple, vertices[fiber["v"], :2])) for fiber in fibers}
    assert paths == {
        ((6, 6), (5, 4), (6, 3)),
        ((6, 6), (7, 4), (6, 3)),
    }
    tips = [fiber["v"][-1] for fiber in fibers]
    assert tips[0] != tips[1]
    assert all(len(info[tip]["f"]) == 1 for tip in tips)


def test_connections_reuse_nucleation_ids_and_reverse_duplicates_are_removed():
    image = np.zeros((9, 13), dtype=np.float32)
    image[4, 2:11] = 1
    points = np.array([[4, 3, 0], [4, 7, 0]], dtype=np.int32)
    vertices, fibers, info, _ = backend.extend_xlink(
        1, 9, 13, image.ravel(), points, parameters(thresh_ext=0.5)
    )
    roots = {tuple(point[:2]): i for i, point in enumerate(vertices)
             if tuple(point[:2]) in {(4, 3), (4, 7)}}
    connections = [fiber for fiber in fibers
                   if {fiber["v"][0], fiber["v"][-1]} == set(roots.values())]
    assert len(connections) == 1
    assert all(i in connections[0]["v"] for i in roots.values())
    for fiber_index, fiber in enumerate(fibers):
        for vertex in fiber["v"]:
            assert fiber_index in info[vertex]["f"]


def test_unused_nucleation_vertices_are_trimmed():
    image = np.zeros((9, 11), dtype=np.float32)
    points = np.array([[4, 3, 0], [4, 7, 0]], dtype=np.int32)
    vertices, fibers, info, radii = backend.extend_xlink(
        1, 9, 11, image.ravel(), points, parameters()
    )
    assert vertices.shape == (0, 3)
    assert fibers == []
    assert info == []
    assert len(radii) == 0
