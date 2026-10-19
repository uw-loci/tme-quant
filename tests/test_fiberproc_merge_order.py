"""Endpoint ordering controls the survivor and orientation of later merges."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
backend = importlib.import_module("ctfire_py.fire_2d_angle").fiber_backend


@pytest.mark.skipif(backend is None, reason="C++ backend is not available")
def test_merged_endpoint_list_keeps_lowest_fiber_id_first():
    # A--B--C--D, with the last input segment stored backwards (D--C).
    # Merging fibers 0 and 1 must change C's endpoint list [1,2] to [0,2],
    # not [2,0]. MATLAB then retains fiber 0 and returns A--B--C--D.
    # Appending 0 without sorting instead retains fiber 2 and reverses it.
    vertices = np.array([[1, 1, 1], [3, 1, 1], [5, 1, 1], [7, 1, 1]], dtype=np.int32)
    output, fibers, _, info, _ = backend.process_fibers(
        1, 9, 9, np.ones(81, dtype=np.float32), vertices,
        [{"v": [0, 1]}, {"v": [1, 2]}, {"v": [3, 2]}],
        np.ones(4, dtype=np.float32), {
            "s_fiberdir": 2, "thresh_linka": -0.8, "thresh_linkd": 0,
            "thresh_flen": 0, "thresh_numv": 3,
        },
    )
    assert len(fibers) == 1
    np.testing.assert_array_equal(output[fibers[0]["v"]], vertices)
    for vertex in fibers[0]["v"]:
        assert info[vertex]["f"] == [0]
