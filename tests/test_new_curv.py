import os
import pytest

# By default, skip curvelops-dependent tests (e.g., on CI). Enable locally with:
#   TMEQ_RUN_CURVELETS=1 pytest -q
if os.environ.get("TMEQ_RUN_CURVELETS") != "1":
    pytest.skip(
        "curvelops tests disabled (set TMEQ_RUN_CURVELETS=1 to enable)",
        allow_module_level=True,
    )

# Optional: attempt import; skip module if curvelet backend is missing
try:
    from pycurvelets.new_curv import new_curv  # type: ignore
except Exception:
    pytest.skip("curvelops not available; skipping new_curv tests", allow_module_level=True)
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os

"""
None of these pytest functions accurately test new_curv -- it only tests the inc value,
which is easily manipulatable. Need to identify how to get the solution matrix and compare it
with a resultant np array. In addition, it is identified that the new_curv.py erroneously
creates more coordinates than the MATLAB function, so the reason must be explored.
"""


def test_new_curv_1():
    """
    Test function of test image -- real1.tif
    """
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), "test_images", "real1.tif"),
        format="TIF",
    )

    # Reference .mat not present in repo; keep only inc smoke check

    in_curves, ct, inc = new_curv(img, {"keep": 0.01, "scale": 1, "radius": 3})
    in_curves_converted = np.array(
        [(d["center"], d["angle"]) for d in in_curves],
        dtype=[("center", "O"), ("angle", "O")],
    )

    assert inc == 5.625
    # assert np.allclose(in_curves_converted["center"], mat["center"])
    # assert np.allclose(in_curves_converted["angle"], mat["angle"])


def test_new_curv_2():
    """
    Test function of test image -- syn1.tif
    """
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), "test_images", "syn1.tif"),
        format="TIF",
    )

    in_curves, ct, inc = new_curv(img, {"keep": 0.1, "scale": 2, "radius": 5})

    assert inc == 11.25
    # assert ct


def test_new_curv_3():
    """
    Test function of test image -- syn2.tif
    """
    img = plt.imread(
        os.path.join(os.path.dirname(__file__), "test_images", "syn2.tif"),
        format="TIF",
    )

    in_curves, ct, inc = new_curv(img, {"keep": 0.01, "scale": 1, "radius": 3})

    assert inc == 5.625
