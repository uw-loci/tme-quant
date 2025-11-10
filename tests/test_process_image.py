import pytest

from pycurvelets.models import CurveletControlParameters

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
except ModuleNotFoundError:
    pytest.skip(
        "curvelops not available; skipping new_curv tests", allow_module_level=True
    )

from pycurvelets.new_curv import new_curv


@pytest.fixture(scope="module")
def standard_test_image():
    """Load a standard test image once for the module."""
    img_path = os.path.join(os.path.dirname(__file__), "test_images", "real1.tif")
    img = plt.imread(img_path, format="TIF")
    return img
