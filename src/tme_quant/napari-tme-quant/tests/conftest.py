import sys
from pathlib import Path

import pytest

# Resolve paths relative to this file so the tests work cross-platform
# (Windows native, MSYS2, WSL) without hard-coded absolute paths.
_TESTS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _TESTS_DIR.parent           # napari-tme-quant/
_PLUGIN_SRC = _PLUGIN_ROOT / "src"        # napari-tme-quant/src  (contains napari_tme_quant)
_LIBRARY_SRC = _PLUGIN_ROOT.parent / "src"  # tme_quant/src        (contains tme_quant)

for _p in (_PLUGIN_SRC, _LIBRARY_SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import napari
    _NAPARI_AVAILABLE = True
except ImportError:
    _NAPARI_AVAILABLE = False


@pytest.fixture
def make_napari_viewer(qtbot):
    if not _NAPARI_AVAILABLE:
        pytest.skip("napari not installed")

    viewers = []

    def factory(**kwargs):
        viewer = napari.Viewer(**kwargs)
        viewers.append(viewer)
        return viewer

    yield factory

    for v in viewers:
        v.close()
