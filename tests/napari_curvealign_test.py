import os
import pytest

# Ensure Qt runs without a display (pre-import)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Skip if napari not available (CI)
pytest.importorskip("napari")
import napari


def test_napari_import_smoke():
    # Smoke test: importing napari should succeed without creating a Viewer
    assert hasattr(napari, "Viewer")