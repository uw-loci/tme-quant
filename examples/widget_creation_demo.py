"""
Demo script to verify CurveAlign widget construction.
Run with: python examples/widget_creation_demo.py
"""

import sys


def run_widget_creation_demo() -> bool:
    """Return True when widget creation succeeds."""
    try:
        import napari
    except ImportError as exc:
        print(f"Failed to import napari: {exc}")
        return False

    try:
        from napari_curvealign.widget import CurveAlignWidget
    except ImportError as exc:
        print(f"Failed to import CurveAlignWidget: {exc}")
        return False

    try:
        viewer = napari.Viewer(show=False)
        widget = CurveAlignWidget(viewer)
        print(f"Widget created: {type(widget).__name__}")
        if hasattr(widget, "roi_manager"):
            print("ROI manager initialized.")
        viewer.close()
        return True
    except Exception as exc:
        print(f"Widget creation failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = run_widget_creation_demo()
    sys.exit(0 if ok else 1)
