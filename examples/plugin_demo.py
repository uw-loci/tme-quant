"""
Interactive demo script for the napari-curvealign plugin.
Run with: python examples/plugin_demo.py
"""

import napari
import numpy as np
from skimage import filters
import sys


def create_synthetic_fibers(size=512):
    """Create a synthetic fiber-like image for demonstration."""
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    x_grid, y_grid = np.meshgrid(x, y)

    fibers = np.sin(x_grid) * np.cos(y_grid / 2) + 0.5 * np.cos(x_grid / 1.5) * np.sin(y_grid)
    fibers += np.random.rand(size, size) * 0.2
    fibers = (fibers - fibers.min()) / (fibers.max() - fibers.min())
    fibers = filters.gaussian(fibers, sigma=1.5)

    return fibers.astype(np.float32)


def main():
    """Launch napari with a synthetic image and load the CurveAlign widget."""
    print("=" * 60)
    print("CurveAlign Napari Plugin Demo")
    print("=" * 60)

    viewer = napari.Viewer()
    print("\nCreating synthetic fiber image...")
    fiber_image = create_synthetic_fibers(size=512)
    viewer.add_image(fiber_image, name="Synthetic Fibers", colormap="gray")

    print("Loading CurveAlign plugin widget...")
    try:
        from napari_curvealign.widget import CurveAlignWidget

        widget_instance = CurveAlignWidget(viewer)
        viewer.window.add_dock_widget(widget_instance, name="CurveAlign", area="right")
        print("Plugin loaded successfully.")
    except Exception as exc:
        print(f"Failed to load plugin: {exc}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Install editable package: uv pip install -e .")
        print("2. Check napari import/version.")
        print("3. Verify widget import: from napari_curvealign.widget import CurveAlignWidget")
        sys.exit(1)

    print("\nStarting napari...")
    napari.run()


if __name__ == "__main__":
    main()
