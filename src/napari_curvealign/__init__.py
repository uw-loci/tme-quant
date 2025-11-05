"""napari-curvealign plugin."""

__version__ = "0.1.0"

def napari_experimental_provide_dock_widget():
    """Provide the CurveAlign widget to napari."""
    from .widget import CurveAlignWidget
    return [(CurveAlignWidget, {})]