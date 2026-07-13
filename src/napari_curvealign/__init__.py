"""napari-curvealign plugin."""

__version__ = "0.1.0"

def napari_experimental_provide_dock_widget(viewer=None):
    """
    Provide the CurveAlign widget to napari.

    For npe2 plugins, this function is called as a command and should
    return a widget instance directly.

    Parameters
    ----------
    viewer : napari.viewer.Viewer, optional
        Viewer instance passed by napari.

    Returns
    -------
    CurveAlignWidget
        Configured widget instance.
    """
    from .widget import CurveAlignWidget
    return CurveAlignWidget(viewer=viewer)
