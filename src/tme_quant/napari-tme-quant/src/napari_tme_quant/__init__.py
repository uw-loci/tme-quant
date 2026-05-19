"""napari-TMEQuant plugin — GUI for the tme_quant library."""

__version__ = "0.1.0"

try:
    from ._main_widget import TMEQuantDockWidget
    __all__ = ["TMEQuantDockWidget"]
except ImportError:
    # Qt not available (headless test environment)
    TMEQuantDockWidget = None  # type: ignore[assignment,misc]
    __all__ = []
