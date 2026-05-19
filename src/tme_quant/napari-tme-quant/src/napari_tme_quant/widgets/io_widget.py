"""IOWidget — Results sub-tab: project and parameter import/export."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class IOWidget(QWidget):
    """I/O tab: project save/load, GeoJSON export, fiber metrics export."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("I/O\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
