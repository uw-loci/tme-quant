"""MeasurementsWidget — Results sub-tab: hierarchy tree + data table."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class MeasurementsWidget(QWidget):
    """Measurements sub-tab: hierarchy tree on left, filtered data table on right."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Measurements\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
