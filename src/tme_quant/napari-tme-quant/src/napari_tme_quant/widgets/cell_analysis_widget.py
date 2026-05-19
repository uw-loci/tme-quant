"""CellAnalysisWidget — Analysis sub-tab: cell segmentation and tumor detection."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class CellAnalysisWidget(QWidget):
    """Cell Analysis sub-tab: StarDist/Cellpose/threshold segmentation + tumor boundary."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Cell Analysis\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
