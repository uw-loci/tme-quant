"""ROIManagerWidget — Tab 3: spatial annotation hub (optional for all pipelines)."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class ROIManagerWidget(QWidget):
    """ROI Manager tab: draw, import, and manage ROI annotations.

    ROI ↔ napari Shapes layer sync is handled by ROIController (Batch 5).
    This tab is optional — no pipeline requires it to be populated.
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("ROI Manager\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
