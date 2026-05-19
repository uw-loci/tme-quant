"""TMEPipelineWidget — Analysis sub-tab: fiber–cell–tumor interaction analysis."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class TMEPipelineWidget(QWidget):
    """TME Pipelines sub-tab: interaction detection, per-fiber metrics, network analysis."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("TME Pipelines\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
