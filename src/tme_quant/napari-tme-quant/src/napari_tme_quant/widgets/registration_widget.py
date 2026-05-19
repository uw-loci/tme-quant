"""RegistrationWidget — Analysis sub-tab: image registration (H&E → SHG)."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class RegistrationWidget(QWidget):
    """Registration sub-tab: affine/rigid/deformable multi-modal alignment."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Registration\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
