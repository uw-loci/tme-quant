"""ProjectWidget — Tab 1: image list, type assignment, and pairing."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt


class ProjectWidget(QWidget):
    """Project tab: manages image list, type assignment, and pairing.

    Image type assignment, auto-pairing, and the "Merge to 2-channel" action
    are planned for a future batch.
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Project\n(not yet implemented)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
