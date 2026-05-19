"""LogWidget — read-only log panel displayed in the Log tab."""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from qtpy.QtGui import QColor, QTextCursor
from qtpy.QtCore import Qt

_LEVEL_COLORS = {
    "INFO":  "#000000",
    "WARN":  "#d07000",
    "ERROR": "#cc0000",
}


class LogWidget(QWidget):
    """Scrolling log panel with color-coded INFO / WARN / ERROR messages."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QHBoxLayout()
        clear_btn = QPushButton("■ Clear")
        clear_btn.setMaximumWidth(80)
        clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(clear_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self._text)

    def append(self, level: str, msg: str) -> None:
        """Append a message with color matching INFO / WARN / ERROR."""
        color = _LEVEL_COLORS.get(level.upper(), "#000000")
        html = f'<span style="color:{color}"><b>[{level.upper()}]</b> {msg}</span>'
        self._text.append(html)
        self._text.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        self._text.clear()
