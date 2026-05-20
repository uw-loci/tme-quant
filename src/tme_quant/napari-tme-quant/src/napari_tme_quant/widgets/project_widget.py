"""ProjectWidget — Tab 1: image list, type assignment, and project operations."""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..controllers.project_controller import IMAGE_FILE_FILTER, detect_image_type


class ProjectWidget(QWidget):
    """Project tab: image list, type assignment, and project save/load."""

    _COL_FILE   = 0
    _COL_TYPE   = 1
    _COL_PAIRED = 2
    _COL_STATUS = 3

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._project_ctrl = None
        self._build_ui()

    def set_controller(self, project_ctrl) -> None:
        self._project_ctrl = project_ctrl
        project_ctrl.connect_image_added(lambda _: self.refresh_table())
        project_ctrl.connect_image_removed(lambda _: self.refresh_table())
        project_ctrl.connect_type_changed(lambda *_: self.refresh_table())

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("+ Add Image...")
        self._add_btn.clicked.connect(self._add_image)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_image)
        self._remove_btn.setEnabled(False)
        toolbar.addWidget(self._add_btn)
        toolbar.addWidget(self._remove_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["File", "Type", "Paired With", "Status"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self._table)

        proj_group = QGroupBox("Project")
        proj_row = QHBoxLayout(proj_group)
        self._new_btn  = QPushButton("New")
        self._save_btn = QPushButton("Save Project...")
        self._load_btn = QPushButton("Load Project...")
        for btn in (self._new_btn, self._save_btn, self._load_btn):
            proj_row.addWidget(btn)
        proj_row.addStretch()
        self._new_btn.clicked.connect(self._new_project)
        self._save_btn.clicked.connect(self._save_project)
        self._load_btn.clicked.connect(self._load_project)
        layout.addWidget(proj_group)

    # ── Table population ───────────────────────────────────────────────────────

    def refresh_table(self) -> None:
        if self._project_ctrl is None:
            return
        state = self._project_ctrl._state
        self._table.setRowCount(0)
        for image_id, itype in state.image_types.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            file_item = QTableWidgetItem(image_id)
            file_item.setData(Qt.UserRole, image_id)
            self._table.setItem(row, self._COL_FILE, file_item)
            self._table.setItem(row, self._COL_TYPE, QTableWidgetItem(itype.name.title()))
            pair = state.image_pairs.get(image_id, "—")
            self._table.setItem(row, self._COL_PAIRED, QTableWidgetItem(pair))
            self._table.setItem(row, self._COL_STATUS,
                                QTableWidgetItem(self._status_text(image_id, state)))
        self._remove_btn.setEnabled(self._table.rowCount() > 0)

    def _status_text(self, image_id: str, state) -> str:
        if image_id == state.active_image_id:
            return "● active"
        if image_id in state.curvealign_pipeline_results or image_id in state.fiber_results:
            return "◑ result"
        return "○"

    def on_image_selected(self, image_id: str) -> None:
        self.refresh_table()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._COL_FILE)
            if item and item.data(Qt.UserRole) == image_id:
                # Block signals while programmatically selecting the row so
                # itemSelectionChanged does not re-fire _on_selection_changed,
                # which would call select_image again → infinite recursion.
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                break

    def on_analysis_complete(self, _step: str, _image_id: str) -> None:
        self.refresh_table()

    # ── User interactions ──────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        self._remove_btn.setEnabled(len(rows) > 0)
        if rows and self._project_ctrl:
            image_id = self._table.item(rows[0].row(), self._COL_FILE).data(Qt.UserRole)
            self._project_ctrl.select_image(image_id)

    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        if self._project_ctrl is None:
            return
        item = self._table.item(row, self._COL_FILE)
        if item is None:
            return
        image_id = item.data(Qt.UserRole)
        current = self._project_ctrl._state.image_types.get(image_id)
        new_type = self._ask_image_type(current)
        if new_type is not None:
            self._project_ctrl.set_image_type(image_id, new_type)

    def _add_image(self) -> None:
        if self._viewer is None or self._project_ctrl is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Add image", "", IMAGE_FILE_FILTER)
        if not path:
            return
        detected = detect_image_type(path)
        confirmed = self._ask_image_type(detected, title="Confirm image type")
        if confirmed is None:
            return
        self._project_ctrl.add_image(path, confirmed, self._viewer)

    def _remove_image(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows or self._project_ctrl is None:
            return
        image_id = self._table.item(rows[0].row(), self._COL_FILE).data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Remove image",
            f"Remove '{image_id}' from the project?\n"
            "Associated napari layers will also be removed.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._project_ctrl.remove_image(image_id, self._viewer)

    # ── Type-selector dialog ───────────────────────────────────────────────────

    def _ask_image_type(self, default=None, title: str = "Image type") -> Optional[object]:
        from ..controllers.state import ImageType
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        combo = QComboBox()
        type_options = [
            (ImageType.FIBER,       "Fiber (SHG, PLM, bright-field collagen, etc.)"),
            (ImageType.CELL,        "Cell (H&E, DAPI, DAB, etc.)"),
            (ImageType.MASK,        "Mask / Boundary annotation"),
            (ImageType.TWO_CHANNEL, "Pre-registered 2-channel"),
            (ImageType.UNKNOWN,     "Unknown"),
        ]
        for itype, label in type_options:
            combo.addItem(label, itype)
        if default is not None:
            for i, (itype, _) in enumerate(type_options):
                if itype == default:
                    combo.setCurrentIndex(i)
                    break
        form.addRow("Type:", combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() == QDialog.Accepted:
            return combo.currentData()
        return None

    # ── Project operations ─────────────────────────────────────────────────────

    def _new_project(self) -> None:
        if self._project_ctrl is None:
            return
        reply = QMessageBox.question(
            self, "New project",
            "Clear the current project? Unsaved results will be lost.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        state = self._project_ctrl._state
        if self._viewer:
            registered = set(state.image_types)
            to_remove = [
                l for l in list(self._viewer.layers)
                if any(l.name == iid or l.name.startswith(f"{iid} :: ")
                       for iid in registered)
            ]
            for layer in to_remove:
                self._viewer.layers.remove(layer)
        state.image_types.clear()
        state.image_paths.clear()
        state.images.clear()
        state.per_image_params.clear()
        state.reset()
        self.refresh_table()

    def _save_project(self) -> None:
        if self._project_ctrl is None:
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Save project to folder")
        if not save_dir:
            return
        try:
            from ..utils.io_utils import save_plugin_state
            from pathlib import Path
            save_plugin_state(self._project_ctrl._state, Path(save_dir))
            QMessageBox.information(self, "Saved", f"Project saved to:\n{save_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_project(self) -> None:
        if self._project_ctrl is None or self._viewer is None:
            return
        load_dir = QFileDialog.getExistingDirectory(self, "Load project from folder")
        if not load_dir:
            return
        try:
            from ..utils.io_utils import load_plugin_state
            from pathlib import Path
            load_plugin_state(
                self._project_ctrl._state, Path(load_dir), self._viewer
            )
            self.refresh_table()
            QMessageBox.information(self, "Loaded", f"Project loaded from:\n{load_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
