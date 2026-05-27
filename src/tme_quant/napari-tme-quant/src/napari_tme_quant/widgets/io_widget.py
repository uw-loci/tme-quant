"""IOWidget — Results sub-tab: project save/load."""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QMessageBox,
)
from qtpy.QtCore import Qt


class IOWidget(QWidget):
    """I/O tab: project save/load, GeoJSON export, fiber metrics export."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._analysis_ctrl = None
        self._proj_ctrl = None
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_controller(self, controller) -> None:
        self._analysis_ctrl = controller

    def set_project_controller(self, controller) -> None:
        self._proj_ctrl = controller

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Project group ─────────────────────────────────────────────────────
        proj_group = QGroupBox("Project")
        proj_layout = QVBoxLayout(proj_group)

        self._save_dir_lbl = QLabel("Save dir: (not set)")
        self._save_dir_lbl.setWordWrap(True)
        proj_layout.addWidget(self._save_dir_lbl)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Project")
        save_btn.setToolTip(
            "Save analysis results, parameters and image paths to the project folder.\n"
            "Uses the project folder set in the Project tab, or prompts for one."
        )
        save_btn.clicked.connect(self._save_project)

        load_btn = QPushButton("Load Project…")
        load_btn.setToolTip(
            "Load a previously saved project folder.\n"
            "Restores analysis results so the Results tab is populated without re-running."
        )
        load_btn.clicked.connect(self._load_project)

        btn_row.addWidget(save_btn)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()
        proj_layout.addLayout(btn_row)
        layout.addWidget(proj_group)

        # ── Export group (stub) ───────────────────────────────────────────────
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(QLabel("GeoJSON / metrics export\n(coming soon)"))
        layout.addWidget(export_group)

        layout.addStretch()

    # ── Save ───────────────────────────────────────────────────────────────────

    def _save_project(self) -> None:
        state = self._get_state()
        if state is None:
            return

        save_dir = getattr(state, "project_dir", None)
        if save_dir is None:
            save_dir = self._pick_directory("Choose project save folder")
            if save_dir is None:
                return
            from pathlib import Path
            state.project_dir = Path(save_dir)

        try:
            from ..utils.io_utils import save_plugin_state
            from pathlib import Path
            save_plugin_state(state, Path(save_dir))
            self._save_dir_lbl.setText(f"Saved to: {save_dir}")
            print(f"[io] Project saved → {save_dir}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save failed", str(exc))

    # ── Load ───────────────────────────────────────────────────────────────────

    def _load_project(self) -> None:
        state = self._get_state()
        if state is None:
            return

        load_dir = self._pick_directory("Choose project folder to load")
        if load_dir is None:
            return

        try:
            from ..utils.io_utils import load_plugin_state
            from pathlib import Path
            load_plugin_state(state, Path(load_dir), self._viewer)
            from pathlib import Path
            state.project_dir = Path(load_dir)
            self._save_dir_lbl.setText(f"Loaded from: {load_dir}")
            print(f"[io] Project loaded ← {load_dir}", flush=True)
            self._notify_widgets_after_load(state)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load failed", str(exc))

    def _notify_widgets_after_load(self, state) -> None:
        """Fire analysis_complete for every restored result so widgets update."""
        if self._analysis_ctrl is None:
            return
        for image_id, result in state.curvealign_pipeline_results.items():
            self._analysis_ctrl.notify_result_loaded("curvealign", image_id, result)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_state(self):
        if self._analysis_ctrl is None:
            QMessageBox.warning(self, "Not ready", "Plugin not fully initialised yet.")
            return None
        return self._analysis_ctrl._state

    def _pick_directory(self, caption: str) -> Optional[str]:
        path = QFileDialog.getExistingDirectory(self, caption, "")
        return path if path else None
