"""FiberAnalysisWidget — Analysis → Fiber sub-tab.

Exposes CT-FIRE fiber extraction and CurveAlign (curvelets mode) pipeline.
All analysis is dispatched through AnalysisController; this widget is pure UI.

Layout per CLAUDE_NAPARI.md §Fiber Analysis sub-tab.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QCheckBox, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QDialog, QDialogButtonBox,
    QScrollArea, QSizePolicy, QFileDialog,
)
from qtpy.QtCore import Qt

from tme_quant import CTFireParams, CurveAlignParams, FiberFeatureParams
from tme_quant.fiber_analysis.utils.ctfire_utils import ctfire_backend_status


# ── Status chip label texts ────────────────────────────────────────────────────
_STATUS_NOT_RUN = "○ not run"
_STATUS_CACHED  = "◑ cached (disk)"
_STATUS_MEMORY  = "● computed (memory)"


class FiberAnalysisWidget(QWidget):
    """Fiber Analysis sub-tab: CT-FIRE and CurveAlign (curvelets mode) panels.

    Method selector shows/hides the matching QGroupBox (G3 rule).
    At most 8 params inline; remaining params in a lazy Advanced QDialog (G2 rule).
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._analysis_controller = None   # wired up by _main_widget in Batch 8
        self._active_image_id: Optional[str] = None
        self._advanced_ctfire_dialog: Optional[QDialog] = None
        self._advanced_curvealign_dialog: Optional[QDialog] = None
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_controller(self, controller) -> None:
        """Wire up the AnalysisController after it is created."""
        self._analysis_controller = controller

    def set_active_image(self, image_id: str) -> None:
        self._active_image_id = image_id

    def update_status(self, step: str, status: str) -> None:
        """Update the status chip.  step: 'ctfire' | 'curvealign'."""
        text = {
            "not_run": _STATUS_NOT_RUN,
            "cached":  _STATUS_CACHED,
            "memory":  _STATUS_MEMORY,
        }.get(status, status)
        if step == "ctfire":
            self._ctfire_status.setText(text)
        elif step == "curvealign":
            self._ca_status.setText(text)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Input image selector
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input image:"))
        self._image_selector = QComboBox()
        self._image_selector.setToolTip("Fiber or 2-channel images only")
        input_row.addWidget(self._image_selector, stretch=1)
        layout.addLayout(input_row)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["CurveAlign (curvelets mode)", "CT-FIRE"])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, stretch=1)
        layout.addLayout(method_row)

        # CurveAlign group box (default method — shown first)
        self._curvealign_group = self._build_curvealign_group()
        layout.addWidget(self._curvealign_group)

        # CT-FIRE group box (initially hidden)
        self._ctfire_group = self._build_ctfire_group()
        self._ctfire_group.setVisible(False)
        layout.addWidget(self._ctfire_group)

        layout.addStretch()

    def _on_method_changed(self, index: int) -> None:
        self._curvealign_group.setVisible(index == 0)
        self._ctfire_group.setVisible(index == 1)

    # ── CT-FIRE group ──────────────────────────────────────────────────────────

    def _build_ctfire_group(self) -> QGroupBox:
        group = QGroupBox("CT-FIRE")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._ctfire_threshold = QDoubleSpinBox()
        self._ctfire_threshold.setRange(0.0, 1.0)
        self._ctfire_threshold.setSingleStep(0.01)
        self._ctfire_threshold.setValue(0.1)
        self._ctfire_threshold.setToolTip("Curvelet energy threshold for fiber detection")
        form.addRow("Threshold:", self._ctfire_threshold)

        self._ctfire_min_len = QSpinBox()
        self._ctfire_min_len.setRange(1, 10000)
        self._ctfire_min_len.setValue(10)
        self._ctfire_min_len.setSuffix(" px")
        self._ctfire_min_len.setToolTip("Minimum fiber length in pixels")
        form.addRow("Min fiber length:", self._ctfire_min_len)

        self._ctfire_max_len = QSpinBox()
        self._ctfire_max_len.setRange(1, 100000)
        self._ctfire_max_len.setValue(1000)
        self._ctfire_max_len.setSuffix(" px")
        self._ctfire_max_len.setToolTip("Maximum fiber length in pixels")
        form.addRow("Max fiber length:", self._ctfire_max_len)

        self._ctfire_pixel_size = QDoubleSpinBox()
        self._ctfire_pixel_size.setRange(0.001, 1000.0)
        self._ctfire_pixel_size.setSingleStep(0.1)
        self._ctfire_pixel_size.setValue(1.0)
        self._ctfire_pixel_size.setSuffix(" µm/px")
        form.addRow("Pixel size:", self._ctfire_pixel_size)

        self._ctfire_spur_len = QSpinBox()
        self._ctfire_spur_len.setRange(0, 100)
        self._ctfire_spur_len.setValue(3)
        self._ctfire_spur_len.setSuffix(" px")
        self._ctfire_spur_len.setToolTip("Spur branches shorter than this are pruned")
        form.addRow("Spur prune length:", self._ctfire_spur_len)

        self._ctfire_close_radius = QSpinBox()
        self._ctfire_close_radius.setRange(0, 50)
        self._ctfire_close_radius.setValue(3)
        self._ctfire_close_radius.setSuffix(" px")
        form.addRow("Mask closing radius:", self._ctfire_close_radius)

        self._ctfire_n_levels = QSpinBox()
        self._ctfire_n_levels.setRange(1, 16)
        self._ctfire_n_levels.setValue(4)
        self._ctfire_n_levels.setToolTip("Number of curvelet scale levels")
        form.addRow("N levels:", self._ctfire_n_levels)

        self._ctfire_n_angles = QSpinBox()
        self._ctfire_n_angles.setRange(4, 64)
        self._ctfire_n_angles.setValue(16)
        self._ctfire_n_angles.setSingleStep(4)
        self._ctfire_n_angles.setToolTip("Number of curvelet orientation angles")
        form.addRow("N angles:", self._ctfire_n_angles)

        vbox.addLayout(form)

        # 3D toggle — disabled until C++ 3D FIRE is compiled
        status = ctfire_backend_status()
        self._ctfire_3d = QCheckBox("3D CT-FIRE (requires C++ extension)")
        self._ctfire_3d.setEnabled(status.get("3d_supported", False))
        self._ctfire_3d.setToolTip(
            "Enable 3D fiber extraction. Available only when the C++ FIRE "
            "extension is compiled and 3d_supported=True."
        )
        vbox.addWidget(self._ctfire_3d)

        # Advanced + bottom row
        vbox.addLayout(self._build_ctfire_bottom_row())
        return group

    def _build_ctfire_bottom_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        advanced_btn = QPushButton("Advanced…")
        advanced_btn.clicked.connect(self._open_ctfire_advanced)
        row.addWidget(advanced_btn)

        row.addStretch()

        self._ctfire_status = QLabel(_STATUS_NOT_RUN)
        row.addWidget(self._ctfire_status)

        self._ctfire_write_roi = QCheckBox("Write results to ROI Manager")
        row.addWidget(self._ctfire_write_roi)

        self._ctfire_run_btn = QPushButton("Run CT-FIRE")
        self._ctfire_run_btn.setMinimumWidth(120)
        self._ctfire_run_btn.clicked.connect(self._run_ctfire)
        row.addWidget(self._ctfire_run_btn)

        self._ctfire_reset_btn = QPushButton("Reset")
        self._ctfire_reset_btn.setToolTip("Clear CT-FIRE result and remove layers")
        self._ctfire_reset_btn.clicked.connect(self._reset_ctfire)
        row.addWidget(self._ctfire_reset_btn)

        self._ctfire_abort_btn = QPushButton("Abort")
        self._ctfire_abort_btn.setEnabled(False)
        self._ctfire_abort_btn.setToolTip("Stop the running CT-FIRE analysis")
        self._ctfire_abort_btn.clicked.connect(self._abort_analysis)
        row.addWidget(self._ctfire_abort_btn)

        self._ctfire_commit_btn = QPushButton("Commit to Hierarchy")
        self._ctfire_commit_btn.setEnabled(False)
        self._ctfire_commit_btn.clicked.connect(self._commit_ctfire)
        row.addWidget(self._ctfire_commit_btn)

        return row

    def _open_ctfire_advanced(self) -> None:
        """Lazily create and open the CT-FIRE Advanced Settings dialog."""
        if self._advanced_ctfire_dialog is None:
            self._advanced_ctfire_dialog = _CTFireAdvancedDialog(self)
        self._advanced_ctfire_dialog.exec()

    def _run_ctfire(self) -> None:
        if self._analysis_controller is None:
            return
        self._analysis_controller.run_fiber_extraction(
            self._active_image_id, method="ctfire"
        )

    def _reset_ctfire(self) -> None:
        if self._analysis_controller is None:
            return
        image_id = self._active_image_id or self._image_selector.currentData()
        if image_id:
            self._analysis_controller.invalidate_result(image_id, "ctfire")

    def _commit_ctfire(self) -> None:
        if self._analysis_controller is None:
            return
        self._analysis_controller.commit_fiber_result(self._active_image_id)

    def on_ctfire_complete(self) -> None:
        """Called by AnalysisController when CT-FIRE finishes."""
        self._ctfire_run_btn.setEnabled(True)
        self._ctfire_reset_btn.setEnabled(True)
        self._ctfire_abort_btn.setEnabled(False)
        self._ctfire_status.setText(_STATUS_MEMORY)
        self._ctfire_commit_btn.setEnabled(True)

    # ── CurveAlign group ───────────────────────────────────────────────────────

    def _build_curvealign_group(self) -> QGroupBox:
        group = QGroupBox("CurveAlign (curvelets mode)")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(4)

        # Pre-computed fiber_structure option (G6 rule)
        precomputed_row = QHBoxLayout()
        self._ca_precomputed = QCheckBox("Use pre-computed fiber_structure")
        self._ca_precomputed.toggled.connect(self._on_precomputed_toggled)
        self._ca_load_btn = QPushButton("Load CSV/XLSX…")
        self._ca_load_btn.setEnabled(False)
        self._ca_load_btn.clicked.connect(self._load_fiber_structure)
        precomputed_row.addWidget(self._ca_precomputed)
        precomputed_row.addWidget(self._ca_load_btn)
        precomputed_row.addStretch()
        vbox.addLayout(precomputed_row)

        # Boundary alignment option
        self._ca_boundary = QCheckBox("Analyze boundary alignment")
        self._ca_boundary.toggled.connect(self._on_boundary_toggled)
        vbox.addWidget(self._ca_boundary)

        self._ca_boundary_opts = QWidget()
        boundary_form = QFormLayout(self._ca_boundary_opts)
        boundary_form.setContentsMargins(16, 0, 0, 0)
        self._ca_mask_selector = QComboBox()
        self._ca_mask_selector.setToolTip("Select a mask layer for boundary extraction")
        boundary_form.addRow("Mask layer:", self._ca_mask_selector)
        self._ca_zone_width = QDoubleSpinBox()
        self._ca_zone_width.setRange(0.1, 10000.0)
        self._ca_zone_width.setValue(50.0)
        self._ca_zone_width.setSuffix(" µm")
        self._ca_zone_width.setToolTip("Width of the boundary analysis zone")
        boundary_form.addRow("Zone width:", self._ca_zone_width)
        self._ca_boundary_opts.setVisible(False)
        vbox.addWidget(self._ca_boundary_opts)

        # Curvelet params (inline — 7 params)
        vbox.addWidget(QLabel("─── Curvelet params ───"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._ca_keep = QDoubleSpinBox()
        self._ca_keep.setRange(0.001, 1.0)
        self._ca_keep.setSingleStep(0.01)
        self._ca_keep.setValue(0.05)
        self._ca_keep.setToolTip("Fraction of curvelet coefficients to keep")
        form.addRow("Keep:", self._ca_keep)

        self._ca_scale = QSpinBox()
        self._ca_scale.setRange(1, 16)
        self._ca_scale.setValue(1)
        form.addRow("Scale:", self._ca_scale)

        self._ca_radius = QDoubleSpinBox()
        self._ca_radius.setRange(0.5, 100.0)
        self._ca_radius.setSingleStep(0.5)
        self._ca_radius.setValue(4.0)
        self._ca_radius.setSuffix(" px")
        self._ca_radius.setToolTip("Spatial grouping radius for fiber candidates")
        form.addRow("Radius:", self._ca_radius)

        self._ca_pixel_size = QDoubleSpinBox()
        self._ca_pixel_size.setRange(0.001, 1000.0)
        self._ca_pixel_size.setSingleStep(0.1)
        self._ca_pixel_size.setValue(1.0)
        self._ca_pixel_size.setSuffix(" µm/px")
        form.addRow("Pixel size:", self._ca_pixel_size)

        self._ca_dist_threshold = QDoubleSpinBox()
        self._ca_dist_threshold.setRange(0.0, 10000.0)
        self._ca_dist_threshold.setValue(50.0)
        self._ca_dist_threshold.setSuffix(" px")
        self._ca_dist_threshold.setToolTip("Max distance from ROI boundary to include fibers")
        form.addRow("Dist threshold:", self._ca_dist_threshold)

        self._ca_min_weight = QDoubleSpinBox()
        self._ca_min_weight.setRange(0.0, 1.0)
        self._ca_min_weight.setSingleStep(0.01)
        self._ca_min_weight.setValue(0.0)
        self._ca_min_weight.setToolTip("Minimum fiber weight threshold")
        form.addRow("Min fiber weight:", self._ca_min_weight)

        self._ca_exclude_mask = QCheckBox("Exclude fibers inside mask")
        form.addRow("", self._ca_exclude_mask)

        vbox.addLayout(form)

        # Advanced + bottom row
        vbox.addLayout(self._build_curvealign_bottom_row())
        return group

    def _build_curvealign_bottom_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        advanced_btn = QPushButton("Advanced…")
        advanced_btn.clicked.connect(self._open_curvealign_advanced)
        row.addWidget(advanced_btn)

        row.addStretch()

        self._ca_status = QLabel(_STATUS_NOT_RUN)
        row.addWidget(self._ca_status)

        self._ca_write_roi = QCheckBox("Write results to ROI Manager")
        row.addWidget(self._ca_write_roi)

        self._ca_run_btn = QPushButton("Run CurveAlign")
        self._ca_run_btn.setMinimumWidth(120)
        self._ca_run_btn.clicked.connect(self._run_curvealign)
        row.addWidget(self._ca_run_btn)

        self._ca_reset_btn = QPushButton("Reset")
        self._ca_reset_btn.setToolTip("Clear CurveAlign result and remove layers")
        self._ca_reset_btn.clicked.connect(self._reset_curvealign)
        row.addWidget(self._ca_reset_btn)

        self._ca_abort_btn = QPushButton("Abort")
        self._ca_abort_btn.setEnabled(False)
        self._ca_abort_btn.setToolTip("Stop the running CurveAlign analysis")
        self._ca_abort_btn.clicked.connect(self._abort_analysis)
        row.addWidget(self._ca_abort_btn)

        self._ca_commit_btn = QPushButton("Commit to Hierarchy")
        self._ca_commit_btn.setEnabled(False)
        self._ca_commit_btn.clicked.connect(self._commit_curvealign)
        row.addWidget(self._ca_commit_btn)

        return row

    def _on_precomputed_toggled(self, checked: bool) -> None:
        self._ca_load_btn.setEnabled(checked)

    def _on_boundary_toggled(self, checked: bool) -> None:
        self._ca_boundary_opts.setVisible(checked)

    def _load_fiber_structure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load fiber_structure", "", "CSV/XLSX (*.csv *.xlsx *.xls)"
        )
        if path:
            self._ca_load_btn.setText(path.split("/")[-1])
            self._ca_load_btn.setProperty("loaded_path", path)

    def _open_curvealign_advanced(self) -> None:
        if self._advanced_curvealign_dialog is None:
            self._advanced_curvealign_dialog = _CurveAlignAdvancedDialog(self)
        self._advanced_curvealign_dialog.exec()

    def _run_curvealign(self) -> None:
        if self._analysis_controller is None:
            return
        self._analysis_controller.run_fiber_extraction(
            self._active_image_id, method="curvealign"
        )

    def _reset_curvealign(self) -> None:
        if self._analysis_controller is None:
            return
        image_id = self._active_image_id or self._image_selector.currentData()
        if image_id:
            self._analysis_controller.invalidate_result(image_id, "curvealign")

    def _abort_analysis(self) -> None:
        if self._analysis_controller is None:
            return
        self._analysis_controller.abort()

    def _commit_curvealign(self) -> None:
        if self._analysis_controller is None:
            return
        self._analysis_controller.commit_fiber_result(
            self._active_image_id, method="curvealign"
        )

    def on_curvealign_complete(self) -> None:
        """Called by AnalysisController when CurveAlign pipeline finishes."""
        self._ca_run_btn.setEnabled(True)
        self._ca_reset_btn.setEnabled(True)
        self._ca_abort_btn.setEnabled(False)
        self._ca_status.setText(_STATUS_MEMORY)
        self._ca_commit_btn.setEnabled(True)

    def on_analysis_started(self, step: str, image_id: str) -> None:
        """Disable Run/Reset/Commit and enable Abort for the running step."""
        if step in ("curvealign",):
            self._ca_run_btn.setEnabled(False)
            self._ca_reset_btn.setEnabled(False)
            self._ca_abort_btn.setEnabled(True)
            self._ca_commit_btn.setEnabled(False)
            self._ca_status.setText("● running…")
        if step in ("fiber", "ctfire"):
            self._ctfire_run_btn.setEnabled(False)
            self._ctfire_reset_btn.setEnabled(False)
            self._ctfire_abort_btn.setEnabled(True)
            self._ctfire_commit_btn.setEnabled(False)
            self._ctfire_status.setText("● running…")

    def on_analysis_aborted(self, step: str, image_id: str) -> None:
        """Re-enable Run/Reset; disable Abort; restore Commit if result still exists."""
        state = getattr(self._analysis_controller, "_state", None)

        if step in ("curvealign",):
            self._ca_run_btn.setEnabled(True)
            self._ca_reset_btn.setEnabled(True)
            self._ca_abort_btn.setEnabled(False)
            has_result = bool(
                state and image_id in state.curvealign_pipeline_results
            )
            self._ca_commit_btn.setEnabled(has_result)
            self._ca_status.setText(_STATUS_MEMORY if has_result else _STATUS_NOT_RUN)

        if step in ("fiber", "ctfire"):
            self._ctfire_run_btn.setEnabled(True)
            self._ctfire_reset_btn.setEnabled(True)
            self._ctfire_abort_btn.setEnabled(False)
            has_result = bool(state and image_id in state.fiber_results)
            self._ctfire_commit_btn.setEnabled(has_result)
            self._ctfire_status.setText(_STATUS_MEMORY if has_result else _STATUS_NOT_RUN)

    # ── Param accessors ────────────────────────────────────────────────────────

    def get_ctfire_params(self) -> CTFireParams:
        """Read inline CT-FIRE widget values into a CTFireParams dataclass."""
        return CTFireParams(
            ctfire_threshold=self._ctfire_threshold.value(),
            min_fiber_length=self._ctfire_min_len.value(),
            max_fiber_length=self._ctfire_max_len.value(),
            pixel_size=self._ctfire_pixel_size.value(),
            spur_length_px=self._ctfire_spur_len.value(),
            mask_closing_radius=self._ctfire_close_radius.value(),
            ctfire_n_levels=self._ctfire_n_levels.value(),
            ctfire_n_angles=self._ctfire_n_angles.value(),
        )

    def get_curvealign_params(self) -> dict:
        """Collect CurveAlign pipeline kwargs from the inline controls."""
        kwargs: dict = {
            "keep": self._ca_keep.value(),
            "scale": self._ca_scale.value(),
            "radius": self._ca_radius.value(),
            "pixel_size": self._ca_pixel_size.value(),
            "distance_threshold": self._ca_dist_threshold.value(),
            "exclude_fibers_in_mask": self._ca_exclude_mask.isChecked(),
            "min_fiber_weight": self._ca_min_weight.value(),
        }
        if self._ca_boundary.isChecked():
            kwargs["tif_boundary"] = 1
        else:
            kwargs["boundary_img"] = None
            kwargs["tif_boundary"] = 0

        if self._ca_precomputed.isChecked():
            loaded_path = self._ca_load_btn.property("loaded_path")
            if loaded_path:
                kwargs["_precomputed_path"] = loaded_path

        return kwargs


# ── Advanced dialogs ───────────────────────────────────────────────────────────

class _CTFireAdvancedDialog(QDialog):
    """Advanced CT-FIRE parameters in four collapsible QGroupBox sections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CT-FIRE — Advanced Settings")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        for title in ("Transform", "Tracing", "Filters", "Measurements"):
            box = QGroupBox(title)
            box_layout = QFormLayout(box)
            box_layout.addRow(QLabel(f"({title} params — not yet wired up)"))
            inner_layout.addWidget(box)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class _CurveAlignAdvancedDialog(QDialog):
    """Advanced CurveAlign parameters in four collapsible QGroupBox sections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CurveAlign — Advanced Settings")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        for title in ("Transform", "Boundary", "Features", "Output"):
            box = QGroupBox(title)
            box_layout = QFormLayout(box)
            box_layout.addRow(QLabel(f"({title} params — not yet wired up)"))
            inner_layout.addWidget(box)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
