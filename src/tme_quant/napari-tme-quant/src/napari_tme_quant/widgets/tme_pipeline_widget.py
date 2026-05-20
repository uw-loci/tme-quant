"""TMEPipelineWidget — Analysis sub-tab: TME analysis pipelines.

Contains the CurveAlign TACS Pipeline section, which combines curvelet-based
fiber orientation analysis with a pre-computed tumor boundary mask to produce
TACS (Tumor-Associated Collagen Signature) classification.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..controllers.project_controller import IMAGE_FILE_FILTER


class TMEPipelineWidget(QWidget):
    """TME Pipelines sub-tab.

    Currently implements the CurveAlign TACS Pipeline section.
    The Standard TME Pipeline (CT-FIRE + StarDist) section is a future addition.
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._analysis_ctrl = None
        self._project_ctrl  = None
        self._active_image_id: Optional[str] = None
        self._advanced_dialog: Optional[QDialog] = None
        self._build_ui()

    def set_controller(self, analysis_ctrl) -> None:
        self._analysis_ctrl = analysis_ctrl

    def set_project_controller(self, project_ctrl) -> None:
        self._project_ctrl = project_ctrl
        project_ctrl.connect_type_changed(self.on_image_type_changed)
        project_ctrl.connect_image_added(lambda _: self._refresh_fiber_selector())
        project_ctrl.connect_image_removed(lambda _: self._refresh_fiber_selector())

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 6)
        inner_layout.setSpacing(8)

        inner_layout.addWidget(self._build_curvealign_tacs_group())
        inner_layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_curvealign_tacs_group(self) -> QGroupBox:
        group = QGroupBox("CurveAlign TACS Pipeline")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Fiber image selector
        fiber_row = QHBoxLayout()
        self._fiber_selector = QComboBox()
        self._fiber_selector.setMinimumWidth(160)
        refresh_fiber_btn = QPushButton("↺")
        refresh_fiber_btn.setMaximumWidth(28)
        refresh_fiber_btn.setToolTip("Refresh fiber image list from project")
        refresh_fiber_btn.clicked.connect(self._refresh_fiber_selector)
        fiber_row.addWidget(self._fiber_selector, stretch=1)
        fiber_row.addWidget(refresh_fiber_btn)
        form.addRow("Fiber image:", fiber_row)

        # Boundary mask selector
        mask_row = QHBoxLayout()
        self._mask_selector = QComboBox()
        self._mask_selector.setMinimumWidth(120)
        refresh_mask_btn = QPushButton("↺")
        refresh_mask_btn.setMaximumWidth(28)
        refresh_mask_btn.setToolTip("Refresh layer list from viewer")
        refresh_mask_btn.clicked.connect(self._refresh_mask_selector)
        load_mask_btn = QPushButton("Load from file...")
        load_mask_btn.clicked.connect(self._load_mask_from_file)
        mask_row.addWidget(self._mask_selector, stretch=1)
        mask_row.addWidget(refresh_mask_btn)
        mask_row.addWidget(load_mask_btn)
        form.addRow("Boundary mask:", mask_row)

        # Distance threshold (primary control — applies to any TACS pipeline method)
        self._dist_thresh = QDoubleSpinBox()
        self._dist_thresh.setRange(1.0, 2000.0)
        self._dist_thresh.setDecimals(1)
        self._dist_thresh.setSuffix(" px")
        self._dist_thresh.setValue(50.0)
        form.addRow("Dist threshold:", self._dist_thresh)

        # Zone width — read-only display synced to dist_thresh; placeholder for future range support
        self._zone_width = QDoubleSpinBox()
        self._zone_width.setRange(1.0, 2000.0)
        self._zone_width.setDecimals(1)
        self._zone_width.setSuffix(" px")
        self._zone_width.setValue(self._dist_thresh.value())
        self._zone_width.setEnabled(False)
        self._dist_thresh.valueChanged.connect(self._zone_width.setValue)
        form.addRow("Zone width:", self._zone_width)
        layout.addLayout(form)

        # Curvelet params inline
        params_group = QGroupBox("Curvelet params")
        pg = QFormLayout(params_group)
        pg.setLabelAlignment(Qt.AlignRight)

        self._keep   = QDoubleSpinBox(); self._keep.setRange(0.001, 1.0);   self._keep.setDecimals(3);  self._keep.setValue(0.05)
        self._scale  = QSpinBox();       self._scale.setRange(1, 10);                                    self._scale.setValue(1)
        self._radius = QDoubleSpinBox(); self._radius.setRange(0.5, 50.0);  self._radius.setDecimals(1); self._radius.setValue(4.0)
        self._exclude_inside = QCheckBox("Exclude fibers inside mask")

        row1 = QHBoxLayout()
        for lbl, w in [("Keep:", self._keep), ("Scale:", self._scale), ("Radius:", self._radius)]:
            row1.addWidget(QLabel(lbl)); row1.addWidget(w)
        row1.addStretch()
        pg.addRow(row1)
        pg.addRow(self._exclude_inside)

        adv_row = QHBoxLayout()
        adv_row.addStretch()
        adv_btn = QPushButton("Advanced...")
        adv_btn.clicked.connect(self._open_advanced)
        adv_row.addWidget(adv_btn)
        pg.addRow(adv_row)
        layout.addWidget(params_group)

        # Batch scope toggle
        scope_group = QGroupBox("Run scope")
        scope_row = QHBoxLayout(scope_group)
        self._scope_current = QRadioButton("Current image")
        self._scope_all     = QRadioButton("All images in project")
        self._scope_current.setChecked(True)
        scope_row.addWidget(self._scope_current)
        scope_row.addWidget(self._scope_all)
        scope_row.addStretch()
        copy_btn = QPushButton("Copy params to all images")
        copy_btn.clicked.connect(self._copy_params_to_all)
        scope_row.addWidget(copy_btn)
        layout.addWidget(scope_group)

        # Progress bar (hidden until run starts)
        self._progress_bar   = QProgressBar()
        self._progress_bar.setRange(0, 4)
        self._progress_bar.setVisible(False)
        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_label)

        # Run / Commit / Status row
        bottom_row = QHBoxLayout()
        self._run_btn    = QPushButton("Run CurveAlign TACS")
        self._commit_btn = QPushButton("Commit to Hierarchy")
        self._commit_btn.setEnabled(False)
        self._status_lbl = QLabel("○ not run")
        self._run_btn.clicked.connect(self._run_curvealign_tacs)
        self._commit_btn.clicked.connect(self._commit_curvealign)
        bottom_row.addWidget(self._run_btn)
        bottom_row.addWidget(self._commit_btn)
        bottom_row.addStretch()
        bottom_row.addWidget(self._status_lbl)
        layout.addLayout(bottom_row)

        return group

    # ── Controller callbacks ───────────────────────────────────────────────────

    def set_active_image(self, image_id: str) -> None:
        """Called by _main_widget when the active image changes."""
        if self._active_image_id and self._analysis_ctrl:
            self._save_params_for_image(self._active_image_id)
        self._active_image_id = image_id
        self._load_params_for_image(image_id)
        # Auto-select this image in the fiber selector if it is a FIBER type
        if self._project_ctrl:
            from ..controllers.state import ImageType
            itype = self._project_ctrl._state.image_types.get(image_id)
            if itype == ImageType.FIBER:
                idx = self._fiber_selector.findData(image_id)
                if idx >= 0:
                    self._fiber_selector.setCurrentIndex(idx)

    def on_image_type_changed(self, image_id: str, image_type) -> None:
        self._refresh_fiber_selector()

    def on_curvealign_complete(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._status_lbl.setText("● computed (memory)")
        self._commit_btn.setEnabled(True)

    def update_progress(self, step: int, total: int, msg: str) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(step)
        self._progress_label.setText(f"{msg} ({step}/{total})")
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)

    # ── Selector population ────────────────────────────────────────────────────

    def _refresh_fiber_selector(self, *_) -> None:
        if self._project_ctrl is None:
            return
        from ..controllers.state import ImageType
        state = self._project_ctrl._state
        self._fiber_selector.clear()
        for iid, itype in state.image_types.items():
            if itype == ImageType.FIBER:
                self._fiber_selector.addItem(iid, iid)

    def _refresh_mask_selector(self, *_) -> None:
        if self._viewer is None:
            return
        import napari.layers
        self._mask_selector.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, (napari.layers.Image,
                                  napari.layers.Labels,
                                  napari.layers.Shapes)):
                self._mask_selector.addItem(layer.name)
        # Also add project-registered MASK images
        if self._project_ctrl:
            from ..controllers.state import ImageType
            state = self._project_ctrl._state
            existing = {self._mask_selector.itemText(i)
                        for i in range(self._mask_selector.count())}
            for iid, itype in state.image_types.items():
                if itype == ImageType.MASK and iid not in existing:
                    self._mask_selector.addItem(iid)

    def _load_mask_from_file(self) -> None:
        if self._viewer is None or self._project_ctrl is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load boundary mask", "",
                                              IMAGE_FILE_FILTER)
        if not path:
            return
        from ..controllers.state import ImageType
        image_id = self._project_ctrl.add_image(path, ImageType.MASK, self._viewer)
        self._refresh_mask_selector()
        idx = self._mask_selector.findText(image_id)
        if idx >= 0:
            self._mask_selector.setCurrentIndex(idx)

    # ── Run ────────────────────────────────────────────────────────────────────

    def _get_boundary_img(self, image_shape: tuple) -> Optional[np.ndarray]:
        if self._viewer is None:
            return None
        layer_name = self._mask_selector.currentText()
        if not layer_name:
            return None
        import napari.layers
        layer = next((l for l in self._viewer.layers if l.name == layer_name), None)
        if layer is None:
            # Try looking it up in state.images
            if self._project_ctrl:
                arr = self._project_ctrl._state.images.get(layer_name)
                if arr is not None:
                    return arr.astype(np.uint8)
            return None
        if isinstance(layer, napari.layers.Shapes):
            return layer.to_labels(labels_shape=image_shape[:2]).astype(np.uint8)
        return np.asarray(layer.data, dtype=np.uint8)

    def _get_curvealign_params(self) -> dict:
        return {
            "keep":                    self._keep.value(),
            "scale":                   self._scale.value(),
            "radius":                  self._radius.value(),
            "exclude_fibers_in_mask":  self._exclude_inside.isChecked(),
            "distance_threshold":      self._dist_thresh.value(),
        }

    def _run_curvealign_tacs(self) -> None:
        if self._analysis_ctrl is None:
            return
        image_id = self._fiber_selector.currentData()
        if not image_id:
            return
        state = self._analysis_ctrl._state
        image = state.images.get(image_id)
        if image is None:
            return

        boundary_img = self._get_boundary_img(image.shape)
        kwargs = self._get_curvealign_params()
        kwargs["boundary_img"] = boundary_img
        kwargs["tif_boundary"] = 3 if boundary_img is not None else 0

        self._save_params_for_image(image_id)
        self._run_btn.setEnabled(False)
        self._status_lbl.setText("◑ running…")

        image_ids = (
            [image_id] if self._scope_current.isChecked()
            else list(state.image_types.keys())
        )
        for iid in image_ids:
            img = state.images.get(iid, image)
            params = (state.per_image_params
                      .get(iid, {})
                      .get("curvealign_tacs", kwargs))
            run_kwargs = dict(params)
            run_kwargs.setdefault("boundary_img", boundary_img)
            run_kwargs.setdefault("tif_boundary", kwargs["tif_boundary"])
            self._analysis_ctrl.run_fiber_extraction(
                iid, method="curvealign", image=img, **run_kwargs
            )

    def _commit_curvealign(self) -> None:
        if self._analysis_ctrl is None or not self._active_image_id:
            return
        self._analysis_ctrl.commit_fiber_result(
            self._active_image_id, method="curvealign"
        )

    # ── Per-image params ───────────────────────────────────────────────────────

    def _save_params_for_image(self, image_id: str) -> None:
        if self._analysis_ctrl is None:
            return
        params = self._get_curvealign_params()
        (self._analysis_ctrl._state.per_image_params
         .setdefault(image_id, {})["curvealign_tacs"]) = params

    def _load_params_for_image(self, image_id: str) -> None:
        if self._analysis_ctrl is None:
            return
        stored = (self._analysis_ctrl._state.per_image_params
                  .get(image_id, {})
                  .get("curvealign_tacs"))
        if stored is None:
            return
        for attr, key in [
            ("_keep",        "keep"),
            ("_scale",       "scale"),
            ("_radius",      "radius"),
            ("_dist_thresh", "distance_threshold"),
        ]:
            widget = getattr(self, attr, None)
            if widget is not None and key in stored:
                widget.blockSignals(True)
                widget.setValue(stored[key])
                widget.blockSignals(False)
        if "exclude_fibers_in_mask" in stored:
            self._exclude_inside.blockSignals(True)
            self._exclude_inside.setChecked(stored["exclude_fibers_in_mask"])
            self._exclude_inside.blockSignals(False)

    def _copy_params_to_all(self) -> None:
        if self._analysis_ctrl is None:
            return
        params = self._get_curvealign_params()
        state = self._analysis_ctrl._state
        for iid in state.image_types:
            state.per_image_params.setdefault(iid, {})["curvealign_tacs"] = dict(params)

    # ── Advanced dialog ────────────────────────────────────────────────────────

    def _open_advanced(self) -> None:
        if self._advanced_dialog is None:
            self._advanced_dialog = _CurveAlignTACSAdvancedDialog(self)
        self._advanced_dialog.show()
        self._advanced_dialog.raise_()


class _CurveAlignTACSAdvancedDialog(QDialog):
    """Advanced parameter groups for the CurveAlign TACS Pipeline.

    Parameter widgets are stubs; groups are labelled to communicate structure.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CurveAlign TACS — Advanced settings")
        self.setModal(False)
        layout = QVBoxLayout(self)
        for title in ("Transform", "Boundary", "Features", "Output"):
            grp = QGroupBox(title)
            QVBoxLayout(grp).addWidget(
                QLabel(f"({title} params — not yet wired up)")
            )
            layout.addWidget(grp)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.close)
        layout.addWidget(btns)
