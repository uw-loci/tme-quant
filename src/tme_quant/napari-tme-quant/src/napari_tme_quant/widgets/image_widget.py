"""ImageWidget — Tab 2: active image metadata and analysis status."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ImageWidget(QWidget):
    """Image tab: file metadata, pixel size (editable), and per-step status chips."""

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._project_ctrl = None
        self._active_image_id: str | None = None
        self._build_ui()

    def set_controller(self, project_ctrl) -> None:
        self._project_ctrl = project_ctrl

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Metadata group
        meta_group = QGroupBox("Image metadata")
        meta_form = QFormLayout(meta_group)

        self._lbl_file  = QLabel("—")
        self._lbl_path  = QLabel("—")
        self._lbl_path.setWordWrap(True)
        self._lbl_dims  = QLabel("—")
        self._lbl_type  = QLabel("—")

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.001, 1000.0)
        self._pixel_size.setDecimals(3)
        self._pixel_size.setSuffix(" µm/px")
        self._pixel_size.setValue(1.0)
        self._pixel_size.valueChanged.connect(self._on_pixel_size_changed)

        meta_form.addRow("File:",       self._lbl_file)
        meta_form.addRow("Path:",       self._lbl_path)
        meta_form.addRow("Dimensions:", self._lbl_dims)
        meta_form.addRow("Type:",       self._lbl_type)
        meta_form.addRow("Pixel size:", self._pixel_size)
        layout.addWidget(meta_group)

        # Analysis status group
        status_group = QGroupBox("Analysis status")
        status_form = QFormLayout(status_group)
        self._chip_ctfire    = QLabel("○ not run")
        self._chip_curvealign = QLabel("○ not run")
        status_form.addRow("CT-FIRE:",    self._chip_ctfire)
        status_form.addRow("CurveAlign TACS:", self._chip_curvealign)
        layout.addWidget(status_group)

        layout.addStretch()

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_image_selected(self, image_id: str) -> None:
        """Called by _main_widget whenever the active image changes."""
        self._active_image_id = image_id
        self._refresh(image_id)

    def on_analysis_complete(self, step: str, image_id: str) -> None:
        """Refresh status chips when an analysis step finishes."""
        if image_id == self._active_image_id:
            self._refresh_status_chips(image_id)

    # ── Internal refresh ───────────────────────────────────────────────────────

    def _refresh(self, image_id: str) -> None:
        if self._project_ctrl is None:
            return
        state = self._project_ctrl._state

        self._lbl_file.setText(image_id)
        path = state.image_paths.get(image_id, "—")
        self._lbl_path.setText(path)

        arr = state.images.get(image_id)
        if arr is not None:
            shape_str = " × ".join(str(d) for d in arr.shape)
            self._lbl_dims.setText(f"{shape_str}  ({arr.dtype})")
        else:
            self._lbl_dims.setText("—")

        itype = state.image_types.get(image_id)
        self._lbl_type.setText(itype.name.title() if itype else "—")

        # Restore stored pixel size for this image
        stored_px = (state.per_image_params
                     .get(image_id, {})
                     .get("image", {})
                     .get("pixel_size"))
        if stored_px is not None:
            self._pixel_size.blockSignals(True)
            self._pixel_size.setValue(float(stored_px))
            self._pixel_size.blockSignals(False)

        self._refresh_status_chips(image_id)

    def _refresh_status_chips(self, image_id: str) -> None:
        if self._project_ctrl is None:
            return
        state = self._project_ctrl._state

        if image_id in state.fiber_results:
            self._chip_ctfire.setText("● computed (memory)")
        else:
            self._chip_ctfire.setText("○ not run")

        if image_id in state.curvealign_pipeline_results:
            self._chip_curvealign.setText("● computed (memory)")
        else:
            self._chip_curvealign.setText("○ not run")

    def _on_pixel_size_changed(self, value: float) -> None:
        if self._active_image_id and self._project_ctrl:
            state = self._project_ctrl._state
            state.per_image_params \
                .setdefault(self._active_image_id, {}) \
                .setdefault("image", {})["pixel_size"] = value
