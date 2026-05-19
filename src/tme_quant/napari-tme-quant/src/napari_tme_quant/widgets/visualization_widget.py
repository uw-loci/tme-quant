"""VisualizationWidget — Results sub-tab: overlay controls and docked plots.

Three-tier output strategy (CLAUDE_NAPARI.md §Visualization sub-widget):
  1. Spatial overlays   → napari Shapes / Points layers (managed by VisualizationController)
  2. Summary plots      → docked napari-matplotlib panel
  3. Exported PNGs      → QDialog with QLabel on demand

CurveAlign TACS View is shown only when PluginState.curvealign_pipeline_results
contains a result for the active image.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QComboBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter,
    QAbstractItemView, QHeaderView,
)
from qtpy.QtCore import Qt

from ..utils.layer_utils import TACS_COLORS


class VisualizationWidget(QWidget):
    """Visualization sub-tab: overlay checkboxes, docked matplotlib plots,
    and the CurveAlign TACS View panel.
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._viz_controller = None        # wired up by _main_widget
        self._curvealign_result = None     # set by on_curvealign_committed()
        self._active_image_id: Optional[str] = None
        self._build_ui()

    def set_controller(self, controller) -> None:
        self._viz_controller = controller

    def set_active_image(self, image_id: str) -> None:
        self._active_image_id = image_id

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Overlay controls ──────────────────────────────────────────────────
        overlay_group = QGroupBox("Overlays")
        overlay_row = QHBoxLayout(overlay_group)
        self._cb_fibers = QCheckBox("Fibers")
        self._cb_fibers.setChecked(True)
        self._cb_rois = QCheckBox("ROIs")
        self._cb_rois.setChecked(True)
        self._cb_tacs = QCheckBox("TACS zones")
        self._cb_tacs.setChecked(True)
        self._cb_cells = QCheckBox("Cells")
        overlay_row.addWidget(self._cb_fibers)
        overlay_row.addWidget(self._cb_rois)
        overlay_row.addWidget(self._cb_tacs)
        overlay_row.addWidget(self._cb_cells)
        overlay_row.addStretch()
        color_lbl = QLabel("Color by:")
        self._color_combo = QComboBox()
        self._color_combo.addItems(["TACS type", "Object type", "ROI", "Uniform"])
        overlay_row.addWidget(color_lbl)
        overlay_row.addWidget(self._color_combo)
        layout.addWidget(overlay_group)

        # ── Docked plots selector ─────────────────────────────────────────────
        plots_group = QGroupBox("Plots (docked napari-matplotlib)")
        plots_row = QHBoxLayout(plots_group)
        self._plot_combo = QComboBox()
        self._plot_combo.addItems([
            "Orientation Heatmap",
            "TACS Distribution",
            "Angle Histogram",
            "Network Graph",
        ])
        show_plot_btn = QPushButton("Show")
        show_plot_btn.clicked.connect(self._show_plot)
        plots_row.addWidget(self._plot_combo, stretch=1)
        plots_row.addWidget(show_plot_btn)
        layout.addWidget(plots_group)

        # ── Exported figures ──────────────────────────────────────────────────
        fig_group = QGroupBox("Exported figures")
        fig_layout = QVBoxLayout(fig_group)
        view_overlay_btn = QPushButton("View Overlay PNG in window")
        view_heatmap_btn = QPushButton("View Density Heatmap in window")
        view_overlay_btn.clicked.connect(self._view_overlay_png)
        view_heatmap_btn.clicked.connect(self._view_heatmap_png)
        fig_layout.addWidget(view_overlay_btn)
        fig_layout.addWidget(view_heatmap_btn)
        layout.addWidget(fig_group)

        # ── CurveAlign TACS View (hidden until result available) ──────────────
        self._tacs_view_group = self._build_curvealign_tacs_view()
        self._tacs_view_group.setVisible(False)
        layout.addWidget(self._tacs_view_group)

        layout.addStretch()

    # ── CurveAlign TACS View ───────────────────────────────────────────────────

    def _build_curvealign_tacs_view(self) -> QGroupBox:
        group = QGroupBox("CurveAlign TACS View")
        layout = QVBoxLayout(group)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("ROI:"))
        self._tacs_roi_combo = QComboBox()
        self._tacs_roi_combo.addItem("All ROIs")
        self._tacs_roi_combo.currentIndexChanged.connect(self._apply_tacs_filter)
        filter_row.addWidget(self._tacs_roi_combo, stretch=1)

        filter_row.addWidget(QLabel("TACS:"))
        self._tacs_class_combo = QComboBox()
        self._tacs_class_combo.addItem("All zones")
        for label in TACS_COLORS:
            self._tacs_class_combo.addItem(label)
        self._tacs_class_combo.addItem("Outside zone")
        self._tacs_class_combo.currentIndexChanged.connect(self._apply_tacs_filter)
        filter_row.addWidget(self._tacs_class_combo, stretch=1)
        layout.addLayout(filter_row)

        self._show_boundary_lines = QCheckBox("Show boundary association lines")
        self._show_boundary_lines.toggled.connect(self._toggle_boundary_lines)
        layout.addWidget(self._show_boundary_lines)

        # Fiber table
        self._tacs_table = QTableWidget()
        self._tacs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tacs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tacs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._tacs_table.itemSelectionChanged.connect(self._on_table_row_selected)
        layout.addWidget(self._tacs_table)

        return group

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_curvealign_committed(self, image_id: str, result) -> None:
        """Show the TACS View and populate it when a CurveAlign result is committed."""
        self._curvealign_result = result
        self._active_image_id = image_id
        self._tacs_view_group.setVisible(True)

        # Populate ROI dropdown from roi_summary_df
        roi_df = getattr(result, "roi_summary_df", None)
        self._tacs_roi_combo.clear()
        self._tacs_roi_combo.addItem("All ROIs")
        if roi_df is not None and len(roi_df) > 0:
            for roi_id in roi_df.index:
                self._tacs_roi_combo.addItem(str(roi_id))

        self._populate_tacs_table(result)

    def on_layer_selected(self, layer_name: str) -> None:
        """Highlight table row matching the selected napari layer point."""
        pass  # TODO: bidirectional selection in Batch 9 follow-up

    # ── Table population ───────────────────────────────────────────────────────

    def _populate_tacs_table(self, result, roi_filter=None, tacs_filter=None) -> None:
        df = getattr(result, "fiber_features_df", None)
        if df is None or len(df) == 0:
            self._tacs_table.setRowCount(0)
            return

        display_cols = [
            c for c in ("fiber_key", "center_row", "center_col", "angle",
                        "abs_angle", "angle_to_boundary_tangent", "tacs_class")
            if c in df.columns
        ]
        if not display_cols:
            display_cols = list(df.columns[:6])

        filtered = df.copy()
        if tacs_filter and tacs_filter != "All zones":
            if "tacs_class" in filtered.columns:
                filtered = filtered[filtered["tacs_class"].astype(str) == tacs_filter]

        self._tacs_table.setColumnCount(len(display_cols))
        self._tacs_table.setHorizontalHeaderLabels(display_cols)
        self._tacs_table.setRowCount(len(filtered))
        for row_idx, (_, row_data) in enumerate(filtered[display_cols].iterrows()):
            for col_idx, val in enumerate(row_data):
                item = QTableWidgetItem(
                    f"{val:.2f}" if isinstance(val, float) else str(val)
                )
                self._tacs_table.setItem(row_idx, col_idx, item)

    def _apply_tacs_filter(self) -> None:
        if self._curvealign_result is None:
            return
        tacs_sel = self._tacs_class_combo.currentText()
        roi_sel = self._tacs_roi_combo.currentText()
        tacs_filter = tacs_sel if tacs_sel != "All zones" else None
        self._populate_tacs_table(self._curvealign_result, roi_filter=roi_sel, tacs_filter=tacs_filter)

    def _on_table_row_selected(self) -> None:
        """Highlight the napari Points layer point for the selected fiber."""
        pass  # TODO: implement bidirectional selection

    def _toggle_boundary_lines(self, checked: bool) -> None:
        """Show or hide the boundary association lines layer."""
        if self._viewer is None or self._curvealign_result is None:
            return
        from ..utils.layer_utils import make_layer_name
        lname = make_layer_name(self._active_image_id or "", "Associations", "boundary")
        existing = next((l for l in self._viewer.layers if l.name == lname), None)

        if checked:
            if existing is None:
                self._create_boundary_lines_layer(lname)
            else:
                existing.visible = True
        elif existing is not None:
            existing.visible = False

    def _create_boundary_lines_layer(self, layer_name: str) -> None:
        """Lazily create dashed lines from each fiber centroid to its nearest boundary point."""
        df = getattr(self._curvealign_result, "fiber_features_df", None)
        if df is None or self._viewer is None:
            return
        if not {"center_row", "center_col", "boundary_point_row", "boundary_point_col"}.issubset(df.columns):
            return
        shapes = []
        for _, row in df.iterrows():
            shapes.append(np.array([
                [float(row["center_row"]), float(row["center_col"])],
                [float(row["boundary_point_row"]), float(row["boundary_point_col"])],
            ]))
        if shapes:
            self._viewer.add_shapes(
                shapes,
                shape_type="line",
                edge_color="white",
                edge_width=1,
                name=layer_name,
            )

    # ── Plot and figure actions ────────────────────────────────────────────────

    def _show_plot(self) -> None:
        pass  # TODO: wire to napari-matplotlib panel

    def _view_overlay_png(self) -> None:
        pass  # TODO: launch QDialog with generated overlay figure

    def _view_heatmap_png(self) -> None:
        pass  # TODO: launch QDialog with density heatmap figure
