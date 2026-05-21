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
        self._overlay_fig = None           # cached matplotlib Figure
        self._heatmap_fig = None           # cached matplotlib Figure
        self._overlay_dialog = None        # keep reference so dialog stays open
        self._heatmap_dialog = None
        self._features_dialog = None
        self._tacs_table_df = None         # cached full DataFrame for fast row hiding
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
        self._plots_group = QGroupBox("Plots (docked napari-matplotlib)")
        self._plots_group.setEnabled(False)
        plots_row = QHBoxLayout(self._plots_group)
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
        layout.addWidget(self._plots_group)

        # ── Exported figures ──────────────────────────────────────────────────
        self._fig_group = QGroupBox("Exported figures")
        self._fig_group.setEnabled(False)
        fig_layout = QVBoxLayout(self._fig_group)
        view_overlay_btn = QPushButton("View Overlay PNG in window")
        view_heatmap_btn = QPushButton("View Orientation Heatmap in window")
        view_overlay_btn.clicked.connect(self._view_overlay_png)
        view_heatmap_btn.clicked.connect(self._view_heatmap_png)
        fig_layout.addWidget(view_overlay_btn)
        fig_layout.addWidget(view_heatmap_btn)
        layout.addWidget(self._fig_group)

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

        # Save results area
        save_group = QGroupBox("Save results")
        save_layout = QVBoxLayout(save_group)
        row1 = QHBoxLayout()
        btn_csv   = QPushButton("Save fiber features CSV")
        btn_xlsx  = QPushButton("Save fiber features XLSX")
        btn_csv.clicked.connect(self._save_fiber_features_csv)
        btn_xlsx.clicked.connect(self._save_fiber_features_xlsx)
        row1.addWidget(btn_csv)
        row1.addWidget(btn_xlsx)
        row1.addStretch()
        save_layout.addLayout(row1)
        row2 = QHBoxLayout()
        btn_overlay = QPushButton("Save overlay PNG")
        btn_heatmap = QPushButton("Save heatmap PNG")
        btn_overlay.clicked.connect(self._save_overlay_png)
        btn_heatmap.clicked.connect(self._save_heatmap_png)
        row2.addWidget(btn_overlay)
        row2.addWidget(btn_heatmap)
        row2.addStretch()
        save_layout.addLayout(row2)
        layout.addWidget(save_group)

        return group

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_analysis_complete(self, image_id: str, result) -> None:
        """Called immediately after CurveAlign analysis finishes (before commit).

        Enables the view buttons and shows the fiber features table.
        """
        if result is None:
            return
        self._curvealign_result = result
        self._active_image_id = image_id
        self._fig_group.setEnabled(True)
        self._plots_group.setEnabled(True)
        df = getattr(result, "fiber_features_df", None)
        if df is not None and len(df) > 0:
            self._show_fiber_features_table(df, image_id)

    def _show_fiber_features_table(self, df, image_id: str) -> None:
        from ..utils.export_utils import open_dataframe_dialog
        self._features_dialog = open_dataframe_dialog(
            df, f"Fiber Features — {image_id}", self
        )

    def on_curvealign_committed(self, image_id: str, result) -> None:
        """Show the TACS View and populate it when a CurveAlign result is committed."""
        self._curvealign_result = result
        self._active_image_id = image_id
        self._fig_group.setEnabled(True)
        self._plots_group.setEnabled(True)
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

    _TACS_TABLE_COLS = [
        "fiber_key",
        "center_row", "center_col",
        "fiber_absolute_angle",
        "nearest_relative_boundary_angle",
        "nearest_distance_to_boundary",
        "tacs_class",
    ]

    def _populate_tacs_table(self, result, roi_filter=None, tacs_filter=None) -> None:
        df = getattr(result, "fiber_features_df", None)
        if df is None or len(df) == 0:
            self._tacs_table.setRowCount(0)
            self._tacs_table_df = None
            return

        display_cols = [c for c in self._TACS_TABLE_COLS if c in df.columns]
        if not display_cols:
            display_cols = list(df.columns[:7])

        self._tacs_table_df = df[display_cols].copy()
        self._tacs_table.setSortingEnabled(False)
        self._tacs_table.setColumnCount(len(display_cols))
        self._tacs_table.setHorizontalHeaderLabels(display_cols)
        self._tacs_table.setRowCount(len(self._tacs_table_df))

        for row_idx, (_, row_data) in enumerate(self._tacs_table_df.iterrows()):
            for col_idx, val in enumerate(row_data):
                item = QTableWidgetItem(
                    f"{val:.3f}" if isinstance(val, float) else str(val)
                )
                self._tacs_table.setItem(row_idx, col_idx, item)

        self._tacs_table.setSortingEnabled(True)
        self._tacs_table.resizeColumnsToContents()
        # Apply any active filter immediately
        self._apply_tacs_filter_fast()

    def _apply_tacs_filter(self) -> None:
        self._apply_tacs_filter_fast()

    def _apply_tacs_filter_fast(self) -> None:
        """Hide/show rows without rebuilding — instant even for thousands of fibers."""
        df = self._tacs_table_df
        if df is None or self._tacs_table.rowCount() == 0:
            return
        tacs_sel = self._tacs_class_combo.currentText()
        want_tacs = tacs_sel if tacs_sel != "All zones" else None

        tacs_col = df.columns.get_loc("tacs_class") if "tacs_class" in df.columns else None

        for row_idx in range(self._tacs_table.rowCount()):
            hide = False
            if want_tacs and tacs_col is not None:
                item = self._tacs_table.item(row_idx, tacs_col)
                if item and item.text() != want_tacs:
                    hide = True
            self._tacs_table.setRowHidden(row_idx, hide)

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
            # Also make boundary mask and source image layers visible
            self._set_mask_layers_visible(True)
            if existing is None:
                self._create_boundary_lines_layer(lname)
            else:
                existing.visible = True
        else:
            if existing is not None:
                existing.visible = False

    def _set_mask_layers_visible(self, visible: bool) -> None:
        """Make all MASK-type image layers visible (or hidden)."""
        if self._viewer is None or self._viz_controller is None:
            return
        from ..controllers.state import ImageType
        state = self._viz_controller._state
        mask_ids = {iid for iid, itype in state.image_types.items()
                    if itype == ImageType.MASK}
        # Also make the active fiber image visible
        if self._active_image_id:
            mask_ids.add(self._active_image_id)
        for layer in self._viewer.layers:
            if layer.name in mask_ids:
                layer.visible = visible

    def _create_boundary_lines_layer(self, layer_name: str) -> None:
        """Lazily create lines from each fiber centroid to its nearest boundary point.

        Note: boundary_point_row/col in fiber_features_df use MATLAB (x,y) convention
        where boundary_point_row=col and boundary_point_col=row. Swap when building
        napari (row, col) coordinates.
        """
        df = getattr(self._curvealign_result, "fiber_features_df", None)
        if df is None or self._viewer is None:
            return
        if not {"center_row", "center_col", "boundary_point_row", "boundary_point_col"}.issubset(df.columns):
            return
        shapes = []
        for _, row in df.iterrows():
            bpr = float(row["boundary_point_row"])
            bpc = float(row["boundary_point_col"])
            shapes.append(np.array([
                [float(row["center_row"]), float(row["center_col"])],
                [bpc, bpr],  # swap: boundary_point_col=row, boundary_point_row=col
            ]))
        if shapes:
            self._viewer.add_shapes(
                shapes,
                shape_type="line",
                edge_color="cyan",
                edge_width=1,
                name=layer_name,
            )

    # ── Plot and figure actions ────────────────────────────────────────────────

    def _show_plot(self) -> None:
        pass  # TODO: wire to napari-matplotlib panel

    def _view_overlay_png(self) -> None:
        print(f"[overlay] active_id={self._active_image_id!r} result={self._curvealign_result is not None}", flush=True)
        fig = self._generate_overlay_fig()
        if fig is not None:
            from ..utils.export_utils import open_figure_dialog
            self._overlay_fig = fig
            self._overlay_dialog = open_figure_dialog(fig, "Fiber Overlay", self)
        else:
            print("[overlay] figure generation returned None", flush=True)

    def _view_heatmap_png(self) -> None:
        fig = self._generate_heatmap_fig()
        if fig is not None:
            from ..utils.export_utils import open_figure_dialog
            self._heatmap_fig = fig
            self._heatmap_dialog = open_figure_dialog(fig, "Orientation Heatmap", self)

    # ── Figure generation ──────────────────────────────────────────────────────

    def _get_image_array(self):
        """Get the raw image array for the active image (from state via viz controller)."""
        if self._viz_controller is None or self._active_image_id is None:
            return None
        return self._viz_controller._state.images.get(self._active_image_id)

    def _generate_overlay_fig(self):
        result = self._curvealign_result
        img = self._get_image_array()
        if result is None or img is None:
            return None
        try:
            from tme_quant.fiber_analysis.visualization.draw_utils import generate_fiber_overlay
            tif_boundary = 3 if getattr(result, "boundary_measurement", False) else 0
            in_flag = getattr(result, "in_curvs_flag", None)
            out_flag = (~in_flag) if in_flag is not None else None
            fig, _ = generate_fiber_overlay(
                img=img,
                fiber_structure=result.fiber_structure,
                coordinates=getattr(result, "roi_coordinates", None),
                in_curvs_flag=in_flag,
                out_curvs_flag=out_flag,
                nearest_angles=getattr(result, "nearest_angles", None),
                measured_boundary=None,
                fiber_mode=0,
                tif_boundary=tif_boundary,
                boundary_measurement=getattr(result, "boundary_measurement", False),
            )
            return fig
        except Exception as exc:
            import traceback
            print(f"[overlay ERROR] {exc}", flush=True)
            traceback.print_exc()
            return None

    def _generate_heatmap_fig(self):
        result = self._curvealign_result
        img = self._get_image_array()
        if result is None or img is None:
            print(f"[heatmap] result={result is not None} img={img is not None}", flush=True)
            return None
        try:
            from tme_quant.fiber_analysis.visualization.draw_utils import generate_fiber_heatmap
            tif_boundary = 3 if getattr(result, "boundary_measurement", False) else 0
            fig, _rawmap, _procmap = generate_fiber_heatmap(
                img=img,
                fiber_structure=result.fiber_structure,
                in_curvs_flag=getattr(result, "in_curvs_flag", None),
                angles=getattr(result, "nearest_angles", None),
                distances=None,
                tif_boundary=tif_boundary,
                boundary_measurement=getattr(result, "boundary_measurement", False),
            )
            return fig
        except Exception as exc:
            import traceback
            print(f"[heatmap ERROR] {exc}", flush=True)
            traceback.print_exc()
            return None

    # ── Save actions ───────────────────────────────────────────────────────────

    def _save_fiber_features_csv(self) -> None:
        if self._curvealign_result is None:
            return
        df = getattr(self._curvealign_result, "fiber_features_df", None)
        if df is None:
            return
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fiber features", "fiber_features.csv",
            "CSV (*.csv);;All files (*)"
        )
        if path:
            from ..utils.export_utils import export_df_to_csv
            export_df_to_csv(df, path)

    def _save_fiber_features_xlsx(self) -> None:
        if self._curvealign_result is None:
            return
        df = getattr(self._curvealign_result, "fiber_features_df", None)
        if df is None:
            return
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save fiber features", "fiber_features.xlsx",
            "Excel (*.xlsx);;All files (*)"
        )
        if path:
            from ..utils.export_utils import export_df_to_excel
            export_df_to_excel(df, path)

    def _save_overlay_png(self) -> None:
        fig = getattr(self, "_overlay_fig", None) or self._generate_overlay_fig()
        if fig is None:
            return
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save overlay", "fiber_overlay.png", "PNG (*.png);;All files (*)"
        )
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")

    def _save_heatmap_png(self) -> None:
        fig = getattr(self, "_heatmap_fig", None) or self._generate_heatmap_fig()
        if fig is None:
            return
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save heatmap", "fiber_heatmap.png", "PNG (*.png);;All files (*)"
        )
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
