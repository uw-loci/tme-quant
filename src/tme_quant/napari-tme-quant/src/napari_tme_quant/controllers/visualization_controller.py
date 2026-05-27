"""VisualizationController — creates and manages napari layers for analysis results.

Layer creation rules (CLAUDE_NAPARI.md §napari layer ↔ TMEHierarchy sync):
  - Layer naming:  [image_id] :: [ObjectType] :: [detail]
  - Layers are never removed — only hidden via layer.visible
  - Layer creation happens only on "Commit to Hierarchy" (on_committed callback)
  - Coordinate translation via coord_utils (row, col) → (y, x)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import QTimer

if TYPE_CHECKING:
    from .state import PluginState

from ..utils.layer_utils import make_layer_name, TACS_COLORS
from ..utils.coord_utils import fiber_df_to_napari_points, fiber_df_to_napari_shapes

# TACS classification thresholds (must match tacs.py)
_TACS3_THRESHOLD = 60.0   # angle_to_boundary_tangent ≥ 60° → TACS-3
_TACS2_THRESHOLD = 30.0   # angle_to_boundary_tangent ≤ 30° → TACS-2


class ColorStrategy(Enum):
    BY_OBJECT_TYPE = auto()
    BY_TACS_TYPE = auto()
    BY_ROI = auto()
    UNIFORM = auto()


class VisualizationController:
    """Creates and manages napari layers; handles layer visibility toggling.

    Layers are never removed — only hidden via layer.visible.
    Layer names follow: [image_id] :: [ObjectType] :: [detail]

    All layer creation happens here, never in widgets or other controllers.
    """

    def __init__(self, state: "PluginState", viewer) -> None:
        self._state = state
        self._viewer = viewer
        self._color_strategy = ColorStrategy.BY_TACS_TYPE

    # ── Signal handlers ────────────────────────────────────────────────────────

    def on_committed(self, image_id: str, obj_type: str) -> None:
        """Create napari layers when analysis results are committed to hierarchy."""
        if obj_type == "fiber":
            self._create_ctfire_layers(image_id)
        elif obj_type == "curvealign":
            self._create_curvealign_layers(image_id)

    def remove_analysis_layers(self, image_id: str) -> None:
        """Remove all analysis overlay layers for image_id (keeps the raw image layer).

        Called when an analysis is invalidated or reset so stale layers disappear.
        """
        if self._viewer is None:
            return
        prefix = image_id + " :: "
        to_remove = [l for l in list(self._viewer.layers) if l.name.startswith(prefix)]
        for layer in to_remove:
            self._viewer.layers.remove(layer)
        self._state.layer_map = {
            k: v for k, v in self._state.layer_map.items()
            if not k.startswith(prefix)
        }

    def on_image_selected(self, image_id: str) -> None:
        """Show only layers belonging to image_id; hide others.

        A layer belongs to image_id when its name is either:
          - exactly image_id  (the raw napari Image layer added by add_image)
          - starts with "image_id :: "  (analysis overlay layers)

        Visibility changes are deferred via QTimer so they run after napari
        finishes processing the current event, preventing the vispy
        RecursionError that occurs when layer.visible is set mid-event.
        """
        if self._viewer is None:
            return
        sep = " :: "
        prefix = image_id + sep

        def _apply():
            for layer in self._viewer.layers:
                layer.visible = (
                    layer.name == image_id
                    or layer.name.startswith(prefix)
                )

        QTimer.singleShot(0, _apply)

    # ── Layer creation: CT-FIRE ────────────────────────────────────────────────

    def _create_ctfire_layers(self, image_id: str) -> None:
        """Create Shapes layers for CT-FIRE fiber centerlines.

        Creates:
          [image_id] :: Fibers :: all   (all fibers as line segments)
          [image_id] :: Fibers :: TACS-3 / TACS-2 / TACS-1  (if TACS data available)
        """
        result = self._state.fiber_results.get(image_id)
        if result is None or self._viewer is None:
            return

        fibers = getattr(result, "fibers", [])
        if not fibers:
            return

        # Build a minimal DataFrame for coord conversion
        import pandas as pd
        rows = []
        for f in fibers:
            pos = f.get_position() if hasattr(f, "get_position") else [0, 0]
            rows.append({
                "center_row": float(pos[0]),
                "center_col": float(pos[1]),
                "angle": float(getattr(f, "orientation", 0.0)),
                "length": float(getattr(f, "length", 10.0)),
                "tacs_type": getattr(f, "tacs_type", None),
            })
        df = pd.DataFrame(rows)

        shapes = fiber_df_to_napari_shapes(df)
        layer_name = make_layer_name(image_id, "Fibers", "all")
        self._add_shapes_layer(layer_name, shapes, edge_color="cyan")

        # TACS sub-layers (only if tacs_type column is populated)
        if df["tacs_type"].notna().any():
            for tacs_label, color in TACS_COLORS.items():
                mask = df["tacs_type"].astype(str) == tacs_label
                if mask.any():
                    tacs_shapes = [s for s, m in zip(shapes, mask) if m]
                    lname = make_layer_name(image_id, "Fibers", tacs_label)
                    self._add_shapes_layer(lname, tacs_shapes, edge_color=color)

    # ── Layer creation: CurveAlign ─────────────────────────────────────────────

    def _create_curvealign_layers(self, image_id: str) -> None:
        """Create Points layers for CurveAlign fiber centroids.

        Creates:
          [image_id] :: Fibers :: curvealign   (all centroids, color by TACS)
          [image_id] :: Fibers :: TACS-3 / TACS-2 / TACS-1   (filtered)
        """
        result = self._state.curvealign_pipeline_results.get(image_id)
        if result is None or self._viewer is None:
            return

        df = getattr(result, "fiber_features_df", None)
        if df is None or len(df) == 0:
            return

        if "center_row" not in df.columns or "center_col" not in df.columns:
            return

        coords = fiber_df_to_napari_points(df)

        # All curvelet groups colored by TACS zone membership
        layer_name = make_layer_name(image_id, "Fibers", "all (TACS-colored)")
        face_colors = self._compute_tacs_face_colors(df, result)
        self._add_points_layer(layer_name, coords, face_color=face_colors)

        # Fiber orientation lines — 6px white line segments at absolute fiber angle
        if "angle" in df.columns:
            shapes = fiber_df_to_napari_shapes(df, line_length=6.0)
            overlay_name = make_layer_name(image_id, "Fibers", "orientation-lines")
            self._add_shapes_layer(overlay_name, shapes, edge_color="white")

        # Per-TACS sub-layers (in-zone fibers by TACS class)
        tacs_classes = self._classify_df_tacs(df, result)
        for tacs_label, color in TACS_COLORS.items():
            mask = tacs_classes == tacs_label
            if mask.any():
                lname = make_layer_name(image_id, "Fibers", tacs_label)
                self._add_points_layer(lname, coords[mask], face_color=color)

        # Out-of-zone fibers
        out_mask = tacs_classes == "outside"
        if out_mask.any():
            lname = make_layer_name(image_id, "Fibers", "out-of-zone")
            self._add_points_layer(lname, coords[out_mask], face_color="lightgray")

        # Orientation heatmap layer (procmap from generate_fiber_heatmap)
        self._create_heatmap_layer(image_id, result)

    def _create_heatmap_layer(self, image_id: str, result) -> None:
        """Generate a fiber orientation heatmap and add it as a napari Image layer."""
        img = self._state.images.get(image_id)
        if img is None or self._viewer is None:
            return
        fs = getattr(result, "fiber_structure", None)
        if fs is None or len(fs) == 0:
            return
        try:
            from tme_quant.fiber_analysis.visualization.draw_utils import generate_fiber_heatmap
            import numpy as np
            tif_boundary = 3 if getattr(result, "boundary_measurement", False) else 0
            _fig, _rawmap, procmap = generate_fiber_heatmap(
                img=img,
                fiber_structure=fs,
                in_curvs_flag=getattr(result, "in_curvs_flag", None),
                angles=getattr(result, "nearest_angles", None),
                distances=None,
                tif_boundary=tif_boundary,
                boundary_measurement=getattr(result, "boundary_measurement", False),
            )
            import matplotlib
            matplotlib.pyplot.close(_fig)  # free memory; we only need procmap
            if procmap is not None:
                layer_name = make_layer_name(image_id, "Heatmap", "orientation")
                existing = self._find_layer(layer_name)
                if existing is not None:
                    existing.data = procmap
                else:
                    self._viewer.add_image(
                        procmap,
                        name=layer_name,
                        colormap="inferno",
                        opacity=0.6,
                    )
        except Exception:
            pass  # heatmap generation is optional; never crash the commit flow

    def _compute_tacs_face_colors(self, df, result) -> list[str]:
        """Return a per-row color list based on TACS classification."""
        tacs = self._classify_df_tacs(df, result)
        return [TACS_COLORS.get(t, "lightgray") for t in tacs]

    def _classify_df_tacs(self, df, result) -> np.ndarray:
        """Classify each row as TACS-1/2/3 or 'outside'."""
        in_flag = getattr(result, "in_curvs_flag", None)
        angle_col = "angle_to_boundary_tangent"

        labels = np.full(len(df), "outside", dtype=object)

        if in_flag is not None and len(in_flag) == len(df):
            in_zone = np.asarray(in_flag, dtype=bool)
        elif angle_col in df.columns:
            in_zone = np.ones(len(df), dtype=bool)
        else:
            return labels

        angles = df[angle_col].to_numpy(dtype=float) if angle_col in df.columns else None

        if angles is not None:
            labels[in_zone & (angles >= _TACS3_THRESHOLD)] = "TACS-3"
            labels[in_zone & (angles <= _TACS2_THRESHOLD)] = "TACS-2"
            labels[in_zone & (angles > _TACS2_THRESHOLD) & (angles < _TACS3_THRESHOLD)] = "TACS-1"

        return labels

    # ── Low-level napari layer helpers ─────────────────────────────────────────

    def _add_shapes_layer(
        self, name: str, shapes: list, edge_color: str = "cyan"
    ) -> None:
        if self._viewer is None or not shapes:
            return
        existing = self._find_layer(name)
        if existing is not None:
            existing.data = shapes
            existing.visible = True
        else:
            layer = self._viewer.add_shapes(
                shapes,
                shape_type="line",
                edge_color=edge_color,
                edge_width=1.5,
                name=name,
            )
            self._state.layer_map[name] = name

    def _add_points_layer(
        self, name: str, coords: np.ndarray, face_color="cyan"
    ) -> None:
        if self._viewer is None or len(coords) == 0:
            return
        existing = self._find_layer(name)
        if existing is not None:
            existing.data = coords
            existing.visible = True
        else:
            self._viewer.add_points(
                coords,
                face_color=face_color,
                size=3,
                name=name,
            )
            self._state.layer_map[name] = name

    def _find_layer(self, name: str):
        if self._viewer is None:
            return None
        for layer in self._viewer.layers:
            if layer.name == name:
                return layer
        return None

    def set_color_strategy(self, strategy: ColorStrategy) -> None:
        self._color_strategy = strategy
