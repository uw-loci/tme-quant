"""Plugin-level project save and restore helpers.

Save layout:
    <save_dir>/
        plugin_state.json               image_types, per_image_params,
                                        active_image_id, image_paths
        results/<image_id>/
            fiber_features.csv
            roi_summary.csv             (only when boundary_measurement=True)
            fiber_structure.csv
            density.csv
            alignment.csv
            in_curvs_flag.npy
            nearest_angles.npy
            params.json                 CurveAlignPipelineResult.params dict

These helpers are called from ProjectWidget and (optionally) IOWidget.
They do NOT call tme_quant.core.io.save_project — that serialises the
TMEHierarchy; here we persist the plugin-level state that the core library
does not know about (DataFrames, numpy arrays, per-image params).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..controllers.state import PluginState


# ── Save ───────────────────────────────────────────────────────────────────────

def save_plugin_state(state: "PluginState", save_dir: Path) -> None:
    """Persist PluginState to *save_dir*.

    Creates the directory if it does not exist. Overwrites existing files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top-level metadata
    from ..controllers.state import ImageType
    meta = {
        "active_image_id": state.active_image_id,
        "image_types":     {k: v.name for k, v in state.image_types.items()},
        "image_pairs":     dict(state.image_pairs),
        "image_paths":     dict(state.image_paths),
        "per_image_params": {
            iid: {step: params for step, params in steps.items()}
            for iid, steps in state.per_image_params.items()
        },
    }
    with open(save_dir / "plugin_state.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    # 2. Per-image CurveAlign results
    for image_id, result in state.curvealign_pipeline_results.items():
        result_dir = save_dir / "results" / image_id
        result_dir.mkdir(parents=True, exist_ok=True)
        _save_curvealign_result(result, result_dir)


def _save_curvealign_result(result, result_dir: Path) -> None:
    """Write all CurveAlignPipelineResult fields to *result_dir*."""
    _write_df(result_dir / "fiber_features.csv",  getattr(result, "fiber_features_df", None))
    _write_df(result_dir / "fiber_structure.csv",  getattr(result, "fiber_structure", None))
    _write_df(result_dir / "density.csv",          getattr(result, "density_df", None))
    _write_df(result_dir / "alignment.csv",        getattr(result, "alignment_df", None))

    roi_m = getattr(result, "roi_measurements_df", None)
    if roi_m is not None:
        _write_df(result_dir / "roi_measurements.csv", roi_m)
    roi_s = getattr(result, "roi_summary_df", None)
    if roi_s is not None:
        _write_df(result_dir / "roi_summary.csv", roi_s)

    _write_npy(result_dir / "in_curvs_flag.npy",  getattr(result, "in_curvs_flag", None))
    _write_npy(result_dir / "nearest_angles.npy",  getattr(result, "nearest_angles", None))

    params = getattr(result, "params", None) or {}
    with open(result_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=_json_default)


def _write_df(path: Path, df) -> None:
    if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
        df.to_csv(path, index=False, encoding="utf-8")


def _write_npy(path: Path, arr) -> None:
    if arr is not None:
        np.save(path, np.asarray(arr))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ── Load ───────────────────────────────────────────────────────────────────────

def load_plugin_state(state: "PluginState", save_dir: Path, viewer) -> None:
    """Restore PluginState from *save_dir* and recreate napari layers.

    The state object is mutated in-place. Existing state is cleared first.
    """
    save_dir = Path(save_dir)
    plugin_state_path = save_dir / "plugin_state.json"
    if not plugin_state_path.exists():
        raise FileNotFoundError(f"No plugin_state.json found in {save_dir}")

    with open(plugin_state_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Clear current transient state
    state.reset()
    state.image_types.clear()
    state.image_paths.clear()
    state.images.clear()
    state.per_image_params.clear()

    from ..controllers.state import ImageType

    # Restore metadata
    state.active_image_id = meta.get("active_image_id")
    for iid, type_name in meta.get("image_types", {}).items():
        try:
            state.image_types[iid] = ImageType[type_name]
        except KeyError:
            state.image_types[iid] = ImageType.UNKNOWN
    for k, v in meta.get("image_pairs", {}).items():
        state.image_pairs[k] = v
    for k, v in meta.get("image_paths", {}).items():
        state.image_paths[k] = v
    for iid, steps in meta.get("per_image_params", {}).items():
        state.per_image_params[iid] = dict(steps)

    # Reload image files and add as napari layers
    import imageio.v3 as iio
    for image_id, path_str in state.image_paths.items():
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            data = np.asarray(iio.imread(str(path)), dtype=np.float32)
            state.images[image_id] = data
            existing = next((l for l in viewer.layers if l.name == image_id), None)
            if existing is None:
                viewer.add_image(data, name=image_id)
        except Exception:
            pass  # missing or unreadable file — skip silently

    # Restore CurveAlign pipeline results
    results_root = save_dir / "results"
    if results_root.exists():
        for image_id in state.image_types:
            result_dir = results_root / image_id
            if result_dir.exists():
                result = _load_curvealign_result(result_dir)
                if result is not None:
                    state.curvealign_pipeline_results[image_id] = result

    # Recreate napari layers for restored results
    # (deferred to caller via _main_widget._rewire_after_load if needed;
    #  basic layer recreation happens here for the active image)
    _recreate_layers(state, viewer)


def _load_curvealign_result(result_dir: Path):
    """Reconstruct a CurveAlignPipelineResult from CSV/npy files."""
    try:
        from tme_quant.tme_analysis.pipelines.curvealign_curveletsMode_pipeline import (
            CurveAlignPipelineResult,
        )
    except ImportError:
        return None

    def _read_df(name: str):
        p = result_dir / name
        return pd.read_csv(p) if p.exists() else None

    def _read_npy(name: str):
        p = result_dir / name
        return np.load(p) if p.exists() else None

    params_path = result_dir / "params.json"
    params = {}
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)

    fiber_features_df  = _read_df("fiber_features.csv")
    fiber_structure    = _read_df("fiber_structure.csv")
    density_df         = _read_df("density.csv")
    alignment_df       = _read_df("alignment.csv")
    roi_measurements   = _read_df("roi_measurements.csv")
    roi_summary_df     = _read_df("roi_summary.csv")
    in_curvs_flag      = _read_npy("in_curvs_flag.npy")
    nearest_angles     = _read_npy("nearest_angles.npy")

    if fiber_features_df is None:
        return None  # no meaningful result to restore

    boundary_measurement = roi_summary_df is not None

    return CurveAlignPipelineResult(
        fiber_structure      = fiber_structure    if fiber_structure is not None else pd.DataFrame(),
        fiber_features_df    = fiber_features_df,
        density_df           = density_df         if density_df is not None else pd.DataFrame(),
        alignment_df         = alignment_df       if alignment_df is not None else pd.DataFrame(),
        roi_measurements_df  = roi_measurements,
        roi_summary_df       = roi_summary_df,
        in_curvs_flag        = in_curvs_flag,
        nearest_angles       = nearest_angles,
        boundary_measurement = boundary_measurement,
        params               = params,
    )


def _recreate_layers(state: "PluginState", viewer) -> None:
    """Recreate napari Points layers for all restored CurveAlign results."""
    try:
        from ..controllers.visualization_controller import VisualizationController
        from ..utils.layer_utils import make_layer_name
        from ..utils.coord_utils import fiber_df_to_napari_points
        from ..utils.layer_utils import TACS_COLORS
    except ImportError:
        return

    for image_id, result in state.curvealign_pipeline_results.items():
        df = getattr(result, "fiber_features_df", None)
        if df is None or "center_row" not in df.columns:
            continue
        coords = fiber_df_to_napari_points(df)
        layer_name = make_layer_name(image_id, "Fibers", "curvealign")
        existing = next((l for l in viewer.layers if l.name == layer_name), None)
        if existing is None:
            viewer.add_points(coords, name=layer_name, size=5)
