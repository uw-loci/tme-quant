"""AnalysisController — dispatches all analysis steps with threading.

All tme_quant calls run in @thread_worker. PluginState is mutated only in the
returned-signal callback on the Qt main thread.

Fiber analysis (CT-FIRE + CurveAlign) is fully implemented here.
Cell, registration, and TME sections are stubs for future batches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .state import PluginState
    from ..controllers.log_controller import LogController


class AnalysisController:
    """Translates widget Run actions → threaded tme_quant calls.

    Emits:
        analysis_complete(step: str, image_id: str, result)
        analysis_started(step: str, image_id: str)
        analysis_aborted(step: str, image_id: str)
        committed_to_hierarchy(image_id: str, obj_type: str)
        batch_progress(step: int, total: int, msg: str)
    """

    def __init__(self, state: "PluginState", log: Optional["LogController"] = None) -> None:
        self._state = state
        self._log = log
        self._on_analysis_complete: list[Callable] = []
        self._on_analysis_started: list[Callable] = []
        self._on_analysis_aborted: list[Callable] = []
        self._on_committed: list[Callable] = []
        self._on_batch_progress: list[Callable] = []
        # Active worker tracking (only one analysis at a time)
        self._active_worker = None
        self._active_step: Optional[str] = None
        self._active_image_for_abort: Optional[str] = None

    # ── Signal subscriptions ───────────────────────────────────────────────────

    def connect_analysis_complete(self, fn: Callable) -> None:
        self._on_analysis_complete.append(fn)

    def connect_analysis_started(self, fn: Callable) -> None:
        self._on_analysis_started.append(fn)

    def connect_analysis_aborted(self, fn: Callable) -> None:
        self._on_analysis_aborted.append(fn)

    def connect_committed_to_hierarchy(self, fn: Callable) -> None:
        self._on_committed.append(fn)

    def connect_batch_progress(self, fn: Callable) -> None:
        self._on_batch_progress.append(fn)

    # ── Fiber extraction ───────────────────────────────────────────────────────

    def run_fiber_extraction(
        self,
        image_id: Optional[str],
        method: str = "ctfire",
        image: Optional[np.ndarray] = None,
        params=None,
        **pipeline_kwargs,
    ) -> None:
        """Launch CT-FIRE or CurveAlign pipeline in a background worker thread.

        Parameters
        ----------
        image_id : str or None
            Key to store results under in PluginState.  Uses
            state.active_image_id if None.
        method : 'ctfire' | 'curvealign'
        image : ndarray or None
            If None, the image is loaded from the napari viewer by image_id
            (not yet wired up in v1 skeleton; pass explicitly for now).
        params : CTFireParams or None
            CT-FIRE params.  Defaults to CTFireParams() if not supplied.
        **pipeline_kwargs
            Additional kwargs forwarded to curvealign_curvelets_mode_pipeline().
        """
        from napari.qt.threading import thread_worker

        _image_id = image_id or self._state.active_image_id or "unknown"

        if method == "ctfire":
            self._run_ctfire_worker(_image_id, image, params)
        elif method == "curvealign":
            self._run_curvealign_worker(_image_id, image, **pipeline_kwargs)
        else:
            raise ValueError(f"Unknown fiber method: {method!r}")

    def _run_ctfire_worker(self, image_id: str, image, params) -> None:
        from napari.qt.threading import thread_worker
        from tme_quant import FiberExtractionAnalyzer, CTFireParams, ExtractionParams

        if self._active_worker is not None:
            if self._log:
                self._log.warn("Analysis already running — abort or wait before re-running.")
            return

        if params is None:
            params = CTFireParams()

        self._active_step = "fiber"
        self._active_image_for_abort = image_id
        self._emit_analysis_started("fiber", image_id)

        def _cb(step: int, total: int, msg: str) -> None:
            self._emit_batch_progress(step, total, msg)

        @thread_worker(connect={
            "returned": lambda result: self._on_ctfire_result(image_id, result),
            "errored":  lambda exc: self._on_error(exc),
        })
        def _worker():
            return FiberExtractionAnalyzer().extract_2d(
                image if image is not None else np.zeros((64, 64), dtype=np.float32),
                params,
                progress_callback=_cb,
            )

        worker = _worker()
        worker.signals.finished.connect(self._on_worker_finished)
        self._active_worker = worker

    def _on_ctfire_result(self, image_id: str, result) -> None:
        self._state.fiber_results[image_id] = result
        self._emit_analysis_complete("fiber", image_id, result)

    def _run_curvealign_worker(self, image_id: str, image, **kwargs) -> None:
        from napari.qt.threading import thread_worker
        from tme_quant import curvealign_curvelets_mode_pipeline

        if self._active_worker is not None:
            if self._log:
                self._log.warn("Analysis already running — abort or wait before re-running.")
            return

        if self._log:
            self._log.info(f"Starting CurveAlign pipeline for {image_id!r} …")

        self._active_step = "curvealign"
        self._active_image_for_abort = image_id
        self._emit_analysis_started("curvealign", image_id)

        def _cb(step: int, total: int, msg: str) -> None:
            print(f"[curvealign {step}/{total}] {msg}", flush=True)
            # Defer GUI update to the main thread — calling Qt widgets directly
            # from a worker thread triggers OpenGL context errors on some platforms.
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, lambda s=step, t=total, m=msg:
                              self._emit_batch_progress(s, t, m))

        @thread_worker(connect={
            "returned": lambda result: self._on_curvealign_result(image_id, result),
            "errored":  lambda exc: self._on_error(exc),
        })
        def _worker():
            img = image if image is not None else np.zeros((64, 64), dtype=np.float32)
            try:
                return curvealign_curvelets_mode_pipeline(
                    img, progress_callback=_cb, **kwargs
                )
            except Exception as exc:
                import traceback
                print(f"[curvealign ERROR] {exc}", flush=True)
                traceback.print_exc()
                raise

        worker = _worker()
        worker.signals.finished.connect(self._on_worker_finished)
        self._active_worker = worker

    def _on_curvealign_result(self, image_id: str, result) -> None:
        if result is not None:
            self._state.curvealign_pipeline_results[image_id] = result
        self._emit_analysis_complete("curvealign", image_id, result)
        if result is not None and getattr(self._state, "project_dir", None):
            self._auto_save_curvealign(image_id, result)

    def _auto_save_curvealign(self, image_id: str, result) -> None:
        out_dir = self._state.project_dir / "output" / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Fiber features CSV
        df = getattr(result, "fiber_features_df", None)
        if df is not None and len(df) > 0:
            from ..utils.export_utils import export_df_to_csv
            export_df_to_csv(df, out_dir / "fiber_features.csv")
            if self._log:
                self._log.info(f"Auto-saved fiber_features.csv → {out_dir}")
        # Figures (non-interactive, Agg backend)
        img = self._state.images.get(image_id)
        if img is None:
            return
        fs = getattr(result, "fiber_structure", None)
        if fs is None or (hasattr(fs, "__len__") and len(fs) == 0):
            if self._log:
                self._log.warn("Auto-save: no fiber_structure — skipping figures")
            return
        try:
            import traceback as _tb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from tme_quant.fiber_analysis.visualization.draw_utils import (
                generate_fiber_heatmap, generate_fiber_overlay,
            )
            tif_boundary = 3 if getattr(result, "boundary_measurement", False) else 0
            bm = getattr(result, "boundary_measurement", False)
            fig_h, _, _ = generate_fiber_heatmap(
                img=img,
                fiber_structure=fs,
                in_curvs_flag=getattr(result, "in_curvs_flag", None),
                angles=getattr(result, "nearest_angles", None),
                distances=None,
                tif_boundary=tif_boundary,
                boundary_measurement=bm,
            )
            if fig_h is not None:
                fig_h.savefig(str(out_dir / "orientation_heatmap.png"), dpi=150, bbox_inches="tight")
                plt.close(fig_h)
                if self._log:
                    self._log.info(f"Auto-saved orientation_heatmap.png → {out_dir}")
            in_flag = getattr(result, "in_curvs_flag", None)
            out_flag = (~in_flag) if in_flag is not None else None
            fig_o, _ = generate_fiber_overlay(
                img=img,
                fiber_structure=fs,
                coordinates=getattr(result, "roi_coordinates", None),
                in_curvs_flag=in_flag,
                out_curvs_flag=out_flag,
                nearest_angles=getattr(result, "nearest_angles", None),
                measured_boundary=None,
                fiber_mode=0,
                tif_boundary=tif_boundary,
                boundary_measurement=bm,
            )
            if fig_o is not None:
                fig_o.savefig(str(out_dir / "fiber_overlay.png"), dpi=150, bbox_inches="tight")
                plt.close(fig_o)
                if self._log:
                    self._log.info(f"Auto-saved fiber_overlay.png → {out_dir}")
        except Exception as exc:
            _tb.print_exc()
            if self._log:
                self._log.warn(f"Auto-save figures failed: {exc}")

    # ── Hierarchy commit ───────────────────────────────────────────────────────

    def commit_fiber_result(
        self,
        image_id: Optional[str],
        method: str = "ctfire",
    ) -> None:
        """Attach raw fiber result from PluginState to TMEHierarchy.

        Emits committed_to_hierarchy(image_id, "fiber") on success.
        """
        _image_id = image_id or self._state.active_image_id or "unknown"

        if method == "ctfire":
            result = self._state.fiber_results.get(_image_id)
            if result is None:
                if self._log:
                    self._log.warn(f"No CT-FIRE result to commit for {_image_id!r}")
                return
            image_entry = self._get_or_create_image_entry(_image_id)
            self._state.hierarchy.attach_fiber_result(result, image_entry, _image_id)
            self._emit_committed(_image_id, "fiber")

        elif method == "curvealign":
            result = self._state.curvealign_pipeline_results.get(_image_id)
            if result is None:
                if self._log:
                    self._log.warn(f"No CurveAlign result to commit for {_image_id!r}")
                return
            if self._log:
                self._log.info(f"Committing CurveAlign result to hierarchy for {_image_id!r} …")
            print(f"[commit] curvealign → {_image_id}", flush=True)
            try:
                image_entry = self._get_or_create_image_entry(_image_id)
                self._attach_curvealign_result(image_entry, _image_id, result)
                self._emit_committed(_image_id, "curvealign")
                if self._log:
                    self._log.info(f"Committed CurveAlign result for {_image_id!r}")
                print(f"[commit] done → {_image_id}", flush=True)
            except Exception as exc:
                import traceback
                print(f"[commit ERROR] {exc}", flush=True)
                traceback.print_exc()
                if self._log:
                    self._log.error(f"Commit failed: {exc}")

    def _get_or_create_image_entry(self, image_id: str):
        """Return the ImageEntry for image_id, creating a root one if needed."""
        from tme_quant import ImageEntry, TMEObject
        existing = self._state.hierarchy.get_object(image_id)
        if existing is not None:
            return existing
        entry = ImageEntry(object_id=image_id, name=image_id)
        self._state.hierarchy.add_object(entry)
        return entry

    def _attach_curvealign_result(self, parent, image_id: str, result) -> None:
        """Convert CurveAlignPipelineResult fiber_features_df into FiberObjects."""
        from tme_quant import FiberObject
        df = getattr(result, "fiber_features_df", None)
        if df is None or len(df) == 0:
            return
        for idx, row in df.iterrows():
            fiber = FiberObject(
                object_id=f"{image_id}_ca_fiber_{idx}",
                centerline=[[float(row.get("center_row", 0.0)),
                              float(row.get("center_col", 0.0))]],
                orientation=float(row.get("angle", 0.0)),
            )
            self._state.hierarchy.add_object(fiber, parent=parent)

    # ── Abort / invalidate ─────────────────────────────────────────────────────

    def abort(self) -> None:
        """Interrupt any running analysis and emit analysis_aborted."""
        if self._active_worker is None:
            return
        step = self._active_step or "unknown"
        image_id = self._active_image_for_abort or "unknown"
        self._active_worker.quit()
        self._active_worker = None
        self._active_step = None
        self._active_image_for_abort = None
        self._emit_analysis_aborted(step, image_id)

    def _on_worker_finished(self) -> None:
        """Called on the Qt main thread when the worker thread exits (success or quit)."""
        self._active_worker = None
        self._active_step = None
        self._active_image_for_abort = None

    def invalidate_result(self, image_id: str, method: str) -> None:
        """Clear a cached analysis result and remove its napari layers.

        Emits analysis_aborted so UI widgets re-check Commit state.
        """
        if method == "curvealign":
            self._state.curvealign_pipeline_results.pop(image_id, None)
            step = "curvealign"
        else:
            self._state.fiber_results.pop(image_id, None)
            step = "fiber"
        self._emit_analysis_aborted(step, image_id)

    def invalidate_all(self) -> None:
        """Abort any running analysis and clear all cached results."""
        self.abort()
        ca_ids = list(self._state.curvealign_pipeline_results.keys())
        for iid in ca_ids:
            self.invalidate_result(iid, "curvealign")
        fiber_ids = list(self._state.fiber_results.keys())
        for iid in fiber_ids:
            self.invalidate_result(iid, "ctfire")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _emit_analysis_complete(self, step: str, image_id: str, result) -> None:
        for fn in self._on_analysis_complete:
            fn(step, image_id, result)
        if self._log:
            self._log.on_analysis_complete(step, image_id, result)

    def notify_result_loaded(self, step: str, image_id: str, result) -> None:
        """Fire analysis_complete after a project is loaded from disk.

        Called by IOWidget after load_plugin_state restores results to state.
        This lets visualization and fiber widgets update without re-running.
        """
        self._emit_analysis_complete(step, image_id, result)

    def _emit_analysis_started(self, step: str, image_id: str) -> None:
        for fn in self._on_analysis_started:
            fn(step, image_id)

    def _emit_analysis_aborted(self, step: str, image_id: str) -> None:
        for fn in self._on_analysis_aborted:
            fn(step, image_id)

    def _emit_committed(self, image_id: str, obj_type: str) -> None:
        for fn in self._on_committed:
            fn(image_id, obj_type)

    def _emit_batch_progress(self, step: int, total: int, msg: str) -> None:
        for fn in self._on_batch_progress:
            fn(step, total, msg)
        if self._log:
            self._log.on_batch_progress(step, total, msg)

    def _on_error(self, exc: Exception) -> None:
        import traceback
        print(f"[analysis ERROR] {exc}", flush=True)
        traceback.print_exc()
        if self._log:
            self._log.on_error(exc)
        # Emit aborted so UI re-enables Run buttons (previous result, if any, stays intact)
        step = self._active_step or "unknown"
        image_id = self._active_image_for_abort or "unknown"
        self._active_worker = None
        self._active_step = None
        self._active_image_for_abort = None
        self._emit_analysis_aborted(step, image_id)
