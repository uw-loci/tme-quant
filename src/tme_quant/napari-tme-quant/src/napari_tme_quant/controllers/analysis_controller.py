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
        committed_to_hierarchy(image_id: str, obj_type: str)
        batch_progress(step: int, total: int, msg: str)
    """

    def __init__(self, state: "PluginState", log: Optional["LogController"] = None) -> None:
        self._state = state
        self._log = log
        self._on_analysis_complete: list[Callable] = []
        self._on_committed: list[Callable] = []
        self._on_batch_progress: list[Callable] = []

    # ── Signal subscriptions ───────────────────────────────────────────────────

    def connect_analysis_complete(self, fn: Callable) -> None:
        self._on_analysis_complete.append(fn)

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

        if params is None:
            params = CTFireParams()

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

        _worker()

    def _on_ctfire_result(self, image_id: str, result) -> None:
        self._state.fiber_results[image_id] = result
        self._emit_analysis_complete("fiber", image_id, result)

    def _run_curvealign_worker(self, image_id: str, image, **kwargs) -> None:
        from napari.qt.threading import thread_worker
        from tme_quant import curvealign_curvelets_mode_pipeline

        def _cb(step: int, total: int, msg: str) -> None:
            self._emit_batch_progress(step, total, msg)

        @thread_worker(connect={
            "returned": lambda result: self._on_curvealign_result(image_id, result),
            "errored":  lambda exc: self._on_error(exc),
        })
        def _worker():
            img = image if image is not None else np.zeros((64, 64), dtype=np.float32)
            return curvealign_curvelets_mode_pipeline(
                img, progress_callback=_cb, **kwargs
            )

        _worker()

    def _on_curvealign_result(self, image_id: str, result) -> None:
        if result is not None:
            self._state.curvealign_pipeline_results[image_id] = result
        self._emit_analysis_complete("curvealign", image_id, result)

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
            image_entry = self._get_or_create_image_entry(_image_id)
            # Attach as population-level fiber result using fiber_features_df
            self._attach_curvealign_result(image_entry, _image_id, result)
            self._emit_committed(_image_id, "curvealign")

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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _emit_analysis_complete(self, step: str, image_id: str, result) -> None:
        for fn in self._on_analysis_complete:
            fn(step, image_id, result)
        if self._log:
            self._log.on_analysis_complete(step, image_id, result)

    def _emit_committed(self, image_id: str, obj_type: str) -> None:
        for fn in self._on_committed:
            fn(image_id, obj_type)

    def _emit_batch_progress(self, step: int, total: int, msg: str) -> None:
        for fn in self._on_batch_progress:
            fn(step, total, msg)
        if self._log:
            self._log.on_batch_progress(step, total, msg)

    def _on_error(self, exc: Exception) -> None:
        if self._log:
            self._log.on_error(exc)
