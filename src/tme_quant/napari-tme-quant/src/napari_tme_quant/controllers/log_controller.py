"""LogController — routes analysis events and messages to LogWidget."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import PluginState
    from ..widgets.log_widget import LogWidget


class LogController:
    """Routes log messages from controllers and analysis steps to LogWidget.

    Usage:
        log = LogController(state, log_widget)
        log.info("Loaded SHG_001.tif  512×512  1.0 µm/px")
        log.warn("No calibration metadata — pixel size defaulted to 1.0 µm/px")
    """

    def __init__(self, state: "PluginState", log_widget: "LogWidget") -> None:
        self._state = state
        self._widget = log_widget

    def info(self, msg: str) -> None:
        self._widget.append("INFO", msg)

    def warn(self, msg: str) -> None:
        self._widget.append("WARN", msg)

    def error(self, msg: str) -> None:
        self._widget.append("ERROR", msg)

    def on_analysis_complete(self, step: str, image_id: str, result) -> None:
        """Called by AnalysisController after each analysis step completes."""
        label = image_id or "?"
        if step == "fiber":
            n = len(getattr(result, "fibers", []))
            self.info(f"Extracted {n} fibers  [{label}]")
        elif step == "curvealign":
            df = getattr(result, "fiber_structure", None)
            n = len(df) if df is not None else "?"
            self.info(f"CurveAlign complete — {n} fiber groups  [{label}]")
        elif step == "cell":
            n = len(getattr(result, "cells", []))
            self.info(f"Segmented {n} cells  [{label}]")
        elif step == "tme":
            self.info(f"TME pipeline complete  [{label}]")
        else:
            self.info(f"{step} complete  [{label}]")

    def on_batch_progress(self, step: int, total: int, msg: str) -> None:
        """Called by AnalysisController during batch processing."""
        self.info(f"[{step}/{total}] {msg}")

    def on_error(self, exc: Exception) -> None:
        self.error(str(exc))
