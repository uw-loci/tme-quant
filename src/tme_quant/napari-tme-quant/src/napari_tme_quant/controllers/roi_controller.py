"""ROIController — single source of truth for ROI Manager ↔ napari Shapes sync."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .state import PluginState


class ROIController:
    """Syncs tme_quant.ROIManager with the napari Shapes layer.

    napari layer.events.data is debounced (300 ms) before syncing to library.
    Full implementation in a future batch; stub provides the interface contract.

    Emits:
        roi_drawn(roi_id: str)
        roi_changed(roi_id: str)
    """

    _DEBOUNCE_MS = 300

    def __init__(self, state: "PluginState", viewer) -> None:
        self._state = state
        self._viewer = viewer
        self._on_roi_drawn: list[Callable] = []
        self._on_roi_changed: list[Callable] = []

    def connect_roi_drawn(self, fn: Callable[[str], None]) -> None:
        self._on_roi_drawn.append(fn)

    def connect_roi_changed(self, fn: Callable[[str], None]) -> None:
        self._on_roi_changed.append(fn)

    def refresh_from_analysis(self) -> None:
        """Pull auto-detected boundaries from PluginState into the ROI list."""
        pass  # TODO: implement in ROI batch

    def apply_to_all_images(self, roi_id: str) -> None:
        """Copy the given ROI to all open ImageEntry nodes."""
        pass  # TODO: implement in ROI batch
