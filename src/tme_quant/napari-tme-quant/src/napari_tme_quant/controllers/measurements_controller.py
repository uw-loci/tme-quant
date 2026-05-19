"""MeasurementsController — hierarchy tree and data table queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .state import PluginState


class MeasurementsController:
    """Populates the Measurements tree and table from TMEHierarchy.

    Full implementation in a future batch; stub provides the interface contract.

    Emits:
        measurement_row_selected(object_id: str)
        global_query_toggled(enabled: bool)
    """

    def __init__(self, state: "PluginState") -> None:
        self._state = state
        self._on_row_selected: list[Callable] = []
        self._global_query = False

    def connect_row_selected(self, fn: Callable[[str], None]) -> None:
        self._on_row_selected.append(fn)

    def refresh(self, image_id: Optional[str] = None) -> None:
        """Re-populate tree and table from hierarchy for the given image."""
        pass  # TODO: implement in measurements batch

    def on_committed(self, image_id: str, obj_type: str) -> None:
        self.refresh(image_id)

    def set_global_query(self, enabled: bool) -> None:
        self._global_query = enabled
        self.refresh()
