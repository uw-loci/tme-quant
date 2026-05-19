"""ProjectController — manages image list, type assignment, and pairing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .state import PluginState, ImageType


class ProjectController:
    """Translates Project widget actions → PluginState mutations.

    Emits:
        image_selected(image_id: str)
        image_type_changed(image_id: str, type: ImageType)
        pair_changed(fiber_id: str, cell_id: str)
    """

    def __init__(self, state: "PluginState") -> None:
        self._state = state
        self._on_image_selected: list[Callable] = []
        self._on_type_changed: list[Callable] = []
        self._on_pair_changed: list[Callable] = []

    # ── Signal subscription ────────────────────────────────────────────────────

    def connect_image_selected(self, fn: Callable[[str], None]) -> None:
        self._on_image_selected.append(fn)

    def connect_type_changed(self, fn: Callable[[str, "ImageType"], None]) -> None:
        self._on_type_changed.append(fn)

    def connect_pair_changed(self, fn: Callable[[str, str], None]) -> None:
        self._on_pair_changed.append(fn)

    # ── Actions ────────────────────────────────────────────────────────────────

    def select_image(self, image_id: str) -> None:
        self._state.active_image_id = image_id
        for fn in self._on_image_selected:
            fn(image_id)

    def set_image_type(self, image_id: str, image_type: "ImageType") -> None:
        self._state.image_types[image_id] = image_type
        for fn in self._on_type_changed:
            fn(image_id, image_type)

    def set_pair(self, fiber_id: str, cell_id: str) -> None:
        self._state.image_pairs[fiber_id] = cell_id
        for fn in self._on_pair_changed:
            fn(fiber_id, cell_id)
