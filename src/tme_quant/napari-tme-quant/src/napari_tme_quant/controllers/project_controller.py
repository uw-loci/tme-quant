"""ProjectController — manages image list, type assignment, and pairing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .state import PluginState, ImageType

# Filename keyword → ImageType auto-detection (case-insensitive substring match).
# Fiber covers any microscopic fiber image: SHG, birefringent/PLM, bright-field
# collagen stain, etc.
_FIBER_KEYWORDS = {"shg", "collagen", "fiber", "fibre", "biref", "plm", "brightfield"}
_CELL_KEYWORDS  = {"he", "dapi", "dab", "cell", "nuclei", "nucleus"}
_MASK_KEYWORDS  = {"mask", "boundary", "annotation", "label", "seg"}
_2CH_KEYWORDS   = {"2ch", "merged", "combined"}

# Supported load formats (TIFF default; PNG/JPEG also accepted)
IMAGE_FILE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*)"


def detect_image_type(path: str) -> "ImageType":
    """Infer ImageType from filename keywords (case-insensitive)."""
    from .state import ImageType
    stem = Path(path).stem.lower()
    if any(kw in stem for kw in _FIBER_KEYWORDS):
        return ImageType.FIBER
    if any(kw in stem for kw in _CELL_KEYWORDS):
        return ImageType.CELL
    if any(kw in stem for kw in _MASK_KEYWORDS):
        return ImageType.MASK
    if any(kw in stem for kw in _2CH_KEYWORDS):
        return ImageType.TWO_CHANNEL
    return ImageType.UNKNOWN


class ProjectController:
    """Translates Project widget actions → PluginState mutations.

    Emits:
        image_selected(image_id: str)
        image_type_changed(image_id: str, type: ImageType)
        pair_changed(fiber_id: str, cell_id: str)
        image_added(image_id: str)
        image_removed(image_id: str)
    """

    def __init__(self, state: "PluginState") -> None:
        self._state = state
        self._on_image_selected: list[Callable] = []
        self._on_type_changed: list[Callable] = []
        self._on_pair_changed: list[Callable] = []
        self._on_image_added: list[Callable] = []
        self._on_image_removed: list[Callable] = []

    # ── Signal subscription ────────────────────────────────────────────────────

    def connect_image_selected(self, fn: Callable[[str], None]) -> None:
        self._on_image_selected.append(fn)

    def connect_type_changed(self, fn: Callable[[str, "ImageType"], None]) -> None:
        self._on_type_changed.append(fn)

    def connect_pair_changed(self, fn: Callable[[str, str], None]) -> None:
        self._on_pair_changed.append(fn)

    def connect_image_added(self, fn: Callable[[str], None]) -> None:
        self._on_image_added.append(fn)

    def connect_image_removed(self, fn: Callable[[str], None]) -> None:
        self._on_image_removed.append(fn)

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

    def add_image(
        self,
        path: str,
        image_type: "ImageType",
        viewer,
    ) -> str:
        """Load an image file, add it as a napari layer, and register in state.

        Supports TIFF, PNG, JPEG. Returns the image_id (= file stem).
        If a layer with the same name already exists, it is reused.
        """
        import imageio.v3 as iio

        image_id = Path(path).stem
        data = np.asarray(iio.imread(path), dtype=np.float32)

        # Add to napari (reuse existing layer if already loaded)
        existing = next((l for l in viewer.layers if l.name == image_id), None)
        if existing is None:
            viewer.add_image(data, name=image_id)

        self._state.images[image_id] = data
        self._state.image_types[image_id] = image_type
        self._state.image_paths[image_id] = str(Path(path).resolve())

        for fn in self._on_type_changed:
            fn(image_id, image_type)
        for fn in self._on_image_added:
            fn(image_id)
        return image_id

    def remove_image(self, image_id: str, viewer) -> None:
        """Remove an image from state and from the napari viewer."""
        layer = next((l for l in viewer.layers if l.name == image_id), None)
        if layer is not None:
            viewer.layers.remove(layer)

        # Also remove all plugin layers for this image
        prefix = f"{image_id} :: "
        to_remove = [l for l in viewer.layers if l.name.startswith(prefix)]
        for layer in to_remove:
            viewer.layers.remove(layer)

        for d in (
            self._state.images,
            self._state.image_types,
            self._state.image_paths,
            self._state.curvealign_pipeline_results,
            self._state.fiber_results,
            self._state.per_image_params,
        ):
            d.pop(image_id, None)

        if self._state.active_image_id == image_id:
            self._state.active_image_id = None

        for fn in self._on_image_removed:
            fn(image_id)
