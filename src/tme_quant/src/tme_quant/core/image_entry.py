"""
ImageEntry — a hierarchy-aware container for a single TME image.

Sits between the project root and annotation/tumor/fiber objects in the
TME hierarchy:

    TMEProject (root)
    └── ImageEntry          ← this class
        ├── TumorRegion
        │   ├── FiberObject
        │   └── CellObject
        └── StromaRegion

Design goals
------------
- Store raw image data (2-D grayscale, 2-D multichannel, 3-D, or
  time-lapse) as a numpy array.
- Provide ``get_channel_data(channel)`` so that calling code can request
  a named channel (e.g. ``'collagen'``, ``'DAPI'``, ``'SHG'``) without
  needing to know the axis layout.
- Carry acquisition metadata (pixel size, magnification, modality …)
  consistent with ``TMEMetadata``.
- Support lazy loading: image data may be ``None`` until ``load()`` is
  called (useful for large projects where not every image is needed at
  once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base_models import TMEObject, TMEType, ObjectType, TMEMetadata


# ---------------------------------------------------------------------------
# ImageEntry
# ---------------------------------------------------------------------------

class ImageEntry(TMEObject):
    """
    A single image in a TME project, with named-channel access.

    Array layout convention
    -----------------------
    - 2-D grayscale      : ``(H, W)``
    - 2-D multichannel   : ``(H, W, C)``
    - 3-D grayscale      : ``(Z, H, W)``
    - 3-D multichannel   : ``(Z, H, W, C)``
    - Time-lapse 2-D     : ``(T, H, W)`` or ``(T, H, W, C)``

    The ``ndim`` property reports the spatial dimensionality (2 or 3)
    independent of whether a channel or time axis is present.

    Channel mapping
    ---------------
    ``channel_names`` is an ordered list of channel labels.  If provided,
    ``get_channel_data('collagen')`` extracts the matching slice along the
    last axis.  If the image is single-channel or channel_names is empty,
    ``get_channel_data`` always returns the full array.

    Examples
    --------
    >>> entry = ImageEntry(
    ...     object_id='img_001',
    ...     name='SHG_collagen',
    ...     image_data=np.zeros((512, 512, 2), dtype=np.float32),
    ...     channel_names=['SHG', 'collagen'],
    ...     pixel_size=(0.5, 0.5),   # µm/pixel
    ... )
    >>> collagen = entry.get_channel_data('collagen')  # shape (512, 512)
    >>> shg      = entry.get_channel_data('SHG')       # shape (512, 512)
    >>> gray     = entry.get_channel_data()             # full array
    """

    def __init__(
        self,
        object_id: str = "",
        name: str = "",
        # Image data (may be None for lazy-loaded entries)
        image_data: Optional[np.ndarray] = None,
        # Source file
        path: Optional[Union[str, Path]] = None,
        # Channel layout
        channel_names: Optional[List[str]] = None,
        # Acquisition metadata
        pixel_size: Optional[Tuple[float, ...]] = None,   # (x, y[, z]) µm/pixel
        magnification: Optional[float] = None,
        modality: str = "",                                # e.g. "SHG", "BF", "IF"
        # Extended metadata (reuses TMEMetadata dataclass)
        tme_metadata: Optional[TMEMetadata] = None,
        # Additional arbitrary metadata
        metadata: Optional[Dict[str, Any]] = None,
        # Hierarchy parent
        parent: Optional[TMEObject] = None,
        # Optional pre-computed mask (same H×W as image)
        mask_data: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            object_id=object_id,
            name=name,
            tme_type=TMEType.IMAGE,
            object_type=ObjectType.REGION,
            parent=parent,
            metadata=metadata,
        )

        # --- image data ---
        self._image_data: Optional[np.ndarray] = image_data
        self.mask_data: Optional[np.ndarray] = mask_data

        # --- file path ---
        self.path: Optional[Path] = Path(path) if path is not None else None

        # --- channel names ---
        self.channel_names: List[str] = channel_names if channel_names is not None else []

        # --- acquisition metadata ---
        self.pixel_size: Optional[Tuple[float, ...]] = pixel_size
        self.magnification: Optional[float] = magnification
        self.modality: str = modality

        # --- extended metadata ---
        self.tme_metadata: Optional[TMEMetadata] = tme_metadata

    # ------------------------------------------------------------------
    # Image data access
    # ------------------------------------------------------------------

    @property
    def image_data(self) -> Optional[np.ndarray]:
        """Raw image array, or None if not yet loaded."""
        return self._image_data

    @image_data.setter
    def image_data(self, data: Optional[np.ndarray]) -> None:
        self._image_data = data

    def is_loaded(self) -> bool:
        """Return True if image data is currently in memory."""
        return self._image_data is not None

    def get_channel_data(
        self,
        channel: Optional[Union[str, int]] = None,
    ) -> np.ndarray:
        """
        Return image data for a named or indexed channel.

        Parameters
        ----------
        channel:
            - ``str``  : channel name (looked up in ``channel_names``)
            - ``int``  : zero-based channel index along the last axis
            - ``None`` : return the full array as-is

        Returns
        -------
        np.ndarray
            For a multichannel array ``(H, W, C)`` the returned slice has
            shape ``(H, W)``.  For a single-channel or no-channel array
            the full array is returned unchanged.

        Raises
        ------
        ValueError
            If the channel name is not in ``channel_names``.
        IndexError
            If the channel index is out of range.
        RuntimeError
            If image data has not been loaded yet.
        """
        if self._image_data is None:
            raise RuntimeError(
                f"ImageEntry '{self.object_id}': image data is not loaded. "
                "Call load() first or supply image_data at construction."
            )

        arr = self._image_data

        # No channel requested — return full array
        if channel is None:
            return arr

        # Resolve string → int index
        if isinstance(channel, str):
            if channel in self.channel_names:
                idx = self.channel_names.index(channel)
            else:
                # Graceful fallback: if only one channel or no names defined,
                # treat any string request as "give me the whole array"
                if arr.ndim == 2 or not self.channel_names:
                    return arr
                raise ValueError(
                    f"Channel '{channel}' not found in channel_names "
                    f"{self.channel_names} for image '{self.object_id}'."
                )
        else:
            idx = int(channel)

        # Extract along last axis (channel axis convention)
        if arr.ndim < 3:
            # 2-D grayscale — ignore index, return as-is
            return arr

        n_channels = arr.shape[-1]
        if idx >= n_channels or idx < -n_channels:
            raise IndexError(
                f"Channel index {idx} out of range for image with "
                f"{n_channels} channels."
            )
        return arr[..., idx]

    def get_all_channels(self) -> Dict[str, np.ndarray]:
        """
        Return a dict mapping channel name → 2-D array for all channels.

        If ``channel_names`` is empty, keys are ``'0'``, ``'1'``, … .
        """
        if self._image_data is None:
            raise RuntimeError(
                f"ImageEntry '{self.object_id}': image data is not loaded."
            )
        arr = self._image_data
        if arr.ndim < 3:
            label = self.channel_names[0] if self.channel_names else "0"
            return {label: arr}
        n = arr.shape[-1]
        names = (
            self.channel_names if len(self.channel_names) == n
            else [str(i) for i in range(n)]
        )
        return {name: arr[..., i] for i, name in enumerate(names)}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(
        self,
        path: Optional[Union[str, Path]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> "ImageEntry":
        """
        Load image data from disk into memory.

        Supports TIFF (including multi-page / OME-TIFF), PNG, and common
        formats via tifffile + imageio.  Returns ``self`` for chaining.

        Parameters
        ----------
        path:
            Override the stored ``self.path``.
        channel_names:
            Override ``self.channel_names`` after loading.
        """
        load_path = Path(path) if path is not None else self.path
        if load_path is None:
            raise ValueError(
                f"ImageEntry '{self.object_id}': no path specified for loading."
            )

        suffix = load_path.suffix.lower()
        try:
            if suffix in (".tif", ".tiff"):
                import tifffile
                self._image_data = tifffile.imread(str(load_path))
            else:
                import imageio
                self._image_data = np.array(imageio.imread(str(load_path)))
        except ImportError as e:
            raise ImportError(
                f"Could not load '{load_path}': {e}. "
                "Install tifffile or imageio to enable image loading."
            ) from e

        self.path = load_path
        if channel_names is not None:
            self.channel_names = channel_names
        return self

    def unload(self) -> "ImageEntry":
        """Release image data from memory. Returns self for chaining."""
        self._image_data = None
        return self

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the image array, or None if not loaded."""
        return self._image_data.shape if self._image_data is not None else None

    @property
    def ndim(self) -> int:
        """
        Spatial dimensionality: 2 for (H, W[, C]) or 3 for (Z, H, W[, C]).
        Returns 0 if not loaded.
        """
        if self._image_data is None:
            return 0
        arr = self._image_data
        n = arr.ndim
        # Subtract channel axis if channel_names are defined and last dim matches
        has_channel = (
            len(self.channel_names) > 0
            and arr.ndim >= 3
            and arr.shape[-1] == len(self.channel_names)
        )
        spatial_ndim = n - 1 if has_channel else n
        return min(spatial_ndim, 3)

    @property
    def n_channels(self) -> int:
        """Number of channels (1 if 2-D grayscale or not loaded)."""
        if self._image_data is None:
            return 0
        if self._image_data.ndim < 3:
            return 1
        if self.channel_names:
            return len(self.channel_names)
        return self._image_data.shape[-1]

    @property
    def spatial_shape(self) -> Optional[Tuple[int, int]]:
        """``(height, width)`` of the image, or None if not loaded."""
        if self._image_data is None:
            return None
        arr = self._image_data
        if arr.ndim == 2:
            return (arr.shape[0], arr.shape[1])
        # (H, W, C) or (Z, H, W) — height/width are axes -3 and -2
        return (arr.shape[-3] if arr.ndim >= 3 else arr.shape[0],
                arr.shape[-2] if arr.ndim >= 3 else arr.shape[1])

    # ------------------------------------------------------------------
    # Pixel-size helpers
    # ------------------------------------------------------------------

    def pixel_to_micron(self, pixels: float, axis: int = 0) -> float:
        """Convert pixel distance to microns along the given axis (0=x, 1=y)."""
        if self.pixel_size is None:
            raise ValueError(
                f"ImageEntry '{self.object_id}': pixel_size is not set."
            )
        return pixels * self.pixel_size[min(axis, len(self.pixel_size) - 1)]

    def micron_to_pixel(self, microns: float, axis: int = 0) -> float:
        """Convert micron distance to pixels along the given axis."""
        if self.pixel_size is None:
            raise ValueError(
                f"ImageEntry '{self.object_id}': pixel_size is not set."
            )
        scale = self.pixel_size[min(axis, len(self.pixel_size) - 1)]
        if scale == 0:
            raise ValueError("pixel_size contains a zero value.")
        return microns / scale

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Shallow serialisation (image array is NOT included)."""
        d = super().to_dict()
        d.update({
            "path": str(self.path) if self.path else None,
            "channel_names": self.channel_names,
            "pixel_size": list(self.pixel_size) if self.pixel_size else None,
            "magnification": self.magnification,
            "modality": self.modality,
            "shape": list(self.shape) if self.shape else None,
            "is_loaded": self.is_loaded(),
        })
        return d

    def __repr__(self) -> str:
        loaded = f"shape={self.shape}" if self.is_loaded() else "not loaded"
        ch = f", channels={self.channel_names}" if self.channel_names else ""
        return (
            f"ImageEntry(object_id={self.object_id!r}, "
            f"{loaded}{ch})"
        )


__all__ = ["ImageEntry"]