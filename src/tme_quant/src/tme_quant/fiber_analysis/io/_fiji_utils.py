# -*- coding: utf-8 -*-
"""
Shared utilities for Fiji bridge modules.

Provides:
  - FijiBackendMixin  — backend detection, pyimagej init, version compat
  - normalise_image   — float32 [0,1] normalisation
  - contrast_value    — normalise contrast threshold to 0-255 scale
  - contrast_to_fraction — normalise contrast threshold to 0-1 fraction
  - make_color_survey — synthesise OrientationJ-style HSB colour survey
  - order_by_nearest_neighbour — order skeleton coords into a sequential path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


class FijiBackendMixin:
    """
    Mixin that provides backend detection and pyimagej lifecycle management
    for OrientationJBridge and RidgeDetectionBridge.

    Subclasses must call ``super().__init__(fiji_path)`` (or set
    ``self._fiji_path`` themselves) before calling any other method.
    """

    def __init__(self, fiji_path: Optional[str] = None) -> None:
        self._fiji_path: str = fiji_path or os.environ.get("FIJI_PATH", "")
        self._ij = None                        # lazy pyimagej handle
        self._ij_version: Optional[int] = None  # 1 or 2
        self._backend: str = self._detect_backend()

    # ── Backend detection ─────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """Return ``'pyimagej'``, ``'subprocess'``, or ``'numpy'``."""
        if self._fiji_path and os.path.isdir(self._fiji_path):
            try:
                import imagej  # type: ignore  # noqa: F401
                return "pyimagej"
            except ImportError:
                pass
            if self._find_fiji_executable() is not None:
                return "subprocess"
        return "numpy"

    def _find_fiji_executable(self) -> Optional[Path]:
        """Return the Fiji executable path for the current OS, or None."""
        root = Path(self._fiji_path)
        for candidate in [
            root / "ImageJ-linux64",
            root / "ImageJ-linux32",
            root / "ImageJ-win64.exe",
            root / "ImageJ-win32.exe",
            root / "Contents" / "MacOS" / "ImageJ-macosx",
        ]:
            if candidate.exists():
                return candidate
        return None

    def is_fiji_available(self) -> bool:
        """Return True if a real Fiji backend (pyimagej or subprocess) is active."""
        return self._backend in ("pyimagej", "subprocess")

    # ── pyimagej lifecycle ────────────────────────────────────────────────────

    def _get_ij(self):
        """Initialise and return the pyimagej ImageJ instance (lazy)."""
        if self._ij is None:
            import imagej  # type: ignore
            self._ij = imagej.init(self._fiji_path, mode="headless")
            # Detect API version: 2.x has ij.py.to_java; 1.x has to_imageplus
            self._ij_version = 2 if hasattr(self._ij.py, "to_java") else 1
        return self._ij

    def _to_imageplus(self, ij: Any, arr: np.ndarray) -> Any:
        """Convert ndarray to ImagePlus, handling pyimagej v1/v2 API."""
        if self._ij_version == 2:
            return ij.py.to_java(arr)
        return ij.py.to_imageplus(arr)

    def _get_results_table(self, ij: Any) -> Any:
        """Return the active ResultsTable, handling pyimagej v1/v2 API."""
        if self._ij_version == 2:
            return ij.ResultsTable.getActiveTable()
        return ij.ResultsTable.getResultsTable()

    def close(self) -> None:
        """Shut down the pyimagej JVM if it was started."""
        if self._ij is not None:
            try:
                self._ij.dispose()
            except Exception:
                pass
            self._ij = None


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function utilities (no class needed)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_image(image: np.ndarray) -> np.ndarray:
    """Return a float32 copy of *image* normalised to [0, 1]."""
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def contrast_value(value: Any) -> float:
    """
    Normalise *value* to the 0–255 scale Fiji expects.

    Values ≥ 1 are assumed to already be on the 0–255 scale.
    Values < 1 are assumed to be a 0–1 fraction and are multiplied by 255.
    """
    v = float(value)
    return v if v >= 1.0 else v * 255.0


def contrast_to_fraction(value: Any) -> float:
    """
    Normalise *value* to a 0–1 fraction for internal use.

    Values ≥ 1 are divided by 255.  Values < 1 are returned as-is.
    """
    v = float(value)
    return v / 255.0 if v >= 1.0 else v


def make_color_survey(
    orientation_map: np.ndarray,
    coherency_map: np.ndarray,
) -> np.ndarray:
    """
    Synthesise an OrientationJ-style HSB colour-survey image.

    Encoding
    --------
    Hue   = orientation mapped from [−90°, +90°] → [0°, 360°]
    Sat   = coherency (0 = grey, 1 = fully saturated)
    Value = 1.0 for valid pixels, 0 for NaN orientations

    Returns
    -------
    ndarray of shape (H, W, 3), dtype uint8, RGB colour order.
    """
    import colorsys

    h_map = (orientation_map + 90.0) / 180.0   # hue in [0, 1]
    s_map = np.clip(coherency_map, 0.0, 1.0)
    valid = ~np.isnan(orientation_map)

    rgb = np.zeros((*orientation_map.shape, 3), dtype=np.uint8)
    for r in range(orientation_map.shape[0]):
        for c in range(orientation_map.shape[1]):
            if valid[r, c]:
                rv, gv, bv = colorsys.hsv_to_rgb(
                    float(h_map[r, c]), float(s_map[r, c]), 1.0
                )
                rgb[r, c] = (int(rv * 255), int(gv * 255), int(bv * 255))
    return rgb


def order_by_nearest_neighbour(coords: np.ndarray) -> np.ndarray:
    """
    Order an unordered set of skeleton coordinates into a sequential path.

    Uses a greedy nearest-neighbour walk starting from the first point.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)

    Returns
    -------
    ndarray, shape (N, 2)
    """
    if len(coords) <= 2:
        return coords

    remaining = list(range(len(coords)))
    path      = [remaining.pop(0)]

    while remaining:
        current = coords[path[-1]]
        dists   = np.linalg.norm(coords[remaining] - current, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        path.append(nearest)
        remaining.remove(nearest)

    return coords[path]


__all__ = [
    "FijiBackendMixin",
    "normalise_image",
    "contrast_value",
    "contrast_to_fraction",
    "make_color_survey",
    "order_by_nearest_neighbour",
]