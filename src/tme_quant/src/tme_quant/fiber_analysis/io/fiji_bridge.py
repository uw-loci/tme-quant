# -*- coding: utf-8 -*-
"""
Fiji / ImageJ bridge — coordinator module.

This module is the single import point for all Fiji plugin functionality.
The implementation is split across plugin-specific files:

    fiji_bridge.py            ← this file (public API / coordinator)
    orientationj_bridge.py    ← OrientationJ: orientation, coherency, energy
    ridge_detection_bridge.py ← Ridge Detection: line coordinates, width map
    _fiji_utils.py            ← shared backend mixin and pure utilities

:class:`FijiBridge` wraps both plugin bridges and exposes the original
``call_orientationj`` / ``call_ridge_detection`` interface so that
``orientationj.py`` and ``ridge_detection.py`` require no changes.

Installation
------------
    pip install pyimagej scyjava          # pyimagej backend
    export FIJI_PATH=/path/to/Fiji.app   # either Fiji backend
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .orientationj_bridge    import OrientationJBridge
from .ridge_detection_bridge import RidgeDetectionBridge
from ._fiji_utils import (        # re-export for convenience
    normalise_image,
    contrast_value,
    contrast_to_fraction,
    make_color_survey,
    order_by_nearest_neighbour,
)


class FijiBridge:
    """
    Coordinator that delegates to :class:`OrientationJBridge` and
    :class:`RidgeDetectionBridge`.

    Callers (``orientationj.py``, ``ridge_detection.py``) only need:

    * :meth:`is_fiji_available`
    * :meth:`call_orientationj`
    * :meth:`call_ridge_detection`

    The individual plugin bridges are also accessible directly via
    :attr:`orientationj` and :attr:`ridge_detection` for advanced use.

    Parameters
    ----------
    fiji_path : str or None
        Path to ``Fiji.app``.  Falls back to ``FIJI_PATH`` env var,
        then to NumPy-only mode.
    """

    def __init__(self, fiji_path: Optional[str] = None) -> None:
        self.orientationj    = OrientationJBridge(fiji_path)
        self.ridge_detection = RidgeDetectionBridge(fiji_path)

    # ── Public interface (used by method files) ───────────────────────────────

    def is_fiji_available(self) -> bool:
        """
        Return True if either plugin bridge has a real Fiji backend active.

        Both bridges share the same ``FIJI_PATH`` so their backends will
        always agree; we check the OrientationJ bridge as the canonical one.
        """
        return self.orientationj.is_fiji_available()

    @property
    def _backend(self) -> str:
        """Active backend name (``'pyimagej'``, ``'subprocess'``, or ``'numpy'``)."""
        return self.orientationj._backend

    def call_orientationj(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Run OrientationJ and return orientation, coherency, and energy maps.

        Delegates to :meth:`OrientationJBridge.run`.
        See :class:`~.orientationj_bridge.OrientationJBridge` for full
        parameter and return-value documentation.
        """
        return self.orientationj.run(image, params)

    def call_ridge_detection(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run Ridge Detection and return detected line coordinates.

        Delegates to :meth:`RidgeDetectionBridge.run`.
        See :class:`~.ridge_detection_bridge.RidgeDetectionBridge` for full
        parameter and return-value documentation.
        """
        return self.ridge_detection.run(image, params)

    def close(self) -> None:
        """Shut down any pyimagej JVM instances held by the plugin bridges."""
        self.orientationj.close()
        self.ridge_detection.close()


__all__ = ["FijiBridge"]