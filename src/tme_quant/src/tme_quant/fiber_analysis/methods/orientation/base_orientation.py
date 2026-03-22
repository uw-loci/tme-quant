"""Base class for fiber orientation analysis methods."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from ...config.orientation_params import OrientationParams, OrientationResult


class BaseOrientationMethod(ABC):
    """Abstract base class for all fiber orientation analysis methods."""

    @abstractmethod
    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationResult:
        """Analyse fiber orientation in a 2-D image."""

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationResult:
        """Analyse fiber orientation in a 3-D image."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support 3-D analysis"
        )

    def supports_3d(self) -> bool:
        """Return True if this method implements analyze_3d."""
        return False

    # ── Shared statistics helper ──────────────────────────────────────────────

    def _compute_statistics(
        self,
        orientation_map: np.ndarray,
        coherency_map: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Compute circular statistics from an orientation map.

        Parameters
        ----------
        orientation_map : ndarray
            1-D array of valid orientation values in degrees, **or** a
            2-D / 3-D map (NaN pixels are ignored automatically).
        coherency_map : ndarray or None
            Not currently used in the calculation; reserved for
            coherency-weighted statistics in a future release.

        Returns
        -------
        dict with keys:
            ``mean_orientation``       – circular mean (degrees)
            ``alignment_score``        – mean resultant length R ∈ [0, 1]
            ``std_orientation``        – circular std dev (degrees)
            ``orientation_distribution`` – 36-bin histogram over [−90°, 90°]
        """
        flat = orientation_map.ravel()
        valid = flat[~np.isnan(flat)] if np.issubdtype(flat.dtype, np.floating) \
                else flat

        if valid.size == 0:
            return {
                'mean_orientation':         0.0,
                'alignment_score':          0.0,
                'std_orientation':          0.0,
                'orientation_distribution': np.zeros(36, dtype=np.int64),
            }

        angles_rad = np.deg2rad(valid)

        # Circular mean using the doubling trick (handles 180° periodicity)
        mean_x = float(np.mean(np.cos(2.0 * angles_rad)))
        mean_y = float(np.mean(np.sin(2.0 * angles_rad)))
        mean_orientation = float(np.rad2deg(np.arctan2(mean_y, mean_x) / 2.0))

        # Mean resultant length (order parameter R)
        alignment_score = float(np.sqrt(mean_x ** 2 + mean_y ** 2))

        # Circular standard deviation
        # Clamp R to (0, 1] — avoids sqrt of negative for near-zero R
        r_clamped = float(np.clip(alignment_score, 1e-10, 1.0 - 1e-10))
        std_orientation = float(np.rad2deg(np.sqrt(-2.0 * np.log(r_clamped))))

        # Orientation histogram
        hist, _ = np.histogram(valid, bins=36, range=(-90.0, 90.0))

        return {
            'mean_orientation':         mean_orientation,
            'alignment_score':          alignment_score,
            'std_orientation':          std_orientation,
            'orientation_distribution': hist,
        }