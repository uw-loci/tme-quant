"""OrientationJ Fiji plugin interface."""

import numpy as np

from .orientation import BaseOrientationMethod
from .config import (
    OrientationParams, OrientationJParams, OrientationJResult,
)
from .fiji_bridge import FijiBridge


class OrientationJMethod(BaseOrientationMethod):
    """
    OrientationJ orientation analysis via the Fiji plugin.

    Computes per-pixel fiber orientation using the structure tensor or
    a gradient method, by calling Fiji's OrientationJ plugin.  Falls
    back to the NumPy structure-tensor implementation in FijiBridge when
    Fiji is unavailable.

    Reference
    ---------
    Rezakhaniha et al. (2012) Experimental investigation of collagen
    waviness and orientation in the arterial adventitia using confocal
    laser scanning microscopy. Biomech Model Mechanobiol 11:461–473.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fiji_bridge = FijiBridge()

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> OrientationJResult:
        """
        Analyse 2-D fiber orientation using OrientationJ.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.
        params : OrientationJParams or OrientationParams
            Analysis parameters.  If a plain ``OrientationParams`` is
            passed, OrientationJ defaults are used for mode-specific fields.

        Returns
        -------
        OrientationJResult
        """
        # Accept either the base class or the specific subclass
        p = params if isinstance(params, OrientationJParams) \
            else OrientationJParams(
                mode=params.mode,
                pixel_size=params.pixel_size,
                compute_statistics=params.compute_statistics,
                keep_values=params.keep_values,
            )

        # Build the plugin parameter dict from the typed params object
        plugin_params = {
            'gradient':       p.gradient_method,
            'min-coherency':  p.coherency_threshold,
            'min-energy':     p.energy_threshold,
            'sigma':          p.sigma_tensor,
        }

        # Call Fiji (or NumPy fallback)
        raw = self.fiji_bridge.call_orientationj(image, plugin_params)

        orientation_map: np.ndarray = raw['orientation']   # −90..90 degrees
        coherency_map:   np.ndarray = raw['coherency']     # 0..1
        energy_map:      np.ndarray = raw['energy']        # 0..1

        # Apply coherency threshold: suppress unreliable pixels
        if p.coherency_threshold > 0:
            orientation_map = orientation_map.copy()
            orientation_map[coherency_map < p.coherency_threshold] = np.nan

        # Compute statistics from valid pixels
        valid = orientation_map[~np.isnan(orientation_map)]
        stats = self._compute_statistics(valid, coherency_map)

        mean_coherency = float(
            np.nanmean(coherency_map[coherency_map >= p.coherency_threshold])
            if np.any(coherency_map >= p.coherency_threshold) else 0.0
        )
        mean_energy = float(
            np.nanmean(energy_map[energy_map >= p.energy_threshold])
            if np.any(energy_map >= p.energy_threshold) else 0.0
        )

        result = OrientationJResult(
            # Base fields
            orientation_map=orientation_map,
            alignment_map=coherency_map,
            mean_orientation=stats['mean_orientation'],
            alignment_score=stats['alignment_score'],
            mean_alignment=stats['alignment_score'],
            std_orientation=stats['std_orientation'],
            orientation_distribution=stats['orientation_distribution'],
            pixel_size=p.pixel_size,
            # OrientationJ-specific fields
            coherency_map=coherency_map,
            energy_map=energy_map,
            mean_coherency=mean_coherency,
            mean_energy=mean_energy,
        )

        # Optional colour survey
        if p.compute_color_survey and self.fiji_bridge.is_fiji_available():
            result.color_survey = self._request_color_survey(image, p)

        # Discard arrays not requested by keep_values
        if 'all' not in p.keep_values:
            if 'energy' not in p.keep_values:
                result.energy_map = None
            if 'alignment' not in p.keep_values:
                result.alignment_map = None
                result.coherency_map = None

        return result

    def supports_3d(self) -> bool:
        return False   # OrientationJ is 2-D only

    # ── Private helpers ───────────────────────────────────────────────────────

    def _request_color_survey(
        self,
        image: np.ndarray,
        params: OrientationJParams,
    ) -> np.ndarray | None:
        """Request OrientationJ's HSB colour survey image (stub)."""
        # Full implementation: call Fiji with 'colour-survey=true' flag
        # and retrieve the resulting RGB stack.
        return None