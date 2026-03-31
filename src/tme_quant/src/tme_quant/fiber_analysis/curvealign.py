# -*- coding: utf-8 -*-
"""
CurveAlign curvelet-based fiber orientation analysis.

3-D analysis semantics
-----------------------
``analyze_3d`` performs **true volumetric** analysis on a (Z, H, W) image.
It applies the curvelet transform to the full 3-D volume (via curvelops FDCT3D
when available), not slice-by-slice.

Slice-by-slice 2-D analysis of a 3-D stack should be performed by the caller
by iterating over slices and calling ``analyze_2d`` on each; it is deliberately
NOT exposed here as a 3-D method to avoid ambiguity.  This means:

  * ``analyze_3d`` returns an orientation volume of shape ``(Z, H, W)``.
  * Each voxel orientation reflects the locally dominant 3-D curvelet wedge.
  * When curvelops is unavailable, ``analyze_3d`` warns (via
    ``curvelet_transform_3d``) that it is using a degraded slice-by-slice
    fallback, but still returns a volume-shaped result.

References
----------
Bredfeldt et al. (2014) Computational segmentation of collagen fibers from
second-harmonic generation images of breast cancer.
J Biomed Opt 19(1):016007.

Peng et al. (2017) CurveAlign 4.0 — quantitative analysis of fiber
organization in the ECM. https://loci.wisc.edu/software/curvealign
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .orientation import BaseOrientationMethod
from .config import (
    OrientationParams, CurveAlignParams, CurveAlignResult,
)
from .utils.curvelet_utils import curvelet_transform_2d, curvelet_transform_3d


class CurveAlignOrientation(BaseOrientationMethod):
    """
    CurveAlign fiber orientation using the curvelet transform.

    Uses a sliding-window multi-scale curvelet decomposition to compute
    local fiber orientations in 2-D or 3-D SHG / fluorescence images.

    The dominant angular energy bin within each window gives the local
    orientation; the ratio of peak energy to total energy gives coherency.

    2-D mode
    ~~~~~~~~
    Slides a window of size ``params.window_size`` across the image with
    stride ``window_size * (1 - overlap)``.  Each window is independently
    transformed; orientation, coherency, and energy are assigned to all
    pixels covered by that window.

    3-D mode (volumetric)
    ~~~~~~~~~~~~~~~~~~~~~
    The curvelet transform is applied to the full volume via
    ``curvelet_transform_3d``, which dispatches to the curvelops FDCT3D
    backend for genuine volumetric analysis (preferred) or warns and uses
    a degraded slice-by-slice fallback.  The returned orientation volume
    has shape ``(Z, H, W)`` where each voxel orientation is derived from
    the dominant 3-D curvelet wedge in its neighbourhood.

    This is **not** the same as independently running ``analyze_2d`` on
    each slice.  The 3-D transform captures fiber orientations in all three
    spatial dimensions, including fibers tilted relative to the imaging plane.
    """

    # ── 2-D analysis ──────────────────────────────────────────────────────────

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> CurveAlignResult:
        """
        Analyse 2-D fiber orientation using CurveAlign.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.
        params : CurveAlignParams or OrientationParams
            Analysis parameters.

        Returns
        -------
        CurveAlignResult
        """
        if image.ndim != 2:
            raise ValueError(
                f"analyze_2d expects a 2-D image (H, W), got shape {image.shape}. "
                "For volumetric data use analyze_3d."
            )

        p = self._coerce_params(params)

        h, w   = image.shape
        stride = max(1, int(p.window_size * (1.0 - p.overlap)))

        orientation_map = np.full((h, w), np.nan, dtype=np.float32)
        coherency_map   = np.zeros((h, w), dtype=np.float32) if p.compute_coherency else None
        energy_map      = np.zeros((h, w), dtype=np.float32) if p.compute_energy    else None

        n_windows    = 0
        total_energy = 0.0

        for y in range(0, h - p.window_size + 1, stride):
            for x in range(0, w - p.window_size + 1, stride):
                window = image[y : y + p.window_size, x : x + p.window_size]

                coeffs = curvelet_transform_2d(
                    window,
                    n_levels=p.curvelet_levels,
                    n_angles=p.curvelet_angles,
                    use_matlab=p.use_matlab_backend,
                )

                orientation, coherency, energy = self._window_orientation(
                    coeffs, p.curvelet_angles
                )

                ys = slice(y, y + p.window_size)
                xs = slice(x, x + p.window_size)
                orientation_map[ys, xs] = orientation
                if coherency_map is not None:
                    coherency_map[ys, xs] = coherency
                if energy_map is not None:
                    energy_map[ys, xs] = energy

                n_windows    += 1
                total_energy += energy

        mean_energy = total_energy / max(n_windows, 1)
        valid       = orientation_map[~np.isnan(orientation_map)]
        stats       = self._compute_statistics(valid, coherency_map)

        result = CurveAlignResult(
            orientation_map          = orientation_map,
            alignment_map            = coherency_map,
            mean_orientation         = stats['mean_orientation'],
            alignment_score          = stats['alignment_score'],
            mean_alignment           = stats['alignment_score'],
            std_orientation          = stats['std_orientation'],
            orientation_distribution = stats['orientation_distribution'],
            pixel_size               = p.pixel_size,
            energy_map               = energy_map,
            n_windows_analyzed       = n_windows,
            mean_energy              = mean_energy,
        )

        if p.return_fiber_segments:
            result.fiber_segments = self._trace_fiber_segments(
                orientation_map, coherency_map
            )

        # Discard arrays not requested by keep_values
        if 'all' not in p.keep_values:
            if 'energy' not in p.keep_values:
                result.energy_map = None
            if 'alignment' not in p.keep_values:
                result.alignment_map = None
                result.coherency_map = None

        return result

    # ── 3-D volumetric analysis ───────────────────────────────────────────────

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> CurveAlignResult:
        """
        Analyse 3-D fiber orientation using volumetric CurveAlign.

        This method treats *image* as a single 3-D volume and applies the
        curvelet transform jointly across all three spatial dimensions.
        It is **not** slice-by-slice 2-D analysis repeated for each z-plane.

        The curvelet transform dispatches to:

        1. **curvelops FDCT3D** (preferred) — genuine volumetric transform
           that detects fiber orientations across all three axes, including
           fibers tilted out of the imaging plane.
        2. **NumPy fallback** — degraded slice-by-slice approximation.  A
           ``UserWarning`` is emitted to make this limitation explicit.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
            3-D grayscale volume.
        params : CurveAlignParams or OrientationParams

        Returns
        -------
        CurveAlignResult
            ``orientation_map`` has shape ``(Z, H, W)``.
            ``n_windows_analyzed`` reflects the sliding-window count used
            to aggregate orientation across the volume.
        """
        if image.ndim != 3:
            raise ValueError(
                f"analyze_3d expects a 3-D volume (Z, H, W), got shape {image.shape}."
            )

        p = self._coerce_params(params)
        z, h, w = image.shape

        # ── Volumetric curvelet transform ────────────────────────────────────
        # curvelet_transform_3d returns (Z, H, W, n_angles).
        # It dispatches to curvelops FDCT3D for true 3-D analysis, or
        # warns and falls back to slice-by-slice approximation.
        coeffs_vol = curvelet_transform_3d(
            image,
            n_levels   = p.curvelet_levels,
            n_angles   = p.curvelet_angles,
            use_matlab = p.use_matlab_backend,
        )

        # ── Derive per-voxel orientation from 3-D curvelet wedges ────────────
        # Apply a sliding-window aggregation in XY; Z is processed in full.
        orientation_vol = np.full((z, h, w), np.nan, dtype=np.float32)
        coherency_vol   = np.zeros((z, h, w), dtype=np.float32) if p.compute_coherency else None
        energy_vol      = np.zeros((z, h, w), dtype=np.float32) if p.compute_energy    else None

        stride      = max(1, int(p.window_size * (1.0 - p.overlap)))
        n_windows   = 0
        total_energy = 0.0

        for zi in range(z):
            for y in range(0, h - p.window_size + 1, stride):
                for x in range(0, w - p.window_size + 1, stride):
                    # Extract the corresponding window from the 3-D coefficient array
                    win_coeffs = coeffs_vol[zi,
                                            y : y + p.window_size,
                                            x : x + p.window_size, :]

                    orientation, coherency, energy = self._window_orientation(
                        win_coeffs, p.curvelet_angles
                    )

                    ys = slice(y, y + p.window_size)
                    xs = slice(x, x + p.window_size)
                    orientation_vol[zi, ys, xs] = orientation
                    if coherency_vol is not None:
                        coherency_vol[zi, ys, xs] = coherency
                    if energy_vol is not None:
                        energy_vol[zi, ys, xs] = energy

                    n_windows    += 1
                    total_energy += energy

        mean_energy = total_energy / max(n_windows, 1)
        valid       = orientation_vol[~np.isnan(orientation_vol)]
        stats       = self._compute_statistics(valid)

        return CurveAlignResult(
            orientation_map          = orientation_vol,
            alignment_map            = coherency_vol,
            mean_orientation         = stats['mean_orientation'],
            alignment_score          = stats['alignment_score'],
            mean_alignment           = stats['alignment_score'],
            std_orientation          = stats['std_orientation'],
            orientation_distribution = stats['orientation_distribution'],
            pixel_size               = p.pixel_size,
            energy_map               = energy_vol,
            n_windows_analyzed       = n_windows,
            mean_energy              = mean_energy,
        )

    def supports_3d(self) -> bool:
        """
        Return True — CurveAlign 3-D volumetric analysis is always available.

        Note that the quality of the 3-D result depends on the backend:
        ``curvelops`` provides genuine volumetric curvelet analysis, while the
        NumPy fallback uses a slice-by-slice approximation.  Check backend
        availability with::

            from tme_quant.fiber_analysis.utils.curvelet_utils import available_backends
            print(available_backends())
        """
        return True

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _coerce_params(params: OrientationParams) -> CurveAlignParams:
        """Return a CurveAlignParams; coerce from base class if needed."""
        if isinstance(params, CurveAlignParams):
            return params
        return CurveAlignParams(
            mode              = params.mode,
            pixel_size        = params.pixel_size,
            compute_statistics= params.compute_statistics,
            keep_values       = params.keep_values,
        )

    @staticmethod
    def _window_orientation(
        coeffs: np.ndarray,
        n_angles: int,
    ) -> Tuple[float, float, float]:
        """
        Derive orientation, coherency, and energy from curvelet coefficients.

        Parameters
        ----------
        coeffs : ndarray, shape (..., n_angles)
            Curvelet energy array for a single window.
        n_angles : int

        Returns
        -------
        orientation : float   degrees in [−90, +90]
        coherency   : float   in [0, 1]  (peak bin fraction)
        energy      : float   total curvelet energy
        """
        angle_energies = np.array([
            float(np.sum(np.abs(coeffs[..., k]) ** 2))
            for k in range(n_angles)
        ])
        total = float(np.sum(angle_energies))
        if total < 1e-12:
            return 0.0, 0.0, 0.0

        dominant    = int(np.argmax(angle_energies))
        orientation = float(dominant * 180.0 / n_angles) - 90.0
        coherency   = float(angle_energies[dominant] / total)
        return orientation, coherency, total

    @staticmethod
    def _trace_fiber_segments(
        orientation_map: np.ndarray,
        coherency_map: Optional[np.ndarray],
        min_coherency: float = 0.3,
    ):
        """
        Stub: return empty list until full fiber segment tracing is integrated.

        TODO: Implement orientation-guided fiber segment tracing that connects
        high-coherency regions with consistent orientation into fiber segments.
        This is analogous to the segment tracing in CurveAlign 4.0.
        """
        return []