"""Windowed structure tensor orientation analysis."""

import numpy as np

from ..orientation import BaseOrientationMethod
from ..config import (
    OrientationParams, StructureTensorParams, StructureTensorResult,
)


class StructureTensorMethod(BaseOrientationMethod):
    """
    Windowed structure tensor fiber orientation.

    Computes the second-moment matrix of image gradients, smoothed
    with a Gaussian of sigma_spatial.  The dominant eigenvector gives
    the local orientation; the eigenvalue ratio gives coherency.

    No external tools required — uses scipy.ndimage only.
    """

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> StructureTensorResult:
        from scipy.ndimage import gaussian_filter

        p = params if isinstance(params, StructureTensorParams) \
            else StructureTensorParams(
                mode=params.mode,
                pixel_size=params.pixel_size,
                compute_statistics=params.compute_statistics,
                keep_values=params.keep_values,
            )

        img = image.astype(np.float64)

        # Inner-scale derivatives
        img_smooth = gaussian_filter(img, sigma=p.sigma_derivative)
        gy, gx = np.gradient(img_smooth)

        # Structure tensor components, smoothed with outer scale
        Jxx = gaussian_filter(gx * gx, sigma=p.sigma_spatial)
        Jxy = gaussian_filter(gx * gy, sigma=p.sigma_spatial)
        Jyy = gaussian_filter(gy * gy, sigma=p.sigma_spatial)

        # Dominant eigenvector orientation
        orientation_map = (
            0.5 * np.degrees(np.arctan2(2.0 * Jxy, Jxx - Jyy))
        ).astype(np.float32)

        # Eigenvalues (for coherency and optional output)
        diff  = Jxx - Jyy
        disc  = np.sqrt(diff ** 2 + 4.0 * Jxy ** 2)
        trace = Jxx + Jyy
        lambda_max = ((trace + disc) / 2.0).astype(np.float32)
        lambda_min = ((trace - disc) / 2.0).astype(np.float32)

        denom = np.where(trace > 1e-10, trace, 1e-10)
        coherency_map = np.clip(disc / denom, 0.0, 1.0).astype(np.float32)

        mean_anisotropy   = float(coherency_map.mean())
        isotropy_fraction = float((coherency_map < 0.1).sum() / coherency_map.size)

        stats = self._compute_statistics(orientation_map, coherency_map)

        result = StructureTensorResult(
            orientation_map=orientation_map,
            alignment_map=coherency_map,
            mean_orientation=stats['mean_orientation'],
            alignment_score=stats['alignment_score'],
            mean_alignment=stats['alignment_score'],
            std_orientation=stats['std_orientation'],
            orientation_distribution=stats['orientation_distribution'],
            pixel_size=p.pixel_size,
            coherency_map=coherency_map,
            lambda_max_map=lambda_max if p.compute_eigenvalues else None,
            lambda_min_map=lambda_min if p.compute_eigenvalues else None,
            mean_anisotropy=mean_anisotropy,
            isotropy_fraction=isotropy_fraction,
        )

        if 'all' not in p.keep_values:
            if 'alignment' not in p.keep_values:
                result.alignment_map  = None
                result.coherency_map  = None

        return result

    def analyze_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> StructureTensorResult:
        """Slice-by-slice 3-D wrapper."""
        slices = [self.analyze_2d(image[z], params) for z in range(image.shape[0])]
        vol = np.stack([s.orientation_map for s in slices])
        stats = self._compute_statistics(vol)
        return StructureTensorResult(
            orientation_map=vol,
            mean_orientation=stats['mean_orientation'],
            alignment_score=stats['alignment_score'],
            mean_alignment=stats['alignment_score'],
            std_orientation=stats['std_orientation'],
            orientation_distribution=stats['orientation_distribution'],
            pixel_size=params.pixel_size,
        )

    def supports_3d(self) -> bool:
        return True