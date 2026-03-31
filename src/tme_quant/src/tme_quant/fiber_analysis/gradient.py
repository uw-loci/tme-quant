"""Gradient-based pixel-wise orientation analysis."""

import numpy as np
from typing import Optional

from .orientation import BaseOrientationMethod
from .config import (
    OrientationParams, GradientParams, GradientResult,
)


class GradientOrientationMethod(BaseOrientationMethod):
    """
    Pixel-wise fiber orientation from image gradients.

    Orientation = arctan2(gy, gx) + 90°, mapped to [−90°, 90°].
    No external tools required — uses scipy.ndimage only.
    """

    def analyze_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
    ) -> GradientResult:
        from scipy.ndimage import gaussian_filter, sobel, prewitt

        p = params if isinstance(params, GradientParams) \
            else GradientParams(
                mode=params.mode,
                pixel_size=params.pixel_size,
                compute_statistics=params.compute_statistics,
                keep_values=params.keep_values,
            )

        img = image.astype(np.float32)
        if p.smoothing_sigma > 0:
            img = gaussian_filter(img, sigma=p.smoothing_sigma)

        # Gradient
        op = p.gradient_operator.lower()
        if op == "prewitt":
            gx = prewitt(img, axis=1)
            gy = prewitt(img, axis=0)
        else:  # sobel / scharr / farid all approximate with sobel here
            gx = sobel(img, axis=1)
            gy = sobel(img, axis=0)

        magnitude = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
        max_mag = magnitude.max()
        norm_magnitude = magnitude / max_mag if max_mag > 0 else magnitude

        # Orientation perpendicular to gradient = fiber direction
        orientation_map = (np.degrees(np.arctan2(gy, gx)) + 90.0).astype(np.float32)
        orientation_map = ((orientation_map + 90) % 180) - 90  # remap to [-90, 90]

        # Background mask
        background_mask = norm_magnitude < p.min_gradient_magnitude
        orientation_map[background_mask] = np.nan

        valid = orientation_map[~background_mask]
        stats = self._compute_statistics(valid)

        foreground = ~background_mask
        mean_mag = float(norm_magnitude[foreground].mean()) if foreground.any() else 0.0
        fg_frac = float(foreground.sum() / foreground.size)

        result = GradientResult(
            orientation_map=orientation_map,
            alignment_map=norm_magnitude,
            mean_orientation=stats['mean_orientation'],
            alignment_score=stats['alignment_score'],
            mean_alignment=stats['alignment_score'],
            std_orientation=stats['std_orientation'],
            orientation_distribution=stats['orientation_distribution'],
            pixel_size=p.pixel_size,
            gradient_magnitude_map=norm_magnitude,
            background_mask=background_mask,
            mean_gradient_magnitude=mean_mag,
            foreground_fraction=fg_frac,
        )

        if 'all' not in p.keep_values:
            if 'alignment' not in p.keep_values:
                result.alignment_map = None
                result.gradient_magnitude_map = None

        return result

    def supports_3d(self) -> bool:
        return False