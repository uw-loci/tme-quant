"""Ridge Detection fiber extraction — Fiji plugin with NumPy fallback."""

import numpy as np
from typing import List

from .base_extraction import BaseExtractionMethod
from ...config.extraction_params import (
    ExtractionParams, RidgeDetectionParams, RidgeDetectionResult, FiberProperties,
)
from ...io.fiji_bridge import FijiBridge
from ...utils.geometry_utils import compute_fiber_properties


class RidgeDetectionMethod(BaseExtractionMethod):
    """
    Ridge Detection fiber extraction via the Fiji plugin.

    Calls Fiji's Ridge Detection plugin to find curvilinear structures.
    Falls back to a Hessian/Frangi NumPy implementation (via FijiBridge)
    when Fiji is not available — no RuntimeError is raised.

    Reference
    ---------
    Steger (1998) An unbiased detector of curvilinear structures.
    IEEE PAMI 20(2):113–125.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fiji_bridge = FijiBridge()

    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> RidgeDetectionResult:
        """
        Extract fibers using Ridge Detection (or NumPy Hessian fallback).

        Parameters
        ----------
        image : ndarray, shape (H, W)
        params : RidgeDetectionParams or ExtractionParams

        Returns
        -------
        RidgeDetectionResult
        """
        p = params if isinstance(params, RidgeDetectionParams) \
            else RidgeDetectionParams(
                mode=params.mode,
                pixel_size=params.pixel_size,
                min_fiber_length=params.min_fiber_length,
                max_fiber_length=params.max_fiber_length,
                min_fiber_width=params.min_fiber_width,
                max_fiber_width=params.max_fiber_width,
                measure_length=params.measure_length,
                measure_width=params.measure_width,
                measure_straightness=params.measure_straightness,
                measure_angle=params.measure_angle,
                measure_curvature=params.measure_curvature,
                extract_centerlines=params.extract_centerlines,
            )

        # Build plugin parameter dict from typed params
        plugin_params = {
            'line_width':    p.ridge_sigma * 2,
            'high_contrast': p.upper_threshold * 255,
            'low_contrast':  p.lower_threshold * 255,
            'extend_line':   p.extend_line,
            'make_binary':   False,
        }

        # Call Fiji or NumPy fallback — FijiBridge never raises on unavailability
        raw = self.fiji_bridge.call_ridge_detection(image, plugin_params)
        line_coords: List[np.ndarray] = raw['lines']

        # Build FiberProperties for each detected line
        fibers: List[FiberProperties] = []
        for fiber_id, coords in enumerate(line_coords):
            if coords is None or len(coords) < 2:
                continue

            props = compute_fiber_properties(
                coords, image, p.pixel_size, p.fiber_width_range
            )

            # Apply size filters
            if not (p.min_fiber_length <= props['length'] <= p.max_fiber_length):
                continue

            fibers.append(FiberProperties(
                fiber_id=fiber_id,
                length=props['length'],
                width=props['width'],
                straightness=props['straightness'],
                angle=props['angle'],
                curvature=props['curvature'] if p.measure_curvature else 0.0,
                centerline=coords if p.extract_centerlines else None,
                aspect_ratio=(
                    props['length'] / props['width']
                    if props['width'] > 0 else None
                ),
            ))

        return RidgeDetectionResult(
            fibers=fibers,
            pixel_size=p.pixel_size,
        )

    def supports_3d(self) -> bool:
        return False