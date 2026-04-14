# -*- coding: utf-8 -*-
"""
CT-FIRE fiber extraction method.

CT-FIRE (Curvelet Transform Fiber Extraction) is a two-stage algorithm:

  **Stage 1 — CT (Curvelet Transform)**
      A multi-scale curvelet transform enhances fiber-like curvilinear
      structures while suppressing noise and background.  Angular selectivity
      of curvelets makes them particularly well-suited to anisotropic
      structures like collagen fibers in SHG images.

      Backend priority: curvelops FDCT → MATLAB Engine → NumPy approximation.
      See :mod:`tme_quant.fiber_analysis.utils.curvelet_utils`.

  **Stage 2 — FIRE (Fiber Extraction)**
      A graph-based tracing algorithm that works on the **fiber mask**
      (not the skeleton).  It uses the Euclidean distance transform so
      that each foreground pixel carries the local fiber radius, enabling
      width-aware tracing:

        1. Compute the distance transform of the fiber mask.
        2. Trace fibers along distance-transform ridges (medial axis).
        3. At each centerline point, local width = distance_transform × 2.
        4. Filters by length, width, and straightness.

      The critical difference from skeleton-based tracing: fiber width is
      integral to tracing, not a post-hoc estimate.  This correctly separates
      touching fibers of different thickness.

      Backend priority: CT-FIRE C++ extension → pure-Python approximation.
      See :mod:`tme_quant.fiber_analysis.utils.ctfire_utils`.

3-D analysis semantics
----------------------
``extract_3d`` performs **true volumetric** analysis on a (Z, H, W) image.
It is **not** slice-by-slice 2-D processing of independent planes.

* The curvelet transform is applied jointly across the full volume via
  ``curvelet_transform_3d`` (curvelops FDCT3D when available).
* The binary fiber mask is thresholded in 3-D.
* Skeletonization is done on the full 3-D mask.
* The FIRE tracer operates on the 3-D fiber mask via the 3-D distance transform,
  centerlines that can span multiple z-planes.

Because the C++ 3-D FIRE extension is not yet compiled, ``extract_3d``
currently raises ``NotImplementedError`` at the FIRE stage.  The curvelet
and mask steps are fully implemented and will be exercised when C++ FIRE
becomes available.  The ``supports_3d()`` method returns ``False`` until
the C++ extension is present to make this limitation explicit to callers.

References
----------
Bredfeldt et al. (2014) Computational segmentation of collagen fibers from
second-harmonic generation images of breast cancer.
J Biomed Opt 19(1):016007.

CT-FIRE C++ source (partially implemented, to be wrapped):
https://github.com/uw-loci/curvelets/tree/master/src/CurveAlign_CT-FIRE/ctFIRE/CPP
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from ..extraction import BaseExtractionMethod
from ..config import (
    ExtractionParams, CTFireParams, CTFireResult, FiberProperties,
)
from ..utils.curvelet_utils import curvelet_transform_2d, curvelet_transform_3d
from ..utils.ctfire_utils import fire_2d, fire_3d, ctfire_backend_status
from ..utils.geometry_utils import compute_fiber_properties


class CTFireExtraction(BaseExtractionMethod):
    """
    CT-FIRE fiber extraction: Curvelet Transform + FIRE graph tracing.

    This class implements the full CT-FIRE pipeline:

      CT step  → curvelet energy map → thresholded fiber mask
      FIRE step → distance transform of mask → trace ridges → ordered centerlines

    For 2-D images, both steps are fully available (with C++ FIRE preferred
    and a Python approximation as fallback).

    For 3-D volumetric images, ``extract_3d`` applies the curvelet transform
    and masking on the full volume, then calls the 3-D FIRE tracer.  Because
    the C++ 3-D FIRE extension is pending, ``extract_3d`` raises
    ``NotImplementedError`` until that extension is compiled.  This is an
    intentional hard boundary — we do not silently fall back to slice-by-slice
    2-D processing, which would not be volumetric analysis.

    Use ``supports_3d()`` to check availability at runtime.

    Parameters
    ----------
    None — all configuration is passed through :class:`CTFireParams`.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> CTFireResult:
        """
        Extract individual fibers from a **2-D** image using CT-FIRE.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.  Typically a single SHG plane or a 2-D
            fluorescence image.
        params : CTFireParams
            Algorithm parameters.  A plain :class:`ExtractionParams` is
            coerced to ``CTFireParams`` with default CT-FIRE values.

        Returns
        -------
        CTFireResult
            Contains the detected fiber list, binary mask, labeled image,
            curvelet energy map, and diagnostic counts.
        """
        if image.ndim != 2:
            raise ValueError(
                f"extract_2d requires a 2-D image (H, W), got shape {image.shape}. "
                "For volumetric data use extract_3d."
            )

        p = self._coerce_params_2d(params)

        # ── Stage 1: Curvelet Transform (CT) ─────────────────────────────────
        #   Normalise to float32 [0, 1] and apply CLAHE to boost local contrast
        #   for faint fibers before the curvelet decomposition.
        img_norm = image.astype(np.float32)
        if img_norm.max() > 0:
            img_norm /= img_norm.max()
        from skimage.exposure import equalize_adapthist
        img_proc = equalize_adapthist(img_norm, clip_limit=0.03).astype(np.float32)

        coeffs = curvelet_transform_2d(
            img_proc,
            n_levels=p.ctfire_n_levels,
            n_angles=p.ctfire_n_angles,
            use_matlab=p.use_matlab_backend,
        )

        # Sum absolute curvelet coefficients to get total ridge response per pixel.
        # Using |coeffs| (not coeffs²) keeps the threshold linear in the Frangi
        # response so that ctfire_threshold maps directly to the normalised
        # Frangi ridge strength  (0 = no ridge, 1 = peak ridge).
        energy_map    = np.sum(np.abs(coeffs), axis=-1).astype(np.float32)
        e_max         = float(energy_map.max())
        reconstructed = energy_map / (e_max + 1e-10)

        # Binary fiber mask: Frangi ridge response above threshold.
        fiber_mask = reconstructed > p.ctfire_threshold

        # Additionally include the brightest pixels in the raw image.
        # Frangi undershoots on very thick/saturated collagen bundles because
        # CLAHE flattens their gradient, making them look like blobs rather
        # than ridges.  In SHG images background is near-zero, so the top
        # ~8 % of intensity reliably corresponds to fiber signal.
        bright_thresh = float(np.percentile(img_norm, 92))
        fiber_mask = fiber_mask | (img_norm > bright_thresh)

        # Optional morphological closing: bridges small gaps between
        # near-touching fiber segments, reducing fragmented detections.
        if p.mask_closing_radius > 0:
            from scipy.ndimage import binary_closing
            from skimage.morphology import disk
            fiber_mask = binary_closing(
                fiber_mask,
                structure=disk(p.mask_closing_radius),
            )

        # ── Stage 2: FIRE (Fiber Extraction) ─────────────────────────────────
        #   FIRE operates on the fiber mask directly, not the skeleton.
        #   Internally it computes the distance transform to get per-pixel
        #   radius information, then traces fibers along the distance-
        #   transform ridge — the key advantage over skeleton tracing.
        traces = fire_2d(
            fiber_mask        = fiber_mask,
            image             = image.astype(np.float64),
            pixel_size        = p.pixel_size,
            min_fiber_length  = p.min_fiber_length,
            max_fiber_length  = p.max_fiber_length,
            straightness_threshold = p.straightness_threshold,
            min_fiber_width   = p.min_fiber_width,
            max_fiber_width   = p.max_fiber_width,
            spur_length_px    = p.spur_length_px,
        )
        n_candidates = len(traces)

        # ── Build FiberProperties from FIRE traces ────────────────────────────
        # Each trace is (N, 3) float32: (row, col, radius_px).
        # Width = mean(radius_px) × 2 × pixel_size (direct from distance transform).
        # Length, straightness, angle, curvature from centerline geometry.
        fibers: List[FiberProperties] = []
        for fiber_id, trace in enumerate(traces):
            if trace is None or len(trace) < 2:
                continue

            centerline = trace[:, :2]   # (N, 2) row, col
            radii_px   = trace[:, 2]    # (N,)  local half-width in pixels

            # Geometric properties from the ordered centerline
            props = compute_fiber_properties(
                centerline, image, p.pixel_size, p.fiber_width_range
            )

            # Override width with the FIRE distance-transform measurement,
            # which is more accurate than the intensity-profile FWHM for
            # irregular or faint fibers.
            width_um = float(radii_px.mean()) * 2.0 * p.pixel_size
            width_um = float(np.clip(width_um, *p.fiber_width_range))

            fibers.append(FiberProperties(
                fiber_id     = fiber_id,
                length       = props['length'],
                width        = width_um,
                straightness = props['straightness'],
                angle        = props['angle'],
                curvature    = props['curvature'] if p.measure_curvature else 0.0,
                centerline   = centerline if p.extract_centerlines else None,
                aspect_ratio = (width_um and props['length'] / width_um) or None,
            ))

        labeled = (
            self._build_labeled_image(fibers, image.shape)
            if fibers else np.zeros(image.shape, dtype=np.int32)
        )

        return CTFireResult(
            fibers              = fibers,
            pixel_size          = p.pixel_size,
            fiber_mask          = fiber_mask,
            labeled_fibers      = labeled,
            curvelet_energy_map = energy_map,
            n_candidates        = n_candidates,
        )

    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> CTFireResult:
        """
        Extract individual fibers from a **3-D volume** using CT-FIRE.

        This is a **true volumetric** method — the curvelet transform and
        FIRE tracing are applied to the full (Z, H, W) volume as a single
        entity.  It does NOT process slices independently.

        Current implementation status
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        * **CT stage**: Fully implemented.  ``curvelet_transform_3d``
          dispatches to curvelops FDCT3D (genuine volumetric) when
          available, or warns and uses slice-by-slice as a degraded fallback.
        * **FIRE stage**: Requires the CT-FIRE C++ extension
          (``_ctfire_cpp``), which has not yet been compiled.
          Until it is available, this method raises ``NotImplementedError``
          so that callers are not misled by approximate results.

        Use ``CTFireExtraction.supports_3d()`` to check at runtime whether
        3-D extraction is currently available.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
            3-D grayscale volume.
        params : CTFireParams

        Raises
        ------
        ValueError
            If *image* is not 3-D.
        NotImplementedError
            Until the C++ 3-D FIRE extension is compiled and available.
        """
        if image.ndim != 3:
            raise ValueError(
                f"extract_3d requires a 3-D volume (Z, H, W), got shape {image.shape}."
            )

        p = self._coerce_params_3d(params)

        # ── Stage 1: Volumetric Curvelet Transform (CT) ───────────────────────
        #   This step is fully implemented — curvelops FDCT3D when available,
        #   with a UserWarning if falling back to slice-by-slice processing.
        coeffs    = curvelet_transform_3d(
            image,
            n_levels   = p.ctfire_n_levels,
            n_angles   = p.ctfire_n_angles,
            use_matlab = p.use_matlab_backend,
        )

        energy_map    = np.sum(coeffs ** 2, axis=-1).astype(np.float32)
        e_max         = float(energy_map.max())
        reconstructed = energy_map / (e_max + 1e-10)
        fiber_mask    = reconstructed > p.ctfire_threshold

        # ── Stage 2: Volumetric FIRE tracing ─────────────────────────────────
        #   FIRE operates on the 3-D fiber mask directly, using the 3-D
        #   distance transform for width-aware tracing across z-planes.
        #   fire_3d raises NotImplementedError if C++ is unavailable.
        traces = fire_3d(
            fiber_mask       = fiber_mask,
            image            = image.astype(np.float64),
            pixel_size       = p.pixel_size,
            z_spacing        = p.z_spacing,
            min_fiber_length = p.min_fiber_length,
            max_fiber_length = p.max_fiber_length,
            straightness_threshold = p.straightness_threshold,
            min_fiber_width  = p.min_fiber_width,
            max_fiber_width  = p.max_fiber_width,
        )

        # If we reach here, C++ is available — build result from 3-D traces.
        # Each trace is (N, 4) float32: (z, row, col, radius_px).
        fibers: List[FiberProperties] = []
        for fiber_id, trace in enumerate(traces):
            if trace is None or len(trace) < 2:
                continue
            z_coords  = trace[:, 0]
            centerline_xy = trace[:, 1:3]   # (N, 2) row, col
            radii_px  = trace[:, 3]

            # Use midpoint z-slice for intensity profile width check
            z_mid = int(round(float(z_coords[len(z_coords) // 2])))
            z_mid = int(np.clip(z_mid, 0, image.shape[0] - 1))
            props = compute_fiber_properties(
                centerline_xy, image[z_mid], p.pixel_size, p.fiber_width_range
            )

            width_um = float(radii_px.mean()) * 2.0 * p.pixel_size
            width_um = float(np.clip(width_um, *p.fiber_width_range))

            # 3-D centerline as (z, row, col) — store full 3-D trace
            centerline_3d = trace[:, :3] if p.extract_centerlines else None

            fibers.append(FiberProperties(
                fiber_id     = fiber_id,
                length       = props['length'],
                width        = width_um,
                straightness = props['straightness'],
                angle        = props['angle'],
                curvature    = props['curvature'] if p.measure_curvature else 0.0,
                centerline   = centerline_3d,
                aspect_ratio = (width_um and props['length'] / width_um) or None,
            ))

        return CTFireResult(
            fibers              = fibers,
            pixel_size          = p.pixel_size,
            fiber_mask          = fiber_mask,
            curvelet_energy_map = energy_map,
            n_candidates        = len(traces),
        )

    def supports_3d(self) -> bool:
        """
        Return True only if volumetric 3-D CT-FIRE extraction is available.

        3-D CT-FIRE requires the C++ FIRE extension (``_ctfire_cpp``).
        When the extension is absent, ``extract_3d`` raises
        ``NotImplementedError`` rather than silently running slice-by-slice.

        Check status details via::

            from tme_quant.fiber_analysis.utils.ctfire_utils import ctfire_backend_status
            print(ctfire_backend_status())
        """
        return ctfire_backend_status()['3d_supported']

    @staticmethod
    def _coerce_params_2d(params: ExtractionParams) -> CTFireParams:
        """Ensure we have a CTFireParams; coerce from base class if needed."""
        if isinstance(params, CTFireParams):
            return params
        return CTFireParams(
            mode                 = params.mode,
            pixel_size           = params.pixel_size,
            min_fiber_length     = params.min_fiber_length,
            max_fiber_length     = params.max_fiber_length,
            min_fiber_width      = params.min_fiber_width,
            max_fiber_width      = params.max_fiber_width,
            measure_length       = params.measure_length,
            measure_width        = params.measure_width,
            measure_straightness = params.measure_straightness,
            measure_angle        = params.measure_angle,
            measure_curvature    = params.measure_curvature,
            extract_centerlines  = params.extract_centerlines,
        )

    @staticmethod
    def _coerce_params_3d(params: ExtractionParams) -> CTFireParams:
        """Ensure we have a CTFireParams for 3-D analysis."""
        if isinstance(params, CTFireParams):
            return params
        return CTFireParams(mode=params.mode, pixel_size=params.pixel_size)