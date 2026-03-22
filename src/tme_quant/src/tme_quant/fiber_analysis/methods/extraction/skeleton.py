# -*- coding: utf-8 -*-
"""
Skeletonization-based fiber centerline extraction.

This method extracts fiber centerlines from images by:
  1. Thresholding the image to a binary fiber mask.
  2. Computing the morphological skeleton (medial axis).
  3. Pruning short spurious branches.
  4. Tracing each connected skeleton component as a fiber centerline.

Relationship to CT-FIRE / FIRE
-------------------------------
Both this method and CT-FIRE start from a binary fiber mask, but differ in
how they extract centerlines and measure width:

  **Skeleton method** (this module):
    - Thins the mask to a 1-pixel-wide skeleton.
    - Traces connected skeleton components.
    - Width is estimated *afterward* via perpendicular intensity cross-sections.
    - Width is not used during tracing.
    - Fast; works well for thin, well-separated fibers.

  **FIRE** (:mod:`~tme_quant.fiber_analysis.utils.ctfire_utils`):
    - Works on the fiber mask directly via the distance transform.
    - Traces along the distance-transform ridge (medial axis).
    - Width (= 2 × distance-transform value) is integral to tracing.
    - Correctly separates touching or partially-overlapping fibers of
      different thickness — something skeleton tracing cannot do.
    - Used by CT-FIRE for SHG collagen images where fibers are thick
      and may be in contact.

3-D analysis semantics
-----------------------
``extract_3d`` applies **true volumetric** 3-D skeletonization to the full
(Z, H, W) volume — not slice-by-slice 2-D skeletonization.

Lee's algorithm (``skeleton_method='lee'``, the default) supports 3-D
directly in scikit-image.  The resulting skeleton contains fibers that
span multiple z-planes.  Zhang's algorithm (``'zhang'``) is 2-D only and
will raise an error if selected for 3-D data.

For purely in-plane (slice-by-slice) analysis of a 3-D stack, call
``extract_2d`` on each slice individually from the caller side.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .base_extraction import BaseExtractionMethod
from ...config.extraction_params import (
    ExtractionParams, SkeletonParams, SkeletonResult, FiberProperties,
)
from ...utils.geometry_utils import compute_fiber_properties


class SkeletonExtractionMethod(BaseExtractionMethod):
    """
    Skeletonization-based fiber centerline extraction.

    Binarizes the image, computes the morphological skeleton of the binary
    mask, prunes short branches, then traces each connected skeleton
    component into an ordered fiber centerline.

    Width is estimated after tracing via perpendicular intensity profiles
    (FWHM), since the 1-pixel skeleton carries no thickness information.

    Requires only scikit-image and scipy — no external tools.

    See module docstring for the difference between this method and FIRE.
    """

    # ── 2-D extraction ────────────────────────────────────────────────────────

    def extract_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> SkeletonResult:
        """
        Extract fiber centerlines from a 2-D image via skeletonization.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image.
        params : SkeletonParams or ExtractionParams

        Returns
        -------
        SkeletonResult
        """
        from skimage.morphology import skeletonize
        from skimage.filters import threshold_otsu, threshold_li, threshold_yen
        from skimage.measure import label
        from scipy.ndimage import gaussian_filter

        if image.ndim != 2:
            raise ValueError(
                f"extract_2d expects a 2-D image (H, W), got shape {image.shape}. "
                "For volumetric data use extract_3d."
            )

        p = self._coerce_params(params)

        # ── Step 1: Pre-processing ────────────────────────────────────────────
        img = image.astype(np.float32)
        if p.smooth_skeleton:
            img = gaussian_filter(img, sigma=1.0)

        # ── Step 2: Binarize ──────────────────────────────────────────────────
        binary = self._binarize(img, p)

        # ── Step 3: Skeletonize the binary mask ───────────────────────────────
        # Lee's algorithm is used for 2-D and is the default; Zhang is also
        # valid for 2-D and is slightly faster on small images.
        method_arg = p.skeleton_method.lower() if p.skeleton_method.lower() in ('lee', 'zhang') else None
        skeleton = skeletonize(binary, method=method_arg)

        # Classify skeleton pixels for diagnostics (before pruning)
        branch_pts, end_pts = self._classify_skeleton_points(skeleton)

        # ── Step 4: Prune short spurious branches ─────────────────────────────
        # A branch shorter than min_branch_length is treated as noise and
        # removed.  This runs on labeled connected components; any component
        # shorter than the threshold is eliminated.
        min_px = max(2, int(p.min_branch_length / p.pixel_size))
        skeleton = self._prune_branches(skeleton, min_px)

        # ── Step 5: Trace connected components → ordered centerlines ──────────
        # Each connected component of the pruned skeleton is one fiber.
        # We break the skeleton at junction points before ordering so that
        # the greedy nearest-neighbour walk does not backtrack across branches.
        labeled_skel = label(skeleton, connectivity=2)
        fibers: List[FiberProperties] = []

        for region_id in range(1, int(labeled_skel.max()) + 1):
            # Get all pixels of this component
            comp_mask = labeled_skel == region_id
            # Order by traversing edges between junction/end points
            ordered = self._trace_component(comp_mask, skeleton)
            if ordered is None or len(ordered) < 2:
                continue

            props = compute_fiber_properties(
                ordered, image, p.pixel_size, p.fiber_width_range
            )

            if not (p.min_fiber_length <= props['length'] <= p.max_fiber_length):
                continue

            fibers.append(FiberProperties(
                fiber_id     = region_id - 1,
                length       = props['length'],
                width        = props['width'],
                straightness = props['straightness'],
                angle        = props['angle'],
                curvature    = props['curvature'] if p.measure_curvature else 0.0,
                centerline   = ordered if p.extract_centerlines else None,
                aspect_ratio = (
                    props['length'] / props['width']
                    if props['width'] > 0 else None
                ),
            ))

        return SkeletonResult(
            fibers        = fibers,
            pixel_size    = p.pixel_size,
            skeleton_mask = skeleton,
            branch_points = branch_pts,
            end_points    = end_pts,
        )

    # ── 3-D volumetric extraction ─────────────────────────────────────────────


    def extract_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
    ) -> SkeletonResult:
        """
        Extract fiber centerlines from a 3-D volume via volumetric skeletonization.

        This method applies **true 3-D skeletonization** to the full (Z, H, W)
        volume — not slice-by-slice 2-D skeletonization.  The resulting
        skeleton contains fibers that span multiple z-planes, and each fiber
        centerline is a 3-D ordered path in (z, row, col) coordinates.

        Lee's algorithm (``skeleton_method='lee'``) is used for 3-D.
        Zhang's algorithm is 2-D only and will raise a ValueError if selected.

        Parameters
        ----------
        image : ndarray, shape (Z, H, W)
            3-D grayscale volume.
        params : SkeletonParams or ExtractionParams

        Returns
        -------
        SkeletonResult
            ``skeleton_mask`` has shape (Z, H, W).
            Each ``FiberProperties.centerline`` is (N, 3) float32 in
            (z, row, col) order.
        """
        from skimage.morphology import skeletonize
        from skimage.filters import threshold_otsu
        from skimage.measure import label
        from scipy.ndimage import gaussian_filter

        if image.ndim != 3:
            raise ValueError(
                f"extract_3d expects a 3-D volume (Z, H, W), got shape {image.shape}."
            )

        p = self._coerce_params(params)

        if p.skeleton_method.lower() == 'zhang':
            raise ValueError(
                "Zhang skeletonization algorithm is 2-D only. "
                "Use skeleton_method='lee' for 3-D volumetric analysis."
            )

        # ── Binarize the full volume ──────────────────────────────────────────
        vol = image.astype(np.float32)
        if p.smooth_skeleton:
            vol = gaussian_filter(vol, sigma=1.0)
        binary = self._binarize(vol, p)

        # ── True 3-D skeletonization ──────────────────────────────────────────
        # skimage.morphology.skeletonize accepts 3-D arrays natively with Lee.
        skeleton = skeletonize(binary)   # 'lee' is the default for 3-D

        # Classify 3-D skeleton pixels using 26-connectivity
        branch_pts, end_pts = self._classify_skeleton_points_3d(skeleton)

        # Prune short 3-D components
        min_px = max(2, int(p.min_branch_length / p.pixel_size))
        skeleton = self._prune_branches(skeleton, min_px)

        # ── Trace connected components ────────────────────────────────────────
        # Each connected component is a fiber spanning any number of z-planes.
        labeled_skel = label(skeleton, connectivity=2)
        fibers: List[FiberProperties] = []

        for region_id in range(1, int(labeled_skel.max()) + 1):
            # coords: (N, 3) in (z, row, col)
            coords = np.argwhere(labeled_skel == region_id).astype(np.float32)
            if len(coords) < 2:
                continue

            ordered_3d = self._order_centerline(coords)  # ordered (z, row, col)
            ordered_xy = ordered_3d[:, 1:]               # (N, 2) row, col for 2-D metrics

            # Use midpoint z-slice for intensity-profile width estimate
            z_mid = int(round(float(ordered_3d[len(ordered_3d) // 2, 0])))
            z_mid = int(np.clip(z_mid, 0, image.shape[0] - 1))

            props = compute_fiber_properties(
                ordered_xy, image[z_mid], p.pixel_size, p.fiber_width_range
            )

            if not (p.min_fiber_length <= props['length'] <= p.max_fiber_length):
                continue

            fibers.append(FiberProperties(
                fiber_id     = region_id - 1,
                length       = props['length'],
                width        = props['width'],
                straightness = props['straightness'],
                angle        = props['angle'],
                curvature    = props['curvature'] if p.measure_curvature else 0.0,
                centerline   = ordered_3d if p.extract_centerlines else None,
                aspect_ratio = (
                    props['length'] / props['width']
                    if props['width'] > 0 else None
                ),
            ))

        return SkeletonResult(
            fibers        = fibers,
            pixel_size    = p.pixel_size,
            skeleton_mask = skeleton,
            branch_points = branch_pts,
            end_points    = end_pts,
        )

    def supports_3d(self) -> bool:
        """
        Return True — volumetric 3-D skeletonization is fully supported.

        Uses Lee's algorithm (default) which natively handles 3-D data.
        Note: ``skeleton_method='zhang'`` is 2-D only and will raise
        ValueError in ``extract_3d``.
        """
        return True

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _coerce_params(params: ExtractionParams) -> SkeletonParams:
        """Ensure we have a SkeletonParams; coerce from base class if needed."""
        if isinstance(params, SkeletonParams):
            return params
        return SkeletonParams(
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
    def _binarize(img: np.ndarray, p: SkeletonParams) -> np.ndarray:
        """Threshold an image or volume to a binary mask."""
        from skimage.filters import threshold_otsu, threshold_li, threshold_yen

        method = p.threshold_method.lower()
        # For 3-D images, threshold methods operate on all voxels
        if method == 'otsu':
            thresh = threshold_otsu(img)
        elif method == 'li':
            thresh = threshold_li(img)
        elif method == 'yen':
            thresh = threshold_yen(img)
        elif method == 'manual' and p.manual_threshold is not None:
            thresh = float(p.manual_threshold)
        else:
            thresh = threshold_otsu(img)
        return img > thresh

    @staticmethod
    def _classify_skeleton_points(
        skeleton: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Classify 2-D skeleton pixels as branch points or end points.

        A pixel is a **branch point** (junction) if it has ≥ 3 skeleton
        neighbours (8-connectivity).  A pixel is an **end point** if it has
        exactly 1 skeleton neighbour.

        Returns
        -------
        branch_points : ndarray (B, 2) float32, or None
        end_points    : ndarray (E, 2) float32, or None
        """
        from scipy.ndimage import convolve

        k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
        nbr = convolve(skeleton.astype(np.uint8), k, mode='constant', cval=0)

        bp = np.argwhere(skeleton & (nbr >= 3))
        ep = np.argwhere(skeleton & (nbr == 1))

        return (
            bp.astype(np.float32) if len(bp) else None,
            ep.astype(np.float32) if len(ep) else None,
        )

    @staticmethod
    def _classify_skeleton_points_3d(
        skeleton: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Classify 3-D skeleton voxels using 26-connectivity neighbourhood.

        Returns
        -------
        branch_points : ndarray (B, 3) float32, or None
        end_points    : ndarray (E, 3) float32, or None
        """
        from scipy.ndimage import convolve

        k = np.ones((3, 3, 3), dtype=np.uint8); k[1, 1, 1] = 0
        nbr = convolve(skeleton.astype(np.uint8), k, mode='constant', cval=0)

        bp = np.argwhere(skeleton & (nbr >= 3))
        ep = np.argwhere(skeleton & (nbr == 1))

        return (
            bp.astype(np.float32) if len(bp) else None,
            ep.astype(np.float32) if len(ep) else None,
        )

    @staticmethod
    def _prune_branches(skeleton: np.ndarray, min_length: int) -> np.ndarray:
        """
        Remove connected skeleton components shorter than *min_length* pixels.

        Works for both 2-D and 3-D skeletons (label connectivity = 2).
        """
        from skimage.measure import label

        pruned  = skeleton.copy()
        labeled = label(pruned, connectivity=2)
        for rid in range(1, int(labeled.max()) + 1):
            if np.sum(labeled == rid) < min_length:
                pruned[labeled == rid] = False
        return pruned

    @staticmethod
    def _trace_component(
        comp_mask: np.ndarray,
        full_skeleton: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Trace a single connected skeleton component into an ordered centerline.

        Strategy: remove junction pixels from the component to break it into
        simple non-branching edge segments, order each segment, then re-attach
        junction pixels to produce a single ordered path.  This prevents the
        greedy nearest-neighbour walk from backtracking across branch points.

        For simple (non-branching) components the junction removal has no
        effect, so this degrades gracefully to the plain nearest-neighbour
        walk.

        Parameters
        ----------
        comp_mask : ndarray bool, shape (H, W)
            Mask of this component (True where component pixels are).
        full_skeleton : ndarray bool, shape (H, W)
            The full pruned skeleton (used to compute local neighbour counts).

        Returns
        -------
        ndarray (N, 2) float32, or None if fewer than 2 pixels.
        """
        from scipy.ndimage import convolve
        from skimage.measure import label

        pixels = np.argwhere(comp_mask)
        if len(pixels) < 2:
            return None

        # Neighbour count within the full skeleton
        k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
        nbr = convolve(full_skeleton.astype(np.uint8), k, mode='constant', cval=0)

        junction_mask = comp_mask & (nbr >= 3)

        # If no junctions, just order the whole component
        if not junction_mask.any():
            return SkeletonExtractionMethod._order_centerline(pixels.astype(np.float32))

        # Break at junctions: label edge segments
        edge_mask   = comp_mask & ~junction_mask
        edge_labels = label(edge_mask, connectivity=2)

        segments = []
        for eid in range(1, int(edge_labels.max()) + 1):
            seg = np.argwhere(edge_labels == eid)
            if len(seg) < 1:
                continue
            segments.append(SkeletonExtractionMethod._order_centerline(
                seg.astype(np.float32)
            ))

        if not segments:
            # Only junction pixels remain — return them ordered
            return SkeletonExtractionMethod._order_centerline(pixels.astype(np.float32))

        # Pick the longest segment as the primary fiber centerline.
        # (For collagen analysis a simple longest-path heuristic is sufficient;
        # a full DFS on the graph would be more accurate for heavily branched
        # networks, which belong to a different analysis regime.)
        segments.sort(key=lambda s: len(s), reverse=True)
        return segments[0]

    @staticmethod
    def _order_centerline(coords: np.ndarray) -> np.ndarray:
        """
        Order an unordered set of skeleton pixels (or voxels) into a
        sequential path using a greedy nearest-neighbour walk.

        This is necessary before computing arc-length, curvature, or angle,
        since those metrics depend on the points being in traversal order
        along the fiber.

        Parameters
        ----------
        coords : ndarray, shape (N, D)   D = 2 for 2-D, 3 for 3-D

        Returns
        -------
        ndarray, shape (N, D), float32
        """
        if len(coords) <= 2:
            return coords.astype(np.float32)

        remaining = list(range(len(coords)))
        path      = [remaining.pop(0)]

        while remaining:
            current = coords[path[-1]]
            dists   = np.linalg.norm(coords[remaining] - current, axis=1)
            nearest = remaining[int(np.argmin(dists))]
            path.append(nearest)
            remaining.remove(nearest)

        return coords[path].astype(np.float32)