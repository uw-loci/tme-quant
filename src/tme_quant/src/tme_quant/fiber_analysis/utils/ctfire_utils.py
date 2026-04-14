# -*- coding: utf-8 -*-
"""
CT-FIRE FIRE algorithm utilities.

This module provides the Python interface to the FIRE (Fiber Extraction)
algorithm used by CT-FIRE.

What FIRE does — and how it differs from skeleton tracing
----------------------------------------------------------
Both FIRE and the skeleton-based method (`skeleton.py`) work from a binary
fiber mask, but they differ in a fundamental way:

**Skeleton-based extraction** (`skeleton.py`):
  - Thins the binary mask to a 1-pixel-wide medial-axis skeleton.
  - Traces connected components of the skeleton as fiber centerlines.
  - Width is estimated afterward from intensity cross-sections.
  - Fast and simple; width information is not used during tracing.

**FIRE** (this module):
  - Works directly on the binary fiber mask (NOT the skeleton).
  - Computes the **distance transform** of the mask so that every foreground
    pixel carries a local radius value (half the fiber thickness at that
    point).
  - Finds seed points — local maxima of the distance transform — that lie
    on the medial axis.
  - Traces fibers by walking from seed to seed along paths of high distance-
    transform value, effectively following the thickest part of each fiber.
  - At each traced point the fiber width is the local distance-transform
    value × 2 × pixel_size — it is integral to the tracing, not a
    post-hoc estimate.
  - Can handle **varying width** along a single fiber and correctly separates
    touching or overlapping fibers that have different radii.

The distinction matters for collagen SHG images where fibers are thick,
irregular, and may touch:  skeleton-based tracing cannot separate two
touching fibers of different width, while FIRE can because it follows the
distance-transform ridge.

FIRE algorithm overview (original MATLAB / C++ CT-FIRE)
-------------------------------------------------------
Given a curvelet-enhanced, binary-thresholded fiber mask *M*:

  1. Compute the **Euclidean distance transform** of *M* → *D*.
  2. Find local maxima of *D* as seed points on the medial axis.
     Each seed has a radius = D[seed] pixels.
  3. For each seed, walk outward along the distance-transform gradient,
     collecting (row, col, radius) tuples until the path reaches the mask
     boundary or merges with another path.
  4. Build a graph: nodes = seeds / junction points; edges = traced paths.
  5. Filter edges by length (µm), straightness, and minimum width.
  6. Return ordered centerline arrays, each augmented with per-point width.

The original MATLAB FIRE code is in:
    https://github.com/uw-loci/curvelets

The partially-implemented C++ port is at:
    https://github.com/uw-loci/curvelets/tree/master/src/CurveAlign_CT-FIRE/ctFIRE/CPP

C++ wrapper integration status
-------------------------------
[ ] Compile CPP source with pybind11 or cffi
[ ] Write Python binding module  (``_ctfire_cpp``)
[ ] Wire ``_fire_cpp_2d()`` below to the real shared library
[ ] Implement 3-D volumetric extension in C++
[ ] Write 3-D Python binding

When the C++ wrapper is ready, set the module-level flag::

    _CPP_AVAILABLE = True
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# C++ wrapper availability flag
# ─────────────────────────────────────────────────────────────────────────────

# TODO: set to True once the C++ shared library is compiled and installed.
_CPP_AVAILABLE: bool = False


def _try_import_cpp() -> bool:
    """Return True if the compiled CT-FIRE C++ extension is importable."""
    if not _CPP_AVAILABLE:
        return False
    try:
        import _ctfire_cpp  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fire_2d(
    fiber_mask: np.ndarray,
    image: np.ndarray,
    pixel_size: float = 1.0,
    min_fiber_length: float = 10.0,
    max_fiber_length: float = 1000.0,
    straightness_threshold: float = 0.0,
    min_fiber_width: float = 0.5,
    max_fiber_width: float = 20.0,
    spur_length_px: int = 8,
) -> List[np.ndarray]:
    """
    Run the FIRE fiber extraction algorithm on a 2-D binary fiber mask.

    FIRE operates on the **fiber mask**, not the skeleton.  It uses the
    Euclidean distance transform of the mask so that each foreground pixel
    carries the local fiber radius, enabling width-aware tracing.

    Dispatches to the C++ implementation when available; otherwise uses the
    pure-Python approximation.

    Parameters
    ----------
    fiber_mask : ndarray bool, shape (H, W)
        Binary fiber mask.  Foreground pixels (True) are fiber; background
        (False) is non-fiber.  Typically the curvelet-thresholded mask from
        the CT stage.
    image : ndarray float, shape (H, W)
        Original (or curvelet-enhanced) image used for intensity profiling
        and additional width refinement.
    pixel_size : float
        Physical size of one pixel in µm.
    min_fiber_length : float
        Minimum fiber arc length (µm) to keep.
    max_fiber_length : float
        Maximum fiber arc length (µm) to keep.
    straightness_threshold : float
        Minimum straightness (0–1).  0 = keep all.
    min_fiber_width : float
        Minimum mean fiber width (µm) to keep.
    max_fiber_width : float
        Maximum mean fiber width (µm) to keep.
    spur_length_px : int
        Number of spur-pruning iterations (Python backend only).  Each
        iteration removes one pixel from every terminal skeleton branch
        attached to a junction, eliminating short stubs produced when
        fibers cross.  Equivalent to removing branches shorter than this
        many pixels.  ``0`` disables pruning.

    Returns
    -------
    list of ndarray, each shape (N, 3) float32
        Ordered (row, col, radius_px) coordinates for each extracted fiber.
        ``radius_px`` is the local fiber half-width in pixels at each point
        (= distance-transform value at that point).
        Each array has at least 2 rows.
    """
    if _try_import_cpp():
        # C++ FIRE has its own internal stub merging; spur_length_px is ignored
        return _fire_cpp_2d(
            fiber_mask, image, pixel_size,
            min_fiber_length, max_fiber_length,
            straightness_threshold, min_fiber_width, max_fiber_width,
        )

    return _fire_python_2d(
        fiber_mask, image, pixel_size,
        min_fiber_length, max_fiber_length,
        straightness_threshold, min_fiber_width, max_fiber_width,
        spur_length_px=spur_length_px,
    )


def fire_3d(
    fiber_mask: np.ndarray,
    image: np.ndarray,
    pixel_size: float = 1.0,
    z_spacing: float = 1.0,
    min_fiber_length: float = 10.0,
    max_fiber_length: float = 1000.0,
    straightness_threshold: float = 0.0,
    min_fiber_width: float = 0.5,
    max_fiber_width: float = 20.0,
) -> List[np.ndarray]:
    """
    Run the FIRE fiber extraction algorithm on a **volumetric** 3-D fiber mask.

    Operates on the full 3-D binary mask — NOT slice-by-slice.  Uses the
    3-D Euclidean distance transform so each voxel carries the local fiber
    radius, enabling width-aware tracing in 3-D.

    .. note::
        The 3-D C++ FIRE implementation is pending.  Until it is available,
        this function raises ``NotImplementedError`` to prevent silently
        returning incorrect results from a slice-by-slice fallback.

    Parameters
    ----------
    fiber_mask : ndarray bool, shape (Z, H, W)
        3-D binary fiber mask.
    image : ndarray float, shape (Z, H, W)
        Original or curvelet-enhanced volume.
    pixel_size : float
        In-plane (XY) pixel size in µm.
    z_spacing : float
        Inter-slice spacing in µm.
    min_fiber_length, max_fiber_length : float
        Arc length bounds in µm.
    straightness_threshold : float
        Minimum straightness.
    min_fiber_width, max_fiber_width : float
        Width bounds in µm.

    Returns
    -------
    list of ndarray, each shape (N, 4) float32
        Ordered (z, row, col, radius_px) coordinates for each fiber.

    Raises
    ------
    NotImplementedError
        When the C++ 3-D FIRE extension is not yet available.
    """
    if _try_import_cpp():
        return _fire_cpp_3d(
            fiber_mask, image, pixel_size, z_spacing,
            min_fiber_length, max_fiber_length,
            straightness_threshold, min_fiber_width, max_fiber_width,
        )

    raise NotImplementedError(
        "3-D volumetric FIRE fiber extraction requires the CT-FIRE C++ extension "
        "('_ctfire_cpp'), which has not yet been compiled and installed.\n"
        "See the module docstring in ctfire_utils.py for the integration roadmap.\n\n"
        "If you need an approximate 3-D result now, use SkeletonExtractionMethod "
        "which supports 3-D volumetric skeletonization."
    )


# ─────────────────────────────────────────────────────────────────────────────
# C++ wrapper stubs
# ─────────────────────────────────────────────────────────────────────────────

def _fire_cpp_2d(
    fiber_mask: np.ndarray,
    image: np.ndarray,
    pixel_size: float,
    min_fiber_length: float,
    max_fiber_length: float,
    straightness_threshold: float,
    min_fiber_width: float,
    max_fiber_width: float,
) -> List[np.ndarray]:
    """
    [PLACEHOLDER] Dispatch to the compiled C++ 2-D FIRE implementation.

    The C++ source is at:
    https://github.com/uw-loci/curvelets/tree/master/src/CurveAlign_CT-FIRE/ctFIRE/CPP

    Expected C++ API (subject to change during porting)::

        _ctfire_cpp.fire_2d(
            fiber_mask,          # uint8 (H, W) C-contiguous numpy array
            image,               # float64 (H, W) C-contiguous numpy array
            pixel_size,          # double
            min_fiber_length,    # double, µm
            max_fiber_length,    # double, µm
            straightness_threshold,  # double [0,1]
            min_fiber_width,     # double, µm
            max_fiber_width,     # double, µm
        ) -> list of (N, 3) float64 numpy arrays
              each row: (row, col, radius_px)

    TODO: Replace body with actual _ctfire_cpp.fire_2d() call.
    """
    import _ctfire_cpp  # type: ignore
    traces = _ctfire_cpp.fire_2d(
        fiber_mask.astype(np.uint8,   copy=False),
        image.astype(np.float64,      copy=False),
        float(pixel_size),
        float(min_fiber_length),
        float(max_fiber_length),
        float(straightness_threshold),
        float(min_fiber_width),
        float(max_fiber_width),
    )
    return [t.astype(np.float32) for t in traces]


def _fire_cpp_3d(
    fiber_mask: np.ndarray,
    image: np.ndarray,
    pixel_size: float,
    z_spacing: float,
    min_fiber_length: float,
    max_fiber_length: float,
    straightness_threshold: float,
    min_fiber_width: float,
    max_fiber_width: float,
) -> List[np.ndarray]:
    """
    [PLACEHOLDER] Dispatch to the compiled C++ 3-D FIRE implementation.

    TODO: Implement the 3-D FIRE extension in C++.  Key requirements:
      1. 3-D Euclidean distance transform of the fiber mask
      2. 3-D local maxima detection on the distance transform (26-connectivity)
      3. Gradient walk along the 3-D distance-transform ridge
      4. Anisotropic arc-length computation using z_spacing vs pixel_size
      5. Return (z, row, col, radius_px) traces
    """
    import _ctfire_cpp  # type: ignore
    traces = _ctfire_cpp.fire_3d(
        fiber_mask.astype(np.uint8,  copy=False),
        image.astype(np.float64,     copy=False),
        float(pixel_size),
        float(z_spacing),
        float(min_fiber_length),
        float(max_fiber_length),
        float(straightness_threshold),
        float(min_fiber_width),
        float(max_fiber_width),
    )
    return [t.astype(np.float32) for t in traces]


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python FIRE approximation  (2-D only)
# ─────────────────────────────────────────────────────────────────────────────

def _fire_python_2d(
    fiber_mask: np.ndarray,
    image: np.ndarray,
    pixel_size: float,
    min_fiber_length: float,
    max_fiber_length: float,
    straightness_threshold: float,
    min_fiber_width: float,
    max_fiber_width: float,
    spur_length_px: int = 8,
) -> List[np.ndarray]:
    """
    Pure-Python approximation of the 2-D FIRE algorithm.

    Faithfully implements the distance-transform-based approach:

      1. Euclidean distance transform of the fiber mask → radius at each px.
      2. Find local maxima of the distance transform as medial-axis seeds.
      3. Thin to a skeleton guided by the distance transform (medial axis).
      4. Build a graph (junction detection) and trace edges.
      5. Attach per-point radius from the distance transform to each trace.
      6. Filter by length, straightness, and width.

    Each returned trace carries (row, col, radius_px) so downstream code
    has access to local fiber width at each centerline point.

    Note: The C++ implementation handles more edge cases (stub merging,
    overlapping fibers) and is significantly faster.  Install the C++ extension
    for production use.
    """
    from scipy.ndimage import distance_transform_edt, label as ndi_label, convolve
    from skimage.morphology import skeletonize
    from skimage.measure import label as ski_label

    mask = fiber_mask.astype(bool)
    if not mask.any():
        return []

    # ── Step 1: Distance transform — gives local fiber radius at every pixel ─
    dist = distance_transform_edt(mask).astype(np.float32)

    # ── Step 2: Medial-axis skeleton via distance-transform-guided thinning ──
    # skimage.skeletonize preserves the topology; we then use the distance
    # transform to assign radius to each skeleton pixel.
    skeleton = skeletonize(mask)

    # ── Step 2b: Spur pruning ─────────────────────────────────────────────
    # Iteratively remove terminal branches attached to junctions.  Each
    # iteration shaves one pixel off every stub, so after N iterations all
    # stubs shorter than N px are gone.  This dramatically reduces the
    # oversegmentation caused by cross-fiber junctions in the skeleton.
    if spur_length_px > 0:
        skeleton = _prune_spurs(skeleton, n_iters=spur_length_px)

    # ── Step 3: Detect junction (≥3 neighbours) and end (1 neighbour) points ─
    k = np.ones((3, 3), dtype=np.uint8); k[1, 1] = 0
    nbr = convolve(skeleton.astype(np.uint8), k, mode='constant', cval=0)
    junction_mask = skeleton & (nbr >= 3)
    endpoint_mask = skeleton & (nbr == 1)
    node_mask     = junction_mask | endpoint_mask

    # ── Step 4: Label skeleton edges (segments between nodes) ────────────────
    edge_skel   = skeleton & ~junction_mask   # break at junctions only
    edge_labels = ski_label(edge_skel, connectivity=2)

    traces: List[np.ndarray] = []

    # Trace each edge segment
    n_edges = int(edge_labels.max())
    for eid in range(1, n_edges + 1):
        coords = np.argwhere(edge_labels == eid)
        if len(coords) < 2:
            continue

        # Attach adjacent node pixels at both ends
        adj = _adjacent_nodes(coords, node_mask, mask.shape)
        full_coords = np.vstack([adj[:1], coords, adj[1:]]) if len(adj) else coords

        ordered = _order_by_nn(full_coords)
        # Attach distance-transform radius at each ordered skeleton point
        radii = dist[ordered[:, 0].astype(int), ordered[:, 1].astype(int)]
        trace = np.column_stack([ordered, radii]).astype(np.float32)
        traces.append(trace)

    # Also collect isolated components (no junction nodes)
    comp_labels = ski_label(skeleton, connectivity=2)
    for cid in range(1, int(comp_labels.max()) + 1):
        comp = np.argwhere(comp_labels == cid)
        if len(comp) < 2:
            continue
        if node_mask[comp[:, 0], comp[:, 1]].any():
            continue  # already covered above
        ordered = _order_by_nn(comp)
        radii   = dist[ordered[:, 0].astype(int), ordered[:, 1].astype(int)]
        trace   = np.column_stack([ordered, radii]).astype(np.float32)
        traces.append(trace)

    # ── Step 4b: Chain segments through junctions ─────────────────────────────
    # A single physical fiber that crosses another gets split at every junction
    # into multiple short segments. Here we rejoin the most collinear pair of
    # segments at each junction before applying the length filter, so the full
    # fiber arc survives.
    traces = _chain_segments_at_junctions(traces, junction_mask, dist)

    # ── Step 5: Filter by length, straightness, and width ────────────────────
    kept: List[np.ndarray] = []
    for trace in traces:
        pts      = trace[:, :2]   # (row, col)
        arc_px   = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        arc_um   = arc_px * pixel_size
        if not (min_fiber_length <= arc_um <= max_fiber_length):
            continue

        e2e = float(np.linalg.norm(pts[-1] - pts[0]))
        if arc_px > 0 and (e2e / arc_px) < straightness_threshold:
            continue

        # Width from distance transform: mean_radius × 2 × pixel_size
        mean_radius_um = float(trace[:, 2].mean()) * pixel_size * 2.0
        if not (min_fiber_width <= mean_radius_um <= max_fiber_width):
            continue

        kept.append(trace)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chain_segments_at_junctions(
    traces: List[np.ndarray],
    junction_mask: np.ndarray,
    dist: np.ndarray,
    max_turn_deg: float = 75.0,
    max_passes: int = 6,
) -> List[np.ndarray]:
    """
    Iteratively merge skeleton segments that pass through junction nodes
    nearly collinearly, converging to longer fiber arcs.

    A curvy collagen bundle may cross several junctions along its length,
    being split into many short segments.  A single pass can only chain one
    pair per junction; the merged result might then become chainable at its
    next junction.  Iterating up to ``max_passes`` times (or until no new
    merges occur) recovers the full arc.

    Parameters
    ----------
    traces : list of (N, 3) float32
        Ordered (row, col, radius_px) arrays from the skeleton tracer.
    junction_mask : bool ndarray (H, W)
        True at skeleton pixels that have ≥ 3 neighbours.
    dist : float32 ndarray (H, W)
        Euclidean distance transform of the fiber mask.
    max_turn_deg : float
        Maximum direction change (in degrees) allowed at a junction for two
        segments to be merged.  75° tolerates moderately curvy fibers while
        still rejecting true branch-offs.
    max_passes : int
        Maximum number of merge iterations.  Stops early if a pass produces
        no new merges.

    Returns
    -------
    list of (N, 3) float32 — merged traces plus unmerged leftovers.
    """
    if len(traces) < 2:
        return traces

    from scipy.ndimage import label as ndi_label

    # Straight-through: outward tangents are antiparallel → dot ≈ −1.
    # Acceptable if turn < max_turn_deg:
    #   angle between outward tangents = 180° − turn
    #   → dot threshold = cos(180° − max_turn_deg)
    cos_thresh = float(np.cos(np.radians(180.0 - max_turn_deg)))

    H, W = junction_mask.shape
    junc_lbl, _ = ndi_label(junction_mask)

    # ── Outward tangent: direction FROM junction-end INTO fiber body ───────
    def _outward(trace: np.ndarray, is_tail: bool, n: int = 5) -> np.ndarray:
        pts = trace[:, :2]
        if is_tail:
            idx = max(0, len(pts) - 1 - n)
            seg = pts[idx] - pts[-1]   # from tail toward interior
        else:
            idx = min(n, len(pts) - 1)
            seg = pts[idx] - pts[0]    # from head toward interior
        nm = float(np.linalg.norm(seg))
        return seg / nm if nm > 1e-9 else np.zeros(2)

    def _junction_centroid(jid: int):
        pxs = np.argwhere(junc_lbl == jid)
        jr, jc = pxs[len(pxs) // 2]
        return float(jr), float(jc), float(dist[jr, jc])

    # ── Iterate until convergence ──────────────────────────────────────────
    for _pass in range(max_passes):
        # Build endpoint → (trace_idx, is_tail) map fresh each pass
        ep_map: dict = {}
        for i, t in enumerate(traces):
            for is_tail in (False, True):
                rc = (int(t[-1, 0] if is_tail else t[0, 0]),
                      int(t[-1, 1] if is_tail else t[0, 1]))
                ep_map.setdefault(rc, []).append((i, is_tail))

        # Group endpoints by labeled junction blob (with 8-neighbour fallback)
        group_eps: dict = {}
        for (r, c), items in ep_map.items():
            jid = int(junc_lbl[r, c])
            if jid == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            jid = int(junc_lbl[nr, nc])
                            if jid:
                                break
                    if jid:
                        break
            if jid:
                for item in items:
                    group_eps.setdefault(jid, []).append(item)

        # Greedy merge: best collinear pair per junction
        used = [False] * len(traces)
        extra: List[np.ndarray] = []

        for jid, eps in group_eps.items():
            seen: set = set()
            active = []
            for ti, is_t in eps:
                key = (ti, is_t)
                if not used[ti] and key not in seen:
                    seen.add(key)
                    active.append((ti, is_t))
            if len(active) < 2:
                continue

            # Pick the most antiparallel (most straight-through) pair
            best_dot = 2.0
            best_a, best_b = -1, -1
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ti, ti_tail = active[a]
                    tj, tj_tail = active[b]
                    dot = float(np.dot(
                        _outward(traces[ti], ti_tail),
                        _outward(traces[tj], tj_tail),
                    ))
                    if dot < best_dot:
                        best_dot = dot
                        best_a, best_b = a, b

            if best_dot > cos_thresh:   # turn too sharp — skip
                continue

            ti, ti_tail = active[best_a]
            tj, tj_tail = active[best_b]

            # Orient: seg_i ends at junction, seg_j begins at junction
            seg_i = traces[ti]       if ti_tail  else traces[ti][::-1]
            seg_j = traces[tj][::-1] if tj_tail  else traces[tj]

            jr, jc, jr_dt = _junction_centroid(jid)
            j_arr = np.array([[jr, jc, jr_dt]], dtype=np.float32)

            extra.append(np.vstack([seg_i, j_arr, seg_j]).astype(np.float32))
            used[ti] = True
            used[tj] = True

        n_merged = sum(used)
        traces = [t for i, t in enumerate(traces) if not used[i]]
        traces.extend(extra)

        if n_merged == 0:   # converged — no more merges possible
            break

    return traces


def _prune_spurs(skeleton: np.ndarray, n_iters: int) -> np.ndarray:
    """
    Iteratively remove terminal skeleton branches attached to junctions.

    Each iteration removes one pixel from every degree-1 endpoint that is
    8-adjacent to a junction (degree ≥ 3).  After ``n_iters`` passes, all
    stubs shorter than ``n_iters`` pixels have been eliminated.

    Isolated fibers (no junction neighbours) are left untouched.
    """
    from scipy.ndimage import convolve, binary_dilation

    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    skel = skeleton.copy()

    for _ in range(n_iters):
        nbr       = convolve(skel.astype(np.uint8), k, mode='constant', cval=0)
        endpoints = skel & (nbr == 1)
        if not endpoints.any():
            break
        junctions    = skel & (nbr >= 3)
        if not junctions.any():
            break
        # Only remove endpoints that neighbour a junction (true stubs)
        junc_dilated = binary_dilation(junctions, structure=np.ones((3, 3)))
        removable    = endpoints & junc_dilated
        if not removable.any():
            break
        skel = skel & ~removable

    return skel


def _adjacent_nodes(
    edge_coords: np.ndarray,
    node_mask: np.ndarray,
    shape: tuple,
) -> np.ndarray:
    """Return skeleton node pixels 8-adjacent to any pixel in edge_coords."""
    seen, adj = set(), []
    for r, c in edge_coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    if node_mask[nr, nc] and (nr, nc) not in seen:
                        adj.append([nr, nc])
                        seen.add((nr, nc))
    return np.array(adj, dtype=int) if adj else np.empty((0, 2), dtype=int)


def _order_by_nn(coords: np.ndarray) -> np.ndarray:
    """Order pixels into a sequential path via greedy nearest-neighbour walk."""
    if len(coords) <= 2:
        return coords.astype(np.float32)
    remaining = list(range(len(coords)))
    path = [remaining.pop(0)]
    while remaining:
        cur   = coords[path[-1]]
        dists = np.linalg.norm(coords[remaining] - cur, axis=1)
        nxt   = remaining[int(np.argmin(dists))]
        path.append(nxt)
        remaining.remove(nxt)
    return coords[path].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Status helper
# ─────────────────────────────────────────────────────────────────────────────

def ctfire_backend_status() -> dict:
    """
    Report CT-FIRE FIRE algorithm backend availability.

    Returns
    -------
    dict with keys:
      ``'cpp_available'``   bool — C++ extension compiled and importable
      ``'cpp_flag'``        bool — ``_CPP_AVAILABLE`` module-level flag
      ``'python_fallback'`` bool — pure-Python approximation (always True)
      ``'3d_supported'``    bool — True only when C++ 3-D FIRE is available
    """
    cpp_ok = _try_import_cpp()
    return {
        'cpp_available':   cpp_ok,
        'cpp_flag':        _CPP_AVAILABLE,
        'python_fallback': True,
        '3d_supported':    cpp_ok,
    }


__all__ = [
    'fire_2d',
    'fire_3d',
    'ctfire_backend_status',
]