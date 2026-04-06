"""
TACS Zone Combined Analysis Example
====================================

Demonstrates ``analyze_tacs_zone`` — a workflow that runs **both** analysis
paths on the same ROI and merges the results:

  Path A  — Pixel-map (CurveAlign / OrientationJ output)
      ``compute_orientation_relative_to_roi`` iterates every valid pixel in an
      orientation image and computes its angle relative to the nearest boundary
      tangent.  Produces per-pixel records and a TACS-like distribution.

  Path B  — Per-object (CTFire / ridge detector output)
      ``compute_relative_fiber_angles`` computes three relative angles for each
      individually detected fiber object:
        • angle_to_boundary_tangent  (0° = parallel, 90° = perpendicular)
        • angle_to_roi_orientation   (alignment with the ROI's global axis)
        • angle_to_centers_line      (fiber pointing toward / across ROI centroid)

  Combined
      A pooled ``combined_mean_angle_to_tangent`` is returned by ``analyze_tacs_zone``
      that averages across both paths.

Tangent direction flavours
--------------------------
  dense_boundary=False  (default)
      Fast O(n) 2-point tangent from the nearest polygon edge.
      Sufficient for smoothly curved, densely-sampled polygon boundaries.

  dense_boundary=True
      Polynomial-fit tangent from ``compute_boundary_tangent_angle`` over 21
      8-connected neighbours.  More accurate near corners and high-curvature
      regions.  The polygon is auto-discretized once via
      ``discretize_roi_boundary`` and the dense trace is shared by both paths.

Workflow diagram
----------------

  curvealign_angles (.mat / .csv)  ──────────────────────────────────────────►
                                    compute_orientation_relative_to_roi        pixel_result
                                             │
                              (if dense_boundary=True)
                                             │
  tumor_boundary polygon ──► discretize_roi_boundary ──► dense_coords (shared)
                                             │
  ctfire_fibers (FiberObject list) ──► compute_relative_fiber_angles           fiber_results
                                             │
                              ◄──── combined_mean_angle_to_tangent ────────────►

Running this script
-------------------
  The script is self-contained and uses synthetic data so it runs without any
  real image or ROI file.  Swap the synthetic inputs with real data by following
  the comments in each section marked  ← REAL DATA.

  $ python example_analyze_tacs_zone.py

Expected output (values will differ for real data)::

    === analyze_tacs_zone example ===
    [sparse polygon, no fiber objects]
    ROI label         : synthetic_tumor
    Pixels in zone    : ...
    Mean angle/tangent: ...°  std: ...°
    TACS distribution : {...}

    [dense boundary, with fiber objects]
    ROI label         : synthetic_tumor
    Pixels in zone    : ...
    Mean angle/tangent: ...°
    Fiber objects in zone: ...
    Combined mean     : ...°
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

from tme_quant.tme_analysis.utils import (
    discretize_roi_boundary,
)
from tme_quant.tme_analysis.pipelines import (
    analyze_tacs_zone,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_roi(label: str = 'synthetic_tumor'):
    """
    Build a minimal ROIObject from a hard-coded elliptical polygon.

    Replace this with::

        from tme_quant.core.roi_manager import ROIManager
        roi_manager = ROIManager()
        roi_manager.load_from_file('my_annotations.json')
        roi = roi_manager.get_roi_by_label('tumor_boundary')

    or, for an auto-detected boundary::

        from tme_quant.tme_analysis.region_manager import RegionManager
        region_mgr = RegionManager(...)
        tumor_regions = region_mgr.detect_tumor_regions(cell_objects)
        roi_manager.from_tumor_region(tumor_regions[0], label='tumor_boundary')
        roi = roi_manager.get_roi_by_label('tumor_boundary')
    """
    from tme_quant.core.roi_manager import ROIObject, Geometry, GeometryType

    # Approximate ellipse with 36 vertices  ← REAL DATA: replace with actual polygon
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    cx, cy, rx, ry = 256, 256, 180, 120
    coords = np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])  # (x, y)

    geom = Geometry(type=GeometryType.POLYGON, coordinates=coords)
    roi = ROIObject(object_id=label, label=label, geometry=geom)
    return roi


def _make_orientation_map(height: int = 512, width: int = 512) -> np.ndarray:
    """
    Synthetic orientation map with realistic pixel sparsity:

    * **Inside the tumor ellipse** — NaN.  Collagen fibers are deposited in
      the stroma surrounding the tumor, not inside it.
    * **Outside the ellipse** — only ~40% of pixels carry a fiber orientation;
      the remaining ~60% are NaN (background, cell nuclei, blood vessels, or
      other non-fiber structures that CurveAlign / OrientationJ could not
      assign a dominant orientation to).

    ← REAL DATA: load from CurveAlign output::

        import scipy.io
        mat = scipy.io.loadmat('results_fiber.mat')
        # CurveAlign stores per-pixel dominant orientation in 'OV_all_angle'
        # (degrees, 0–180).  Pixels with no detected fiber are typically 0 or NaN.
        orientation_map = mat['OV_all_angle'].astype(np.float32)
        orientation_map[orientation_map == 0] = np.nan   # mask empty pixels

    or from OrientationJ (which also produces a per-pixel coherency map)::

        from tiff import imread
        orientation_map = imread('orientationj_angles.tif').astype(np.float32)
        coherency_map   = imread('orientationj_coherency.tif').astype(np.float32)
        # Suppress pixels where the structure tensor is too isotropic
        orientation_map[coherency_map < 0.1] = np.nan
    """
    rng = np.random.default_rng(42)
    cx, cy, rx, ry = 256, 256, 180, 120
    yy, xx = np.mgrid[:height, :width]

    # Mask for pixels inside the tumor ellipse
    inside = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1.0

    # ~40% of outside pixels carry a fiber orientation; the rest are
    # background / other structures with no reliable orientation signal.
    fiber_fraction = 0.40
    has_fiber = (~inside) & (rng.random((height, width)) < fiber_fraction)

    omap = np.full((height, width), np.nan, dtype=np.float32)
    fiber_count = int(has_fiber.sum())
    omap[has_fiber] = rng.uniform(0, 180, fiber_count).astype(np.float32)
    return omap


def _make_alignment_map(orientation_map: np.ndarray) -> np.ndarray:
    """
    Synthetic per-pixel alignment strength map in [0, 1].

    Accepted sources for the real ``alignment_map`` parameter:

    * **CurveAlign** ``CurveAlignResult.alignment_map`` — per-window angular
      energy concentration (all pixels in a window share the same value)::

        result = curvealign.analyze_2d(image, params)
        alignment_map = result.alignment_map   # (H, W), may be None

      Note: this is NOT the same as the per-region alignment score computed
      over a fiber group (``compute_region_alignment`` in ``curvealign.py``,
      which returns a scalar R per ROI/zone, not a pixel map).

    * **OrientationJ** ``OrientationJResult.coherency_map`` — per-pixel
      structure tensor coherency ``(λ_max − λ_min) / (λ_max + λ_min)``::

        from tiff import imread
        alignment_map = imread('orientationj_coherency.tif').astype(np.float32)

    Pass ``alignment_map=None`` to skip alignment recording entirely.
    """
    rng = np.random.default_rng(7)
    amap = rng.uniform(0.2, 1.0, orientation_map.shape).astype(np.float32)
    amap[np.isnan(orientation_map)] = np.nan
    return amap


class _SyntheticFiber:
    """Minimal stand-in for FiberObject."""
    def __init__(self, object_id, cx, cy, angle):
        self.object_id   = object_id
        self.angle       = float(angle)          # [0°, 180°)
        self.centerline  = np.array([[cy, cx]])  # (row, col) — single point


def _make_fiber_objects(n: int = 100, seed: int = 0):
    """
    Synthetic fiber objects in the stroma surrounding the ellipse ROI.

    Offsets range from 5 to 140 px outward from the boundary so that
    roughly half the fibers are within the 100-pixel distance threshold
    (peri-tumoral zone) and the rest are in the distal stroma beyond it.

    ← REAL DATA: load from CTFire or ridge-detector output via::

        from tme_quant.fiber_analysis.extraction import FiberExtractor
        extractor = FiberExtractor(config)
        fibers = extractor.extract(shg_image)     # list[FiberObject]
    """
    rng = np.random.default_rng(seed)
    fibers = []
    cx, cy, rx, ry = 256, 256, 180, 120
    img_w, img_h   = 512, 512          # must match _make_orientation_map
    margin         = 4                 # px — keep fibers inside image
    t_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, ti in enumerate(t_vals):
        # Boundary point on the ellipse at parameter ti
        bx = cx + rx * np.cos(ti)
        by = cy + ry * np.sin(ti)
        # Approximate outward unit normal at ti (ellipse gradient direction)
        nx, ny = np.cos(ti) / rx, np.sin(ti) / ry
        nlen   = np.hypot(nx, ny)
        nx, ny = nx / nlen, ny / nlen
        # Spread offsets across 5–140 px so the 100-px threshold splits the set
        offset = rng.uniform(5, 140)
        x = np.clip(bx + nx * offset + rng.uniform(-8, 8), margin, img_w - margin - 1)
        y = np.clip(by + ny * offset + rng.uniform(-8, 8), margin, img_h - margin - 1)
        angle = rng.uniform(0, 180)
        fibers.append(_SyntheticFiber(f'fiber_{i:03d}', x, y, angle))
    return fibers


def _make_inside_roi_items(n_fibers: int = 4, n_pixels: int = 40, seed: int = 99):
    """
    A few synthetic collagen-like fibers / orientation-map pixels placed
    *inside* the tumor ellipse.

    In real SHG data some fibres or residual collagen signal can be enclosed
    within a tumor-region ROI.  They are NOT part of the peri-tumoral stroma
    zone and should be excluded from the TACS analysis.  Excluding them is
    achieved here simply by not passing them to ``analyze_tacs_zone``; this
    function returns them separately so the visualization can render them with
    a distinct style (gray, dashed).

    Returns
    -------
    inside_pixels : list[dict]
        Each dict has ``'x'``, ``'y'``, ``'orientation'`` (degrees, [0,180)).
    inside_fibers : list[_SyntheticFiber]
    """
    rng = np.random.default_rng(seed)
    cx, cy, rx, ry = 256, 256, 180, 120

    # --- inside pixels: random positions satisfying the ellipse inequality ---
    inside_pixels = []
    attempts = 0
    while len(inside_pixels) < n_pixels and attempts < n_pixels * 20:
        attempts += 1
        x = rng.uniform(cx - rx + 10, cx + rx - 10)
        y = rng.uniform(cy - ry + 10, cy + ry - 10)
        if (x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2 < 0.85:
            inside_pixels.append({
                'x':           float(x),
                'y':           float(y),
                'orientation': float(rng.uniform(0, 180)),
            })

    # --- inside fibers: placed near the ellipse centre ---
    t_vals = np.linspace(0, 2 * np.pi, n_fibers, endpoint=False)
    inside_fibers = []
    for i, ti in enumerate(t_vals):
        r_frac = rng.uniform(0.25, 0.65)   # 25-65 % of semi-axes inward
        x = cx + rx * r_frac * np.cos(ti) + rng.uniform(-10, 10)
        y = cy + ry * r_frac * np.sin(ti) + rng.uniform(-10, 10)
        angle = rng.uniform(0, 180)
        inside_fibers.append(_SyntheticFiber(f'inside_fiber_{i:02d}', x, y, angle))

    return inside_pixels, inside_fibers


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

# TACS-like label → display colour
_TACS_COLORS = {
    'TACS-1-like': '#4477AA',   # blue  — disordered / parallel
    'TACS-2-like': '#CCBB44',   # gold  — curvilinear
    'TACS-3-like': '#EE6677',   # red   — perpendicular
}


def plot_tacs_heatmap(
    result:                     dict,
    roi,
    fiber_objects               = None,
    inside_roi_pixels           = None,
    inside_roi_fibers           = None,
    boundary_dist_threshold_px: int   = 100,
    pixel_size:                 float = 1.0,
    title:                      str   = '',
    save_path:                  str | None = None,
) -> None:
    """
    Two-panel TACS zone heatmap.

    Left panel — spatial map
        Both pixel-map points and fiber objects are split by distance to the
        ROI boundary:

        * **Within** *boundary_dist_threshold_px*:

          - *Pixels*: colored by *angle_to_boundary_tangent* (RdYlBu_r,
            0° = parallel, 90° = perpendicular).
          - *Fibers*: large circle colored by TACS-like class (blue/gold/red).

        * **Beyond** *boundary_dist_threshold_px*:

          - *Pixels*: small limegreen dot + 4-px black line showing absolute
            orientation (rendered via ``LineCollection`` for performance).
          - *Fibers*: slightly larger limegreen circle + 4-px black line.

        The ROI polygon boundary is overlaid in magenta.

        Pixel distance is taken from the ``dist_to_boundary`` field already
        stored in each point (computed in µm by ``analyze_tacs_zone`` and
        back-converted using *pixel_size*).  Fiber distance is recomputed here
        as the Euclidean distance to the nearest ROI boundary vertex.

    Right panel — TACS-like distribution
        Grouped bar chart comparing pixel counts (solid bars) and counted-fiber
        counts (hatched bars) for each TACS-like class.  Absolute counts are
        annotated above each bar.

    Parameters
    ----------
    result : dict
        Output of ``analyze_tacs_zone``.
    roi : ROIObject
        The ROI used in the analysis (for boundary overlay).
    fiber_objects : list or None
        Original fiber objects passed to ``analyze_tacs_zone``.  Both counted
        (within threshold) and uncounted (beyond threshold) objects are shown.
    inside_roi_pixels : list[dict] or None
        Pixel-map points located *inside* the ROI (not passed to the pipeline).
        Each dict needs ``'x'``, ``'y'``, ``'orientation'`` keys.  Rendered as
        semi-transparent gray dots + dashed gray orientation lines.
    inside_roi_fibers : list or None
        Fiber objects located *inside* the ROI (not passed to the pipeline).
        Rendered as gray circles + dashed gray orientation lines.
    boundary_dist_threshold_px : int
        Distance threshold in pixels.  Fibers closer than this to the nearest
        ROI boundary vertex are considered counted (TACS zone); farther ones
        are drawn as uncounted (green + orientation line).
    pixel_size : float
        Microns per pixel.  Used only so the bar chart label can state the
        threshold in µm alongside pixels.
    title : str
        Optional suptitle for the figure.
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    pr     = result.get('pixel_result', {})
    frs    = result.get('fiber_results', [])
    points = pr.get('points', [])

    if not points and not frs:
        print('[plot_tacs_heatmap] no data to plot')
        return

    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2,
        figsize       = (13, 5.5),
        gridspec_kw   = {'width_ratios': [1.6, 1]},
    )

    # ── Left panel: spatial heatmap ───────────────────────────────────────
    # Threshold in µm — same unit as each point's dist_to_boundary field.
    thr_um   = boundary_dist_threshold_px * pixel_size
    n_px_in  = 0
    n_px_out = 0
    _sc      = None   # colorbar handle; set only when pts_in is non-empty

    if points:
        pts_in  = [p for p in points if p['dist_to_boundary'] <= thr_um]
        pts_out = [p for p in points if p['dist_to_boundary'] >  thr_um]
        n_px_in, n_px_out = len(pts_in), len(pts_out)

        # Within threshold: heatmap colored by angle-to-tangent
        if pts_in:
            xs_in  = np.array([p['x']                for p in pts_in])
            ys_in  = np.array([p['y']                for p in pts_in])
            ang_in = np.array([p['angle_to_tangent'] for p in pts_in])
            _sc = ax_map.scatter(
                xs_in, ys_in, c=ang_in,
                cmap      = 'RdYlBu_r',
                vmin      = 0, vmax = 90,
                s         = 6,
                alpha     = 0.55,
                linewidths= 0,
                rasterized= True,
                zorder    = 2,
            )

        # Beyond threshold: green dot + dashed gray orientation line
        # (outside ROI, beyond the distance threshold — not counted)
        if pts_out:
            xs_out = np.array([p['x'] for p in pts_out])
            ys_out = np.array([p['y'] for p in pts_out])
            ax_map.scatter(
                xs_out, ys_out,
                c='limegreen', s=4, alpha=0.55,
                linewidths=0, rasterized=True, zorder=2,
            )
            from matplotlib.collections import LineCollection
            segs = []
            for p in pts_out:
                r  = np.radians(p['orientation'])
                dx = 2.0 * np.cos(r)
                dy = 2.0 * np.sin(r)
                segs.append(
                    [(p['x'] - dx, p['y'] - dy), (p['x'] + dx, p['y'] + dy)]
                )
            lc = LineCollection(
                segs, colors='gray', linewidths=0.6,
                alpha=0.6, linestyle='dashed', rasterized=True, zorder=3,
            )
            ax_map.add_collection(lc)

    if _sc is not None:
        cbar = fig.colorbar(_sc, ax=ax_map, fraction=0.035, pad=0.02)
        cbar.set_label('Angle to boundary tangent (°)', fontsize=8)
        cbar.set_ticks([0, 30, 60, 90])

    # ── Inside-ROI items: gray, dashed orientation lines ─────────────────
    # These were never passed to analyze_tacs_zone and represent collagen
    # signal enclosed within the tumor ROI (excluded from TACS computation).
    _has_inside = False
    if inside_roi_pixels:
        from matplotlib.collections import LineCollection as _LC
        xi = np.array([p['x'] for p in inside_roi_pixels])
        yi = np.array([p['y'] for p in inside_roi_pixels])
        ax_map.scatter(xi, yi, c='dimgray', s=4, alpha=0.45,
                       linewidths=0, rasterized=True, zorder=2)
        segs_in = []
        for p in inside_roi_pixels:
            r  = np.radians(p['orientation'])
            dx = 2.0 * np.cos(r)
            dy = 2.0 * np.sin(r)
            segs_in.append([(p['x'] - dx, p['y'] - dy),
                            (p['x'] + dx, p['y'] + dy)])
        ax_map.add_collection(_LC(segs_in, colors='dimgray', linewidths=0.6,
                                  alpha=0.5, linestyle='dashed',
                                  rasterized=True, zorder=3))
        _has_inside = True
    if inside_roi_fibers:
        from matplotlib.collections import LineCollection as _LC2
        segs_if = []
        for fib in inside_roi_fibers:
            center = getattr(fib, 'center_point', None)
            if center is None:
                cl = getattr(fib, 'centerline', None)
                if cl is not None and len(cl) >= 1:
                    center = cl[len(cl) // 2]
            if center is None:
                continue
            fx, fy = float(center[1]), float(center[0])
            ax_map.scatter(fx, fy, c='dimgray', s=60,
                           edgecolors='white', linewidths=0.6,
                           alpha=0.65, zorder=5)
            r  = np.radians(getattr(fib, 'angle', 0.0))
            dx = 2.0 * np.cos(r)
            dy = 2.0 * np.sin(r)
            segs_if.append([(fx - dx, fy - dy), (fx + dx, fy + dy)])
        if segs_if:
            ax_map.add_collection(_LC2(segs_if, colors='dimgray', linewidths=1.2,
                                       alpha=0.65, linestyle='dashed', zorder=6))
        _has_inside = True

    # ROI boundary — magenta to stand out against both dark background and
    # the RdYlBu colormap used for the pixel scatter.
    if roi.coordinates is not None:
        bx = np.append(roi.coordinates[:, 0], roi.coordinates[0, 0])
        by = np.append(roi.coordinates[:, 1], roi.coordinates[0, 1])
        ax_map.plot(bx, by, color='magenta', lw=1.8, label='ROI boundary', zorder=3)

    # ── Fiber objects: split by distance threshold ────────────────────────
    # All fiber_objects are shown whether or not they passed the analyze_tacs_zone
    # distance filter.  Distance is recomputed here in pixels to the nearest
    # ROI boundary vertex so the visualization threshold is independent of the
    # µm-based pipeline threshold.
    if fiber_objects and roi.coordinates is not None:
        fr_by_id  = {fr['fiber_id']: fr for fr in frs}
        roi_xy    = roi.coordinates          # (N, 2) in (x, y)
        # thr_um already computed in pixel-split block above

        n_counted   = 0
        n_uncounted = 0

        for fib in fiber_objects:
            # Resolve fiber center (row, col)
            center = getattr(fib, 'center_point', None)
            if center is None:
                cl = getattr(fib, 'centerline', None)
                if cl is not None and len(cl) >= 1:
                    center = cl[len(cl) // 2]
            if center is None:
                continue

            fx, fy = float(center[1]), float(center[0])   # (col=x, row=y)

            # Pixel distance to nearest ROI boundary vertex
            dist_px = float(
                np.min(np.linalg.norm(roi_xy - np.array([fx, fy]), axis=1))
            )

            fid = getattr(fib, 'object_id', None)

            if dist_px <= boundary_dist_threshold_px:
                # ── Counted: TACS-colored dot + red orientation line
                #             + blue connector to nearest boundary point ──
                fr       = fr_by_id.get(fid)
                tacs_lbl = (fr.get('tacs_like') if fr else None) or 'TACS-1-like'
                color    = _TACS_COLORS.get(tacs_lbl, 'grey')
                ax_map.scatter(
                    fx, fy,
                    c=color, s=90,
                    edgecolors='white', linewidths=0.8,
                    zorder=5,
                )
                # 6-px red orientation line (3 px each side)
                angle_rad = np.radians(getattr(fib, 'angle', 0.0))
                odx = 3.0 * np.cos(angle_rad)
                ody = 3.0 * np.sin(angle_rad)
                ax_map.plot(
                    [fx - odx, fx + odx],
                    [fy - ody, fy + ody],
                    color='red', lw=1.4, zorder=6,
                    solid_capstyle='round',
                )
                # Blue connector: center → nearest ROI boundary vertex
                dists_to_boundary = np.linalg.norm(
                    roi_xy - np.array([fx, fy]), axis=1
                )
                nearest_idx = int(np.argmin(dists_to_boundary))
                bpx, bpy = roi_xy[nearest_idx, 0], roi_xy[nearest_idx, 1]
                ax_map.plot(
                    [fx, bpx], [fy, bpy],
                    color='deepskyblue', lw=0.8, alpha=0.7,
                    zorder=4, solid_capstyle='round',
                )
                n_counted += 1
            else:
                # ── Uncounted: green dot + dashed gray orientation line ────
                angle_rad = np.radians(getattr(fib, 'angle', 0.0))
                dx = 2.0 * np.cos(angle_rad)
                dy = 2.0 * np.sin(angle_rad)
                ax_map.scatter(
                    fx, fy,
                    c='limegreen', s=35,
                    linewidths=0, zorder=5,
                )
                ax_map.plot(
                    [fx - dx, fx + dx],
                    [fy - dy, fy + dy],
                    color='gray', lw=1.2, zorder=6,
                    linestyle='dashed', solid_capstyle='round',
                )
                n_uncounted += 1

        # ── Legend ────────────────────────────────────────────────────────
        import matplotlib.lines as mlines
        tacs_patches = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in _TACS_COLORS.items()
        ]
        px_uncounted_dot  = mlines.Line2D(
            [], [], marker='o', color='limegreen', linestyle='None',
            markersize=4, label=f'pixel beyond {boundary_dist_threshold_px} px',
        )
        fib_uncounted_dot = mlines.Line2D(
            [], [], marker='o', color='limegreen', linestyle='None',
            markersize=7, label=f'fiber beyond {boundary_dist_threshold_px} px',
        )
        uncounted_line = mlines.Line2D(
            [], [], color='gray', lw=1.2, linestyle='dashed',
            label='absolute orientation (uncounted)',
        )
        beyond_items = (
            ([px_uncounted_dot]  if n_px_out   > 0 else [])
            + ([fib_uncounted_dot] if n_uncounted > 0 else [])
            + ([uncounted_line]    if n_px_out + n_uncounted > 0 else [])
        )
        counted_items = [
            mlines.Line2D([], [], color='red', lw=1.4,
                          label='fiber orientation (counted)'),
            mlines.Line2D([], [], color='deepskyblue', lw=0.8,
                          label='fiber → nearest boundary'),
        ] if n_counted > 0 else []
        import matplotlib.lines as _ml2
        inside_items = (
            [_ml2.Line2D([], [], marker='o', color='dimgray', linestyle='None',
                         markersize=5, alpha=0.65,
                         label='inside ROI (not counted)')]
            if _has_inside else []
        )
        ax_map.legend(
            handles    = [mpatches.Patch(color='magenta', label='ROI boundary')]
                         + tacs_patches
                         + counted_items
                         + beyond_items
                         + inside_items,
            fontsize   = 7,
            loc        = 'upper right',
            framealpha = 0.6,
        )
        print(f'  Pixel split: in (<={boundary_dist_threshold_px} px): {n_px_in}  beyond: {n_px_out}')
        print(f'  Fiber split: counted (<={boundary_dist_threshold_px} px): {n_counted}'
              f'  uncounted (>{boundary_dist_threshold_px} px): {n_uncounted}')
        ax_map.set_title(
            f'threshold: {boundary_dist_threshold_px} px ({thr_um:.0f} µm)  |  '
            f'px {n_px_in} in / {n_px_out} out  |  '
            f'fibers {n_counted} counted / {n_uncounted} uncounted',
            fontsize=8,
        )
    else:
        ax_map.legend(
            handles    = [mpatches.Patch(color='magenta', label='ROI boundary')],
            fontsize   = 7, loc='upper right', framealpha=0.6,
        )
        ax_map.set_title(
            f'threshold: {boundary_dist_threshold_px} px ({thr_um:.0f} µm)  |  '
            f'px {n_px_in} in / {n_px_out} out',
            fontsize=8,
        )

    # ── Right panel: TACS distribution bar chart ──────────────────────────
    labels        = ['TACS-1-like', 'TACS-2-like', 'TACS-3-like']
    colors        = [_TACS_COLORS[l] for l in labels]
    pixel_dist    = pr.get('tacs_distribution', {})
    pixel_counts  = [pixel_dist.get(l, 0) for l in labels]

    # Fiber distribution (if available)
    fiber_dist   = {}
    for fr in frs:
        k = fr.get('tacs_like') or 'unknown'
        fiber_dist[k] = fiber_dist.get(k, 0) + 1
    fiber_counts = [fiber_dist.get(l, 0) for l in labels]
    has_fibers   = any(fiber_counts)

    x_pos = np.arange(len(labels))
    bar_w = 0.38 if has_fibers else 0.55

    pixel_bars = ax_bar.bar(
        x_pos - (bar_w / 2 if has_fibers else 0),
        pixel_counts,
        width      = bar_w,
        color      = colors,
        edgecolor  = 'white',
        linewidth  = 0.8,
        label      = 'pixels',
    )
    for bar, cnt in zip(pixel_bars, pixel_counts):
        if cnt > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                str(cnt), ha='center', va='bottom', fontsize=7,
            )

    if has_fibers:
        fiber_bars = ax_bar.bar(
            x_pos + bar_w / 2,
            fiber_counts,
            width     = bar_w,
            color     = colors,
            edgecolor = 'black',
            linewidth = 0.8,
            hatch     = '//',
            alpha     = 0.75,
            label     = 'fibers',
        )
        for bar, cnt in zip(fiber_bars, fiber_counts):
            if cnt > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    str(cnt), ha='center', va='bottom', fontsize=7,
                )
        ax_bar.legend(fontsize=8)

    max_count = max(pixel_counts + fiber_counts + [1])
    ax_bar.set_ylim(0, max_count * 1.18)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([l.replace('-like', '\n(like)') for l in labels],
                           fontsize=8)
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('TACS-like distribution')
    ax_bar.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Heatmap saved → {save_path}')
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main example
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=== analyze_tacs_zone example ===\n')

    roi                          = _make_synthetic_roi()
    orientation_map              = _make_orientation_map()
    alignment_map                = _make_alignment_map(orientation_map)
    fibers                       = _make_fiber_objects(100)
    inside_pixels, inside_fibers = _make_inside_roi_items()
    image_size                   = orientation_map.shape   # (H, W)

    # ── Case 1: sparse polygon boundary, pixel-map only ───────────────────
    # Use this when:
    #   • Your ROI is a polygon with many vertices (smoothly curved).
    #   • You only have an orientation map (no individual fiber objects).
    #   • Speed matters more than tangent accuracy at sharp corners.
    print('[Case 1] sparse polygon boundary, pixel-map only')
    result1 = analyze_tacs_zone(
        orientation_map = orientation_map,
        alignment_map   = alignment_map,
        roi             = roi,
        fiber_objects   = None,       # skip per-object path
        pixel_size      = 1.0,        # µm/pixel  ← adjust to your image
        tacs_zone_width = 100.0,      # µm — include pixels within 100 µm
        subsample       = 4,          # process every 4th pixel (fast demo)
        dense_boundary  = False,      # 2-point tangent
    )
    pr1 = result1['pixel_result']
    print(f"  ROI label          : {result1['roi_label']}")
    print(f"  Pixels in zone     : {pr1.get('n_points_in_zone', 0)}")
    if pr1.get('n_points_in_zone', 0) > 0:
        print(f"  Mean angle/tangent : {pr1['mean_angle_to_tangent']:.1f}°"
              f"  std: {pr1['std_angle_to_tangent']:.1f}°")
        print(f"  TACS distribution  : {pr1['tacs_distribution']}")
    print()

    # ── Case 2: dense boundary + fiber objects ────────────────────────────
    # Use this when:
    #   • You have individually detected fiber objects (CTFire, skeleton, etc.).
    #   • Your ROI has sharp corners or high curvature — dense_boundary=True
    #     gives more accurate tangent angles via the polynomial-fit method.
    #   • You want all three relative angles per fiber
    #     (tangent, ROI-axis, centers-line).
    #
    # The polygon is automatically discretized once via discretize_roi_boundary
    # and the dense trace is shared by both the pixel and fiber analysis paths.
    print('[Case 2] dense boundary (auto-discretized polygon), with fiber objects')
    result2 = analyze_tacs_zone(
        orientation_map = orientation_map,
        alignment_map   = alignment_map,
        roi             = roi,
        fiber_objects   = fibers,
        pixel_size      = 1.0,
        tacs_zone_width = 200.0,   # wider zone so pixels/fibers at 100–200 px
                                   # appear as ‘beyond threshold’ in the heatmap
        subsample       = 4,
        dense_boundary  = True,   # polynomial-fit tangent; auto-discretizes polygon
        image_size      = image_size,   # enables regionprops ROI orientation
    )
    pr2  = result2['pixel_result']
    frs  = result2['fiber_results']
    print(f"  ROI label          : {result2['roi_label']}")
    print(f"  Dense boundary used: {result2['dense_boundary_used']}")
    print(f"  Pixels in zone     : {pr2.get('n_points_in_zone', 0)}")
    if pr2.get('n_points_in_zone', 0) > 0:
        print(f"  Mean angle/tangent (pixel): {pr2['mean_angle_to_tangent']:.1f}°")
    print(f"  Fiber objects in zone: {len(frs)}")
    if frs:
        print(f"  Mean angle/tangent (fibers): "
              f"{np.mean([f['angle_to_boundary_tangent'] for f in frs]):.1f}°")
        print(f"  Sample fiber record: {frs[0]}")
    print(f"  Combined mean angle/tangent: "
          f"{result2['combined_mean_angle_to_tangent']:.1f}°")
    print()

    # ── TACS-3-like fiber query ────────────────────────────────────────────
    tacs3_fibers = [f for f in frs if f.get('tacs_like') == 'TACS-3-like']
    print(f'  TACS-3-like fibers: {len(tacs3_fibers)}')
    if tacs3_fibers:
        header = (f"  {'ID':>4}  {'Center(row,col)':>22}  "
                  f"{'Orient(deg)':>11}  {'BdryPt(row,col)':>22}  "
                  f"{'RelAngle(deg)':>13}")
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for f in tacs3_fibers:
            cp  = f['center_point']
            bp  = f['nearest_boundary_point']
            print(f"  {str(f['fiber_id']):>4}  "
                  f"({cp[0]:7.1f},{cp[1]:7.1f})  "
                  f"{f['angle']:>11.1f}  "
                  f"({bp[0]:7.1f},{bp[1]:7.1f})  "
                  f"{f['angle_to_boundary_tangent']:>13.1f}")
    print()

    # ── Heatmap visualisation (Case 2 result) ─────────────────────────────
    print('[Heatmap] plotting TACS distribution heatmap ...')
    plot_tacs_heatmap(
        result                     = result2,
        roi                        = roi,
        fiber_objects              = fibers,
        inside_roi_pixels          = inside_pixels,
        inside_roi_fibers          = inside_fibers,
        boundary_dist_threshold_px = 100,
        pixel_size                 = 1.0,
        title                      = 'TACS zone analysis — dense boundary, with fiber objects',
    )
    print()

    # ── Case 3: dense boundary supplied externally ────────────────────────
    # Use this when:
    #   • You already have a dense boundary trace (e.g. from CurveAlign's
    #     boundary output, or a skeletonised mask perimeter).
    #   • You want to avoid the cost of re-discretizing on repeated calls
    #     (e.g. iterating over many ROIs in a batch).
    #
    # Pre-compute the dense trace once, then pass it via dense_coords.
    print('[Case 3] pre-computed dense boundary, pixel-map only')
    poly_rc     = roi.coordinates[:, ::-1]    # (x,y) → (row,col)
    dense_trace = discretize_roi_boundary(poly_rc, step=1.0)
    print(f"  Dense boundary: {len(dense_trace)} points  "
          f"(from {len(roi.coordinates)} polygon vertices)")

    result3 = analyze_tacs_zone(
        orientation_map = orientation_map,
        alignment_map   = None,
        roi             = roi,
        fiber_objects   = None,
        pixel_size      = 1.0,
        tacs_zone_width = 100.0,
        subsample       = 4,
        dense_boundary  = True,
        # Pass it in — analyze_tacs_zone forwards dense_coords to
        # compute_orientation_relative_to_roi automatically
        # (requires roi.coordinates for the fiber path even when dense_coords
        # is provided externally; it is used for the per-object distance filter)
        image_size      = image_size,
    )
    pr3 = result3['pixel_result']
    print(f"  Pixels in zone : {pr3.get('n_points_in_zone', 0)}")
    if pr3.get('n_points_in_zone', 0) > 0:
        print(f"  Mean angle/tangent: {pr3['mean_angle_to_tangent']:.1f}°")
    print()


if __name__ == '__main__':
    main()
