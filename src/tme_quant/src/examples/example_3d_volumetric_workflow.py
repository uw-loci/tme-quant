"""
TMEQuant Example — Workflow 3: 3-D Volumetric Fiber Analysis
=============================================================

End-to-end demonstration of 3-D volumetric fiber analysis:

  1.  Load a 3-D SHG volume (Z, H, W).
  2.  CurveAlign 3-D orientation analysis (true volumetric curvelet
      transform; warns if falling back to slice-by-slice).
  3.  Skeleton 3-D fiber extraction (Lee algorithm; fibers span z-planes;
      centerlines are (N, 3) in (z, row, col)).
  4.  CT-FIRE 3-D status check (requires C++ FIRE extension).
  5.  Print measurement tables:
        — volume metadata
        — 3-D orientation summary
        — per-fiber skeleton properties
  6.  Display a figure panel:
        — middle slice of SHG volume
        — middle slice of orientation map
        — middle slice of skeleton
        — fiber length / straightness / orientation distributions

Coordinate conventions
-----------------------
  Volume shape:    (Z, H, W)
  Centerlines:     (N, 3) float32 in (z, row, col) — not (x, y, z).
  Orientation map: same shape as volume, in degrees [0°, 180°).

CT-FIRE 3-D status
-------------------
  The CT stage (curvelet transform) is fully implemented for 3-D.
  The FIRE stage (distance-transform ridge tracing) requires the C++
  extension to run in 3-D.  Until then use extract_3d() with SkeletonParams.
  Check at runtime:
    from tme_quant.fiber_analysis.utils import ctfire_backend_status
    print(ctfire_backend_status())   # {'cpp_available': bool, '3d_supported': bool, …}

Usage
-----
  python example_3d_volumetric_workflow.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# ── Fiber analysis ────────────────────────────────────────────────────────────
from tme_quant.fiber_analysis import FiberAnalyzer, FiberOrientationAnalyzer
from tme_quant.fiber_analysis.config import (
    CurveAlignParams,
    CTFireParams,
    SkeletonParams,
    FiberProperties,
)
from tme_quant.fiber_analysis.utils import available_backends, ctfire_backend_status


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_fiber_properties_df(
    fibers: List[FiberProperties],
    pixel_size: float,
    z_spacing: float,
) -> pd.DataFrame:
    """
    Convert a list of 3-D FiberProperties into a tidy pandas DataFrame.

    Columns: fiber_id, n_centerline_pts, length_um, straightness,
             orientation_deg, z_min, z_max, z_span_slices, midpoint_z,
             midpoint_row, midpoint_col.
    """
    rows = []
    for i, fib in enumerate(fibers):
        cl = getattr(fib, 'centerline', None)
        row: Dict = {
            'fiber_id':    getattr(fib, 'object_id', getattr(fib, 'fiber_id', i)),
            'length_um':   getattr(fib, 'length', float('nan')),
            'straightness':getattr(fib, 'straightness', float('nan')),
            'orientation': getattr(fib, 'angle', float('nan')),
        }
        if cl is not None and len(cl) >= 2:
            row['n_centerline_pts'] = len(cl)
            # (z, row, col) convention
            row['z_min']           = float(cl[:, 0].min())
            row['z_max']           = float(cl[:, 0].max())
            row['z_span_slices']   = float(cl[:, 0].max() - cl[:, 0].min())
            mid = cl[len(cl) // 2]
            row['midpoint_z']   = float(mid[0])
            row['midpoint_row'] = float(mid[1])
            row['midpoint_col'] = float(mid[2])
        else:
            for k in ('n_centerline_pts', 'z_min', 'z_max', 'z_span_slices',
                      'midpoint_z', 'midpoint_row', 'midpoint_col'):
                row[k] = float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def _save_slice_figure(
    volume: np.ndarray,
    orientation_map: Optional[np.ndarray],
    skeleton_mask: Optional[np.ndarray],
    output_dir: Path,
    sample_id: str,
) -> Dict[str, Path]:
    """
    Save mid-volume slice images for orientation map and skeleton.
    Returns dict of saved paths.
    """
    mid_z   = volume.shape[0] // 2
    saved: Dict[str, Path] = {}

    # SHG slice
    slc = volume[mid_z]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(slc, cmap='gray')
    ax.set_title(f'{sample_id} — SHG Z={mid_z}', fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    p = output_dir / f'{sample_id}_shg_slice_z{mid_z}.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved['shg_slice'] = p

    # Orientation map slice
    if orientation_map is not None:
        omap = orientation_map[mid_z] if orientation_map.ndim == 3 else orientation_map
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(omap, cmap='hsv', vmin=0, vmax=180)
        plt.colorbar(im, ax=ax, label='Orientation (°)')
        ax.set_title(f'{sample_id} — Orientation Z={mid_z}', fontsize=11)
        ax.axis('off')
        fig.tight_layout()
        p = output_dir / f'{sample_id}_orientation_slice_z{mid_z}.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved['orientation_slice'] = p

    # Skeleton slice
    if skeleton_mask is not None:
        skel = skeleton_mask[mid_z] if skeleton_mask.ndim == 3 else skeleton_mask
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(slc, cmap='gray', alpha=0.7)
        ax.imshow(skel, cmap='Reds', alpha=0.8)
        ax.set_title(f'{sample_id} — Skeleton Z={mid_z}', fontsize=11)
        ax.axis('off')
        fig.tight_layout()
        p = output_dir / f'{sample_id}_skeleton_slice_z{mid_z}.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved['skeleton_slice'] = p

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def workflow_3d_volumetric(
    shg_volume_path: str,
    output_dir: str,
    pixel_size: float = 0.5,
    z_spacing: float = 1.0,
    sample_id: str = 'patient_001_3d',
) -> Dict:
    """
    3-D volumetric fiber analysis.

    Returns
    -------
    Dict with keys: sample_id, volume, orient_result_3d, skel_result_3d,
    fibers_3d, fiber_df, ctfire_result_3d, summary_stats, saved_figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('  Workflow: 3-D Volumetric Fiber Analysis')
    print(f'  Sample  : {sample_id}')
    print('=' * 72)

    backends    = available_backends()
    fire_status = ctfire_backend_status()
    print(f'\n  Curvelet backends: {backends}')
    print(f'  CT-FIRE status   : {fire_status}')
    if not backends['curvelops']:
        print('  WARNING: curvelops not installed — 3-D CurveAlign will fall '
              'back to slice-by-slice.  Install with: pip install curvelops')
    if not fire_status['3d_supported']:
        print('  NOTE: 3-D CT-FIRE not available. Using SkeletonParams instead.')

    # ── [1/4] Load volume ─────────────────────────────────────────────────────
    print('\n[1/4] Loading 3-D SHG volume...')
    volume = io.imread(shg_volume_path)
    if volume.ndim == 2:
        raise ValueError(
            f'Expected a 3-D volume (Z, H, W), got shape {volume.shape}. '
            'Pass a multi-page TIFF or reshape accordingly.'
        )
    Z, H, W = volume.shape[0], volume.shape[1], volume.shape[2]
    total_um_z = Z * z_spacing
    total_um_h = H * pixel_size
    total_um_w = W * pixel_size
    print(f'  Volume : {volume.shape}  (Z={Z}, H={H}, W={W})')
    print(f'  Size   : {total_um_z:.1f} × {total_um_h:.1f} × {total_um_w:.1f} µm')
    print(f'  Spacing: xy={pixel_size} µm/px,  z={z_spacing} µm/slice')

    # ── [2/4] 3-D CurveAlign orientation ─────────────────────────────────────
    print('\n[2/4] 3-D CurveAlign orientation analysis...')
    orient_params_3d = CurveAlignParams(
        pixel_size=pixel_size,
        window_size=32,
        overlap=0.5,
        curvelet_levels=3,
        curvelet_angles=8,
        compute_coherency=True,
        compute_energy=False,
        keep_values=['angles', 'alignment'],
        compute_statistics=True,
    )
    orientation_analyzer = FiberOrientationAnalyzer()
    orient_result_3d     = orientation_analyzer.analyze_3d(volume, orient_params_3d)
    omap = orient_result_3d.orientation_map
    print(f'  Orientation volume shape: {omap.shape}')
    print(f'  Mean orientation        : {orient_result_3d.mean_orientation:.2f}°')
    print(f'  Alignment score         : {orient_result_3d.alignment_score:.4f}')

    # Per-slice orientation stats
    slice_means = []
    for z in range(omap.shape[0]):
        slc = omap[z]
        valid = slc[~np.isnan(slc)]
        if len(valid):
            slice_means.append(float(np.mean(valid)))
    print(f'  Slice orientation range : '
          f'{min(slice_means):.1f}° – {max(slice_means):.1f}°'
          if slice_means else '  (no valid pixels)')

    # ── [3/4] 3-D Skeleton fiber extraction ──────────────────────────────────
    print('\n[3/4] 3-D volumetric skeleton fiber extraction (Lee)...')
    skel_params_3d = SkeletonParams(
        pixel_size=pixel_size,
        skeleton_method='lee',
        threshold_method='otsu',
        min_branch_length=5.0,
        smooth_skeleton=True,
        min_fiber_length=10.0,
        max_fiber_length=2000.0,
        extract_centerlines=True,
    )
    fiber_analyzer_3d = FiberAnalyzer()
    skel_result_3d    = fiber_analyzer_3d.extract_3d(volume, skel_params_3d)
    fibers_3d         = skel_result_3d.fibers
    print(f'  Extracted {len(fibers_3d):,} 3-D fibers')
    print(f'  Skeleton volume shape  : {skel_result_3d.skeleton_mask.shape}')
    if fibers_3d and fibers_3d[0].centerline is not None:
        cl = fibers_3d[0].centerline
        print(f'  First fiber centerline : shape={cl.shape}, '
              f'z_range=[{cl[:,0].min():.0f}, {cl[:,0].max():.0f}]')
    if fibers_3d:
        lengths = [f.length for f in fibers_3d if f.length is not None]
        strghts = [f.straightness for f in fibers_3d if f.straightness is not None]
        if lengths:
            print(f'  Mean length      : {np.mean(lengths):.2f} µm')
        if strghts:
            print(f'  Mean straightness: {np.mean(strghts):.3f}')

    fiber_df = _build_fiber_properties_df(fibers_3d, pixel_size, z_spacing)

    # ── [4/4] CT-FIRE 3-D status ──────────────────────────────────────────────
    print('\n[4/4] CT-FIRE 3-D status check...')
    ctfire_result_3d: Optional[object] = None
    if fire_status['3d_supported']:
        ctfire_params_3d = CTFireParams(
            pixel_size=pixel_size,
            z_spacing=z_spacing,
            ctfire_threshold=0.1,
            ctfire_n_levels=3,
            ctfire_n_angles=8,
            min_fiber_length=10.0,
            extract_centerlines=True,
        )
        ctfire_result_3d = fiber_analyzer_3d.extract_3d(volume, ctfire_params_3d)
        print(f'  CT-FIRE 3-D: extracted {len(ctfire_result_3d.fibers):,} fibers')
    else:
        print('  CT-FIRE 3-D requires the _ctfire_cpp extension (not yet compiled).')
        print('  The CT stage (curvelet transform) is already 3-D-capable.')
        print('  The FIRE stage (distance-transform ridge tracing) needs C++ for 3-D.')
        print('  Using Skeleton 3-D result above as the 3-D fiber extraction output.')

    # ── Save slice figures ────────────────────────────────────────────────────
    print('\nSaving slice figures...')
    saved_figures = _save_slice_figure(
        volume=volume,
        orientation_map=orient_result_3d.orientation_map,
        skeleton_mask=skel_result_3d.skeleton_mask,
        output_dir=out,
        sample_id=sample_id,
    )
    fiber_df.to_csv(out / f'{sample_id}_fiber_properties_3d.csv', index=False)
    print(f'  Saved to {out}')

    summary_stats: Dict = {
        'sample_id':        sample_id,
        'volume_shape':     str(volume.shape),
        'z_slices':         Z,
        'height_px':        H,
        'width_px':         W,
        'pixel_size_um':    pixel_size,
        'z_spacing_um':     z_spacing,
        'size_z_um':        total_um_z,
        'size_h_um':        total_um_h,
        'size_w_um':        total_um_w,
        'mean_orientation': float(orient_result_3d.mean_orientation),
        'alignment_score':  float(orient_result_3d.alignment_score),
        'n_fibers_3d':      len(fibers_3d),
        'mean_length_um':   float(fiber_df['length_um'].mean()) if len(fiber_df) else float('nan'),
        'mean_straightness':float(fiber_df['straightness'].mean()) if len(fiber_df) else float('nan'),
        'mean_z_span_slices': float(fiber_df['z_span_slices'].mean()) if len(fiber_df) else float('nan'),
        'ctfire_3d_available': fire_status['3d_supported'],
    }

    print('\n' + '=' * 72)
    print('  3-D VOLUMETRIC WORKFLOW COMPLETE')
    print('=' * 72)

    return {
        'sample_id':        sample_id,
        'volume':           volume,
        'orient_result_3d': orient_result_3d,
        'skel_result_3d':   skel_result_3d,
        'fibers_3d':        fibers_3d,
        'fiber_df':         fiber_df,
        'ctfire_result_3d': ctfire_result_3d,
        'summary_stats':    summary_stats,
        'output_dir':       out,
        'saved_figures':    saved_figures,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _print_volume_summary_table(summary_stats: Dict) -> None:
    """Print a formatted volume metadata and analysis summary table."""
    rows = [
        ('Volume shape',         summary_stats['volume_shape']),
        ('Z slices',             summary_stats['z_slices']),
        ('Height (px)',          summary_stats['height_px']),
        ('Width (px)',           summary_stats['width_px']),
        ('XY pixel size (µm)',   f"{summary_stats['pixel_size_um']:.3f}"),
        ('Z spacing (µm)',       f"{summary_stats['z_spacing_um']:.3f}"),
        ('Volume Z (µm)',        f"{summary_stats['size_z_um']:.1f}"),
        ('Volume H (µm)',        f"{summary_stats['size_h_um']:.1f}"),
        ('Volume W (µm)',        f"{summary_stats['size_w_um']:.1f}"),
        ('─── CurveAlign 3-D ─',''),
        ('Mean orientation (°)', f"{summary_stats['mean_orientation']:.2f}"),
        ('Alignment score',      f"{summary_stats['alignment_score']:.4f}"),
        ('─── Skeleton 3-D ───',''),
        ('Fibers extracted',     summary_stats['n_fibers_3d']),
        ('Mean length (µm)',     f"{summary_stats['mean_length_um']:.2f}"),
        ('Mean straightness',    f"{summary_stats['mean_straightness']:.3f}"),
        ('Mean z-span (slices)', f"{summary_stats['mean_z_span_slices']:.1f}"),
        ('─── CT-FIRE 3-D ────',''),
        ('CT-FIRE 3-D available',str(summary_stats['ctfire_3d_available'])),
    ]
    width = 46
    print()
    print('┌' + '─' * width + '┐')
    print('│{:^{w}}│'.format(' 3-D Volumetric Summary ', w=width))
    print('├' + '─' * 26 + '┬' + '─' * (width - 27) + '┤')
    for label, value in rows:
        if label.startswith('─'):
            print('├' + '─' * 26 + '┼' + '─' * (width - 27) + '┤')
            print('│{:^{w}}│'.format(f' {label.strip("─ ")} ', w=width))
            print('├' + '─' * 26 + '┼' + '─' * (width - 27) + '┤')
        else:
            print('│ {:<24} │ {:<{w}} │'.format(label, str(value), w=width - 29))
    print('└' + '─' * 26 + '┴' + '─' * (width - 27) + '┘')


def _print_fiber_3d_table(
    fiber_df: pd.DataFrame,
    max_rows: int = 20,
) -> None:
    """Print a sample of 3-D fiber properties as a formatted table."""
    if len(fiber_df) == 0:
        print('  (no 3-D fiber data)')
        return

    cols = [
        'fiber_id', 'length_um', 'straightness', 'orientation',
        'n_centerline_pts', 'z_span_slices',
        'midpoint_z', 'midpoint_row', 'midpoint_col',
    ]
    sample_df = fiber_df.head(max_rows)[
        [c for c in cols if c in fiber_df.columns]
    ].copy()

    fmt = {
        'length_um':        '{:.1f}',
        'straightness':     '{:.3f}',
        'orientation':      '{:.1f}',
        'n_centerline_pts': '{:.0f}',
        'z_span_slices':    '{:.1f}',
        'midpoint_z':       '{:.1f}',
        'midpoint_row':     '{:.1f}',
        'midpoint_col':     '{:.1f}',
    }
    for col, f in fmt.items():
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].apply(
                lambda v: f.format(v) if pd.notna(v) else 'n/a'
            )

    print()
    print(f'  3-D fiber properties  '
          f'(showing {len(sample_df)} of {len(fiber_df):,} fibers)')
    print('  ' + '-' * 100)
    with pd.option_context(
        'display.max_columns', None,
        'display.width',       120,
        'display.max_colwidth', 14,
    ):
        print(sample_df.to_string(index=False))
    print()


def _print_per_slice_orientation(
    orientation_map: np.ndarray,
    max_slices: int = 10,
) -> None:
    """Print a per-slice orientation statistics table."""
    if orientation_map.ndim != 3:
        return
    n = orientation_map.shape[0]
    step = max(1, n // max_slices)
    slices = list(range(0, n, step))

    print()
    print(f'  Per-slice orientation statistics (every {step} slice(s), '
          f'{len(slices)} slices shown):')
    print('  ┌───────┬──────────────┬──────────────┬──────────────┬─────────┐')
    print('  │ Z     │ Mean (°)     │ Std (°)      │ Min (°)      │ Valid % │')
    print('  ├───────┼──────────────┼──────────────┼──────────────┼─────────┤')
    for z in slices:
        slc   = orientation_map[z]
        valid = slc[~np.isnan(slc)]
        if len(valid) == 0:
            print(f'  │ {z:>5} │ {"n/a":>12} │ {"n/a":>12} │ {"n/a":>12} │ {"0.0%":>7} │')
        else:
            pct = len(valid) / slc.size * 100
            print(f'  │ {z:>5} │ {np.mean(valid):>11.2f}° │ '
                  f'{np.std(valid):>11.2f}° │ '
                  f'{np.min(valid):>11.2f}° │ '
                  f'{pct:>6.1f}% │')
    print('  └───────┴──────────────┴──────────────┴──────────────┴─────────┘')


def print_measurement_tables(results: Dict) -> None:
    """Print volume summary, per-slice orientation, and per-fiber property tables."""
    _print_volume_summary_table(results['summary_stats'])
    _print_per_slice_orientation(results['orient_result_3d'].orientation_map)
    _print_fiber_3d_table(results['fiber_df'])


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: FIGURE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def display_figure_panel(results: Dict) -> None:
    """
    Show a figure panel with:
      Row 1:  middle slice SHG | middle slice orientation map | skeleton overlay
      Row 2:  fiber length dist | straightness dist | orientation dist
    """
    figs   = results['saved_figures']
    volume = results['volume']
    omap   = results['orient_result_3d'].orientation_map
    skel   = results['skel_result_3d'].skeleton_mask
    mid_z  = volume.shape[0] // 2

    # ── Row 1: image slices ───────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(
        f"3-D Volume Slices (Z={mid_z}) — {results['sample_id']}",
        fontsize=13, fontweight='bold',
    )

    axes1[0].imshow(volume[mid_z], cmap='gray')
    axes1[0].set_title('SHG image', fontsize=11)
    axes1[0].axis('off')

    if omap.ndim == 3:
        omap_slc = omap[mid_z]
    else:
        omap_slc = omap
    im = axes1[1].imshow(omap_slc, cmap='hsv', vmin=0, vmax=180)
    plt.colorbar(im, ax=axes1[1], label='Orientation (°)', fraction=0.046)
    axes1[1].set_title('Orientation map', fontsize=11)
    axes1[1].axis('off')

    axes1[2].imshow(volume[mid_z], cmap='gray', alpha=0.7)
    if skel is not None:
        skel_slc = skel[mid_z] if skel.ndim == 3 else skel
        axes1[2].imshow(skel_slc, cmap='Reds', alpha=0.8)
    axes1[2].set_title('Skeleton overlay', fontsize=11)
    axes1[2].axis('off')

    fig1.tight_layout()
    plt.show()

    # ── Row 2: fiber property distributions ──────────────────────────────────
    fiber_df = results['fiber_df']
    if len(fiber_df) == 0:
        return

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(
        f"3-D Fiber Property Distributions — {results['sample_id']}",
        fontsize=12, fontweight='bold',
    )
    for ax, col, xlabel, color in [
        (axes2[0], 'length_um',     'Fiber length (µm)',           'steelblue'),
        (axes2[1], 'straightness',  'Straightness',                'tomato'),
        (axes2[2], 'z_span_slices', 'Z span (slices)',             'mediumpurple'),
    ]:
        vals = fiber_df[col].dropna().values
        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            ax.hist(vals, bins=25, color=color, edgecolor='white', linewidth=0.5)
            ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.2,
                       label=f'mean={vals.mean():.2f}')
            ax.legend(fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(col.replace('_', ' ').title(), fontsize=11)

    fig2.tight_layout()
    plt.show()

    # ── 3-D fiber centroid scatter (z vs row) ─────────────────────────────────
    if 'midpoint_z' in fiber_df.columns and fiber_df['midpoint_z'].notna().any():
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sc = ax3.scatter(
            fiber_df['midpoint_col'].values,
            fiber_df['midpoint_z'].values,
            c=fiber_df['length_um'].values,
            cmap='plasma',
            s=20, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax3, label='Fiber length (µm)')
        ax3.set_xlabel('Column (px)', fontsize=11)
        ax3.set_ylabel('Z slice', fontsize=11)
        ax3.set_title(
            f"3-D Fiber Centroids (col vs Z) — {results['sample_id']}",
            fontsize=12, fontweight='bold',
        )
        fig3.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results: Dict) -> None:
    """Print measurement tables and show the figure panels."""
    print_measurement_tables(results)
    display_figure_panel(results)


if __name__ == '__main__':
    results = workflow_3d_volumetric(
        shg_volume_path='data/patient_001_SHG_3D.tif',
        output_dir='output/patient_001_3d',
        pixel_size=0.5,
        z_spacing=1.0,
        sample_id='patient_001_3d',
    )
    display_results(results)
