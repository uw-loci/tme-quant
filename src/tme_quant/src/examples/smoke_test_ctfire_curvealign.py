"""
Smoke test for examples_tmequant_complete_cl_usage_ctfire_curvealign.py
=======================================================================

Exercises every non-IO code path in the CT-FIRE / CurveAlign example without
requiring real image files, a GPU, or any optional backend (curvelops, Fiji,
MATLAB, CT-FIRE C++).

Tests covered (11 groups, 41+ checks)
--------------------------------------
  1.  Backend status          — available_backends(), ctfire_backend_status()
  2.  Param construction      — CurveAlignParams, CTFireParams, SkeletonParams
  3.  2-D CurveAlign          — analyze_orientation_2d, result shape / stats
  4.  2-D CT-FIRE             — extract_fibers_2d, fiber_mask, DT-based width
  5.  CurveAlign segments     — _extract_fiber_segments_from_curvealign
  6.  Segment metrics         — _compute_fiber_segment_metrics
  7.  Fiber metrics           — _compute_individual_fiber_metrics (FiberObject)
  8.  3-D CurveAlign          — analyze_orientation_3d, (Z,H,W) output
  9.  3-D Skeleton            — extract_fibers_3d with Lee algorithm
  10. 3-D CT-FIRE guard       — raises correctly when C++ absent
  11. TACS classification     — classify_fiber_tacs, segment_tacs_like, colors

Usage
-----
From the project root (where pyproject.toml lives):

    # With editable install (recommended):
    python src/examples/smoke_test_ctfire_curvealign.py

    # Without install, using PYTHONPATH:
    PYTHONPATH=src python src/examples/smoke_test_ctfire_curvealign.py

The shapely stub is kept so the test can also run in environments where
shapely is not yet installed.  When shapely IS installed (as in the normal
[examples] environment) the stub has no effect because sys.modules entries
are only inserted for modules not already imported.
"""

from __future__ import annotations

import sys
import types
import warnings
import traceback
import importlib.util
import pathlib

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Shapely stub (no-op when shapely is already installed) ────────────────────
# Inserted only for modules not already present in sys.modules so that
# environments with shapely installed are unaffected.
for _n in ['shapely', 'shapely.geometry', 'shapely.ops']:
    if _n not in sys.modules:
        sys.modules[_n] = types.ModuleType(_n)

if not hasattr(sys.modules['shapely.geometry'], 'Point'):
    class _G:
        """Minimal shapely geometry stub."""
        def __init__(self, *a, **k): pass
        def distance(self, other): return 150.0
        def project(self, pt):     return 0.0
        def interpolate(self, d):  return _G()
        @property
        def boundary(self):  return self
        @property
        def exterior(self):  return self
        @property
        def centroid(self):  return self
        @property
        def coords(self):    return [(0., 0.), (1., 0.), (1., 1.)]
        x = y = 0.0
        def __bool__(self):  return True

    for _cls in ['Point', 'Polygon', 'LineString', 'MultiPolygon', 'MultiPoint']:
        setattr(sys.modules['shapely.geometry'], _cls, _G)
    sys.modules['shapely.ops'].nearest_points = lambda a, b: (_G(), _G())

# ── Matplotlib backend — headless so no display is needed ────────────────────
import matplotlib
matplotlib.use('Agg')

# ── tme_quant imports ─────────────────────────────────────────────────────────
from tme_quant.fiber_analysis import FiberAnalyzer
from tme_quant.fiber_analysis.config import (
    CurveAlignParams,
    CTFireParams,
    SkeletonParams,
    CurveAlignResult,
    CTFireResult,
    SkeletonResult,
    FiberProperties,
)
from tme_quant.fiber_analysis.utils import available_backends, ctfire_backend_status
from tme_quant.fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
)
from tme_quant.tme_analysis.core.tacs_classifier import (
    classify_fiber_tacs,
    classify_fiber_segment_tacs_like,
    get_tacs_color,
)

print("PASS  All imports resolved")

# ── Load helper functions from the example module ─────────────────────────────
# The example module is loaded by path so that its private helpers
# (_extract_fiber_segments_from_curvealign, _compute_fiber_segment_metrics,
# _compute_individual_fiber_metrics) are accessible without executing the
# __main__ block.
_example_path = (
    pathlib.Path(__file__).parent
    / 'examples_tmequant_complete_cl_usage_ctfire_curvealign.py'
)
_spec = importlib.util.spec_from_file_location('_example_ctfire_ca', _example_path)
ex = importlib.util.module_from_spec(_spec)
ex.__name__ = '_example_ctfire_ca'   # prevents __main__ block from running
_spec.loader.exec_module(ex)
print("PASS  Example module loaded\n")

# ── Test harness ──────────────────────────────────────────────────────────────
PASS: list[str] = []
FAIL: list[str] = []


def ok(cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "assertion failed")


def check(name: str, fn):
    """Run fn(); record PASS or FAIL; return the result or None on failure."""
    try:
        result = fn()
        PASS.append(name)
        print(f"  PASS  {name}")
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc).split(chr(10))[0]}")
        relevant = [
            l for l in tb.split('\n')
            if l.strip() and ('Error' in l or '_example' in l or 'tme_quant' in l)
        ]
        for line in relevant[-3:]:
            print(f"        {line}")
        return None


# ── Synthetic fixtures ────────────────────────────────────────────────────────
# A 128×128 float32 image with a bright horizontal band — enough structure
# for the curvelet transform to find fibers without real SHG data.
IMG = np.zeros((128, 128), dtype=np.float32)
IMG[40:88, 20:108] = 0.85 + 0.1 * np.random.rand(48, 88).astype(np.float32)

# A 6×64×64 float32 volume with a bright slab for 3-D analysis.
VOL = np.zeros((6, 64, 64), dtype=np.float32)
VOL[2:4, 16:48, 16:48] = 0.85

# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Backend status
# ─────────────────────────────────────────────────────────────────────────────
print("── Group 1: Backend status ──────────────────────────────────────────")

check("available_backends() returns dict with 'numpy'",
      lambda: ok(available_backends()['numpy'] is True))
check("ctfire_backend_status() returns dict with '3d_supported'",
      lambda: ok('3d_supported' in ctfire_backend_status()))

# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Parameter construction
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 2: Param construction ──────────────────────────────────────")

ca_p = check("CurveAlignParams constructed",
    lambda: CurveAlignParams(
        pixel_size=0.5, window_size=32, overlap=0.5,
        curvelet_levels=3, curvelet_angles=8,
        compute_coherency=True, compute_energy=True,
        return_fiber_segments=False,
        keep_values=['angles', 'alignment', 'energy'],
        compute_statistics=True,
    ))

ct_p = check("CTFireParams constructed (incl. z_spacing)",
    lambda: CTFireParams(
        pixel_size=0.5, ctfire_threshold=0.05,
        ctfire_n_levels=3, ctfire_n_angles=8,
        straightness_threshold=0.0, use_matlab_backend=False,
        z_spacing=1.0,
        min_fiber_length=5.0, max_fiber_length=500.0,
        min_fiber_width=0.5, max_fiber_width=15.0,
        measure_length=True, measure_width=True,
        measure_straightness=True, measure_angle=True,
        measure_curvature=False, extract_centerlines=True,
    ))

sk_p = check("SkeletonParams constructed",
    lambda: SkeletonParams(
        pixel_size=0.5, skeleton_method='lee',
        threshold_method='otsu', min_branch_length=5.0,
        smooth_skeleton=True, min_fiber_length=5.0,
        extract_centerlines=True,
    ))

# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — 2-D CurveAlign orientation analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 3: 2-D CurveAlign orientation ──────────────────────────────")

a1 = FiberAnalyzer()
o2d = check("analyze_orientation_2d returns CurveAlignResult",
    lambda: a1.analyze_orientation_2d(IMG, ca_p, image_id="smoke"))

if o2d:
    check("orientation_map shape == (H,W)",
          lambda: ok(o2d.orientation_map.shape == (128, 128)))
    check("mean_orientation is float",
          lambda: ok(isinstance(o2d.mean_orientation, float)))
    check("mean_alignment in [0,1]",
          lambda: ok(0.0 <= o2d.mean_alignment <= 1.0))
    check("alignment_map not None (keep_values includes 'alignment')",
          lambda: ok(o2d.alignment_map is not None))
    check("energy_map not None (keep_values includes 'energy')",
          lambda: ok(o2d.energy_map is not None))
    check("n_windows_analyzed > 0",
          lambda: ok(o2d.n_windows_analyzed > 0))
    check("processing_time > 0",
          lambda: ok(o2d.processing_time > 0))

# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — 2-D CT-FIRE fiber extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 4: 2-D CT-FIRE extraction ──────────────────────────────────")

a2 = FiberAnalyzer()
ctr = check("extract_fibers_2d returns CTFireResult",
    lambda: a2.extract_fibers_2d(IMG, ct_p, image_id="smoke"))

if ctr:
    check("fibers is a list",
          lambda: ok(isinstance(ctr.fibers, list)))
    check("fiber_mask is bool array, shape (H,W)",
          lambda: ok(ctr.fiber_mask.dtype == bool
                     and ctr.fiber_mask.shape == (128, 128)))
    check("curvelet_energy_map shape == (H,W)",
          lambda: ok(ctr.curvelet_energy_map.shape == (128, 128)))
    check("n_candidates >= 0",
          lambda: ok(ctr.n_candidates >= 0))
    check("processing_time > 0",
          lambda: ok(ctr.processing_time > 0))
    check("total_fiber_count == len(fibers)",
          lambda: ok(ctr.total_fiber_count == len(ctr.fibers)))

    if ctr.fibers:
        # After extraction_analyzer conversion, fibers are FiberObject instances.
        f0 = ctr.fibers[0]
        check("fiber has object_id (FiberObject after conversion)",
              lambda: ok(hasattr(f0, 'object_id') and f0.object_id != ''))
        check("fiber.width >= 0 (distance-transform based)",
              lambda: ok(f0.width >= 0))
        check("fiber.length > 0",
              lambda: ok(f0.length > 0))
        check("centerline is (N,2)",
              lambda: ok(
                  f0.centerline is not None
                  and f0.centerline.ndim == 2
                  and f0.centerline.shape[1] == 2
              ))

# ─────────────────────────────────────────────────────────────────────────────
# Group 5 — _extract_fiber_segments_from_curvealign
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 5: _extract_fiber_segments_from_curvealign ─────────────────")

segs = None
if o2d:
    segs = check("returns a DataFrame",
        lambda: ex._extract_fiber_segments_from_curvealign(
            o2d.orientation_map, o2d.alignment_map,
            pixel_size=0.5, subsample=4,
        ))
    if segs is not None:
        check("required columns present",
              lambda: ok({'segment_id', 'position_x', 'position_y',
                          'orientation', 'local_alignment_intrinsic'}
                         .issubset(segs.columns)))
        check("non-empty DataFrame",
              lambda: ok(len(segs) > 0))

# ─────────────────────────────────────────────────────────────────────────────
# Group 6 — _compute_fiber_segment_metrics (no tumors)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 6: _compute_fiber_segment_metrics ───────────────────────────")

fm_seg = None
if segs is not None and len(segs) > 3:
    fm_seg = check("returns a DataFrame",
        lambda: ex._compute_fiber_segment_metrics(
            segs.head(15), tumor_regions=[],
            k_neighbors=3, bbox_size=50.0,
            tacs_zone_width=100.0, pixel_size=0.5,
        ))
    if fm_seg is not None:
        check("has local_alignment, local_density, tacs_type columns",
              lambda: ok({'local_alignment', 'local_density', 'tacs_type'}
                         .issubset(fm_seg.columns)))
        check("local_alignment in [0,1]",
              lambda: ok(
                  ((fm_seg['local_alignment'] >= 0)
                   & (fm_seg['local_alignment'] <= 1)).all()
              ))
        check("tacs_type is NaN when no tumor regions supplied",
              lambda: ok(fm_seg['tacs_type'].isna().all()))

# ─────────────────────────────────────────────────────────────────────────────
# Group 7 — _compute_individual_fiber_metrics (no tumors)
# Uses FiberObject instances (post extraction_analyzer conversion).
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 7: _compute_individual_fiber_metrics ───────────────────────")

fm_fib = None
if ctr and ctr.fibers:
    fm_fib = check("returns a DataFrame",
        lambda: ex._compute_individual_fiber_metrics(
            ctr.fibers[:10], tumor_regions=[],
            k_neighbors=3, bbox_size=50.0,
            tacs_zone_width=100.0, straightness_threshold=0.7,
            pixel_size=0.5,
        ))
    if fm_fib is not None:
        check("required columns present",
              lambda: ok({'length', 'width', 'straightness',
                          'local_fiber_density'}.issubset(fm_fib.columns)))
        check("width values non-negative (DT-based)",
              lambda: ok((fm_fib['width'].dropna() >= 0).all()))
        check("row count matches input",
              lambda: ok(len(fm_fib) == len(ctr.fibers[:10])))
        check("tacs_type is NaN when no tumor regions supplied",
              lambda: ok(fm_fib['tacs_type'].isna().all()))

# ─────────────────────────────────────────────────────────────────────────────
# Group 8 — 3-D CurveAlign orientation analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 8: 3-D CurveAlign orientation ──────────────────────────────")

ca_3d = CurveAlignParams(
    pixel_size=0.5, window_size=16, overlap=0.5,
    curvelet_levels=2, curvelet_angles=4,
    compute_coherency=True, compute_energy=False,
    keep_values=['angles', 'alignment'],
    compute_statistics=True,
)
a3 = FiberAnalyzer()

o3d = check("analyze_orientation_3d returns CurveAlignResult",
    lambda: a3.analyze_orientation_3d(VOL, ca_3d, image_id="smoke_3d"))
if o3d:
    check("orientation_map shape == (Z,H,W)",
          lambda: ok(o3d.orientation_map.shape == (6, 64, 64)))
    check("mean_orientation is float",
          lambda: ok(isinstance(o3d.mean_orientation, float)))
    check("alignment_score in [0,1]",
          lambda: ok(0.0 <= o3d.alignment_score <= 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# Group 9 — 3-D Skeleton fiber extraction (Lee algorithm)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 9: 3-D Skeleton extraction ─────────────────────────────────")

sk3d = check("extract_fibers_3d (SkeletonParams, Lee) returns SkeletonResult",
    lambda: a3.extract_fibers_3d(VOL, sk_p, image_id="smoke_3d"))
if sk3d:
    check("skeleton_mask shape == (Z,H,W)",
          lambda: ok(sk3d.skeleton_mask.shape == (6, 64, 64)))
    check("fibers is a list",
          lambda: ok(isinstance(sk3d.fibers, list)))
    if sk3d.fibers:
        cl = sk3d.fibers[0].centerline
        check("3-D centerline is (N,3)",
              lambda: ok(
                  cl is not None and cl.ndim == 2 and cl.shape[1] == 3
              ))

# ─────────────────────────────────────────────────────────────────────────────
# Group 10 — 3-D CT-FIRE raises the expected exception (C++ absent)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 10: 3-D CT-FIRE raises correctly ────────────────────────────")

try:
    a3.extract_fibers_3d(
        VOL, CTFireParams(ctfire_threshold=0.05, z_spacing=1.0)
    )
    FAIL.append("3-D CT-FIRE should raise but did not")
    print("  FAIL  3-D CT-FIRE should have raised ValueError/NotImplementedError")
except (ValueError, NotImplementedError):
    PASS.append("3-D CT-FIRE raises correctly")
    print("  PASS  3-D CT-FIRE raises correctly (C++ extension absent)")

# ─────────────────────────────────────────────────────────────────────────────
# Group 11 — TACS classification helpers
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Group 11: TACS classification ────────────────────────────────────")

check("TACS-3: perpendicular fiber (75°, straightness 0.85, dist 50µm)",
      lambda: ok(classify_fiber_tacs(75.0, 0.85, 50.0) == 'TACS-3'))
check("TACS-2: parallel fiber (15°, straightness 0.85, dist 50µm)",
      lambda: ok(classify_fiber_tacs(15.0, 0.85, 50.0) == 'TACS-2'))
check("TACS-1: curly fiber (45°, straightness 0.3, dist 50µm)",
      lambda: ok(classify_fiber_tacs(45.0, 0.3, 50.0) == 'TACS-1'))
check("None: fiber outside 100µm TACS zone",
      lambda: ok(classify_fiber_tacs(75.0, 0.85, 200.0) is None))
check("classify_fiber_segment_tacs_like: TACS-3-like (75°, dist 50µm)",
      lambda: ok(classify_fiber_segment_tacs_like(75.0, 50.0) == 'TACS-3-like'))
check("compute_angle_to_boundary_normal returns value in [0,90]",
      lambda: ok(0 <= compute_angle_to_boundary_normal(45.0, (0, 0), (10, 0)) <= 90))
check("get_tacs_color returns 3-element RGB tuple",
      lambda: ok(len(get_tacs_color('TACS-3')) == 3))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
total = len(PASS) + len(FAIL)
print(f"\n{'=' * 62}")
print(f"  PASSED : {len(PASS)}/{total}")
if FAIL:
    print(f"  FAILED : {len(FAIL)}")
    for f in FAIL:
        print(f"    ✗  {f}")
else:
    print("  All checks passed.")
print("=" * 62)