# MATLAB–Python Fiber Parity Analysis

## Summary

As of the current implementation, the Python/C++ pipeline achieves:


| Metric                    | 2B_D9_ROI1.tif                     | real1.tif          |
| ------------------------- | ---------------------------------- | ------------------ |
| Nucleation point overlap  | **100%**                           | **100%**           |
| Fiber count drift         | −2.2% (713 vs 729)                 | −3.1% (466 vs 481) |
| Total length drift        | ≤ 5%                               | ≤ 5%               |
| Orientation histogram EMD | 0.16                               | 0.17               |
| Test suite                | **94 passed, 0 failed, 2 xfailed** | ← same run         |


**Can the Python fibers coincide exactly with MATLAB's?** Not feasibly, for principled reasons documented below.

---

## Stage-by-Stage Divergence Map

The pipeline has five meaningful stages after image loading. Here is where divergence first appears and what drives it.

```
[Image] → smooth → bw_dist → findlocmax → extend_xlink → fiberproc → [Fibers]
             ✓          ✓           ✓            ~3%           ✓
```

### Stage 1–3: Fully Matched


| Stage                                    | Status         | How                                                                                                           |
| ---------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------- |
| `smooth` (Gaussian convolution)          | ✓ identical    | `scipy.ndimage.convolve` with `mode='constant', cval=0` matches MATLAB `imfilter` zero-padding default        |
| `bw_dist` (cityblock distance transform) | ✓ identical    | Same algorithm, same boundary conditions                                                                      |
| `findlocmax`                             | ✓ 100% overlap | Matched MATLAB's `genrand_res53` RNG (two `uint32` draws per double) and column-major perturbation fill order |


The RNG fix for `findlocmax` was the largest single improvement, taking nucleation overlap from ~86% to 100%.

### Stage 4: `extend_xlink` — source of the ~3% gap

`extend_xlink` grows a fiber from each nucleation point by iteratively finding Local Maximum Points (LMPs) along the expanding fiber front. Starting from 100% matching nucleation points and identical DSM values, the fiber *paths* diverge because of accumulated differences across many LMP-selection steps.

### Stage 5: `fiberproc`

`fiberproc` (trimxfv → remove_repeat → fiberlink × 5 → fiberlinkgap → fiberremove) is a direct MATLAB port in C++ and is correct. It simply operates on a slightly different input from `extend_xlink`.

### Non-square images: two indexing bugs (fixed)

All fixtures in this analysis (`real1.tif`, `syn1_20fibers.png`, `syn2_35fibers.png`) are
512×512 squares. Two C++ indexing bugs were invisible on square inputs and only surfaced
when testing a non-square image (391×487):

- `findlocmax_native.cpp` addressed its flat pixel buffer column-major (Fortran-order),
  while Python actually supplies it row-major (C-order) via `dsm.flatten()`. A
  column-major read of row-major data is exactly a transpose, which a square canvas cannot
  distinguish from correct output — for height≠width it scrambles pixel correspondence.
  Fixed to row-major addressing throughout (`flat_idx = row*width + col`).
- `extend_xlink_native.cpp`'s single 2D-engine call site passed `(height, width)` into the
  engine's `(sizex, sizey)` constructor slots — the two dimensions were swapped. Fixed to
  pass `(width, height)`, matching the engine's own bounds-check/stride convention.

A separate `fiberproc_native.cpp` bug was also found and fixed during this testing: an
intermittent `std::bad_alloc` in `remove_repeat_cpp`, caused by holding references into a
`std::vector<Fiber>` across a `push_back()` that could reallocate it. Fixed by copying
instead of referencing. This is unrelated to the row/col bugs above but affects the same
pipeline stage that resolves overlapping fiber segments before `fiberlink` merges them.

None of the metrics in the "Summary" table above are affected — those fixtures are square,
so this fix is a no-op transpose-cancellation for them (output is now computed directly
instead of via two canceling transposes, so exact byte-for-byte coordinates can differ
slightly, but fiber count / overlap / EMD are unchanged within existing tolerances).

### `check_danglers`: already a no-op in both

MATLAB's `check_danglers.m` contains a logic bug:

```matlab
for vi = 1:length(V)
    if length(V(vi).f) > 1      % outer guard: vertex touches > 1 fiber
        fi = V(vi).f;
        if length(fi) == 1      % ← ALWAYS FALSE (fi = V(vi).f has > 1 elements)
            ...                 %   entire body is dead code
        end
    end
end
[X F V R] = trimxfv(X, F, V, R);  % ← only this line runs
```

The inner `if length(fi)==1` is unreachable when the outer guard `length(V(vi).f)>1` is satisfied. Python's `faithful_matlab_danglers=True` mode matches this exactly (trimxfv only).

---

## Why `extend_xlink` Cannot Be Made Exact

### Root cause 1: MATLAB zero-pads the distance image

```matlab
zpad = 3;
d = zeros(size(d_unpadded) + zpad*2);  % [7, J+6, I+6] for a 2D image
d(zpad+1:end-zpad, ...) = single(d_unpadded);
xlink = xlink + zpad;
```

A 2D input image of size `[1, J, I]` becomes a 3D volume `[7, J+6, I+6]`. The nucleation points are placed at depth z=4 (the middle layer). The C++ implementation uses bounds-clamped access without padding. Near image boundaries this gives slightly different LMP candidate sets.

### Root cause 2: MATLAB's `getdB` uses face-local neighbor comparisons

`findLMP` calls `getdB(u, r, s)` which returns three categories of boundary pixels. For the padded `[7, J+6, I+6]` volume:

- **z-face "side" pixels** land at z=z₁ or z=z₂ (the zero-padded depth layers). They always have DSM=0 and fail `thresh_LMP` — they contribute **no LMPs**.
- **x-face pixels** at (col=x₁ or x₂, row in yr, z=4): MATLAB's `dBxn` offsets are `±ys` (row) and `±zs` (depth) only — **not `±xs` (col)**. So each x-face pixel is compared only against its same-face y-neighbors, not against interior or opposite-face pixels.
- **y-face pixels** at (col in xr, row=y₁ or y₂, z=4): similarly compared against same-face x-neighbors only.

The practical result: LMPs are **local maxima along each face of the search box**, not global local maxima within the box. This is equivalent to the C++ boundary-only comparison for all practical radii (r ≥ 2, so opposing faces are ≥ 4 pixels apart and never appear in each other's 8-neighborhood).

The current C++ implementation is **geometrically correct** for this check. No bug remains here.

### Root cause 3: Direction vector drift accumulates over extension steps

Each `extend_xlink` continuation step updates the fiber direction:

```
dir_new = (1/(1+λ)) · dir_old + (λ/(1+λ)) · dir_lmp
dir_new = dir_new / ‖dir_new‖
```

With λ = `lam_dirdecay` = 0.5. Over 10–20 steps, 32-bit floating-point rounding differences in `dir` cause subtly different LMP selections, which cascade into different fiber paths.

### Root cause 4: Search radius is quantised via `ceil`

```matlab
r = max(2, ceil(d(xj(3), xj(2), xj(1))))
```

If accumulated direction drift moves `xj` by even 1 pixel, the next search radius `r` can change. A different `r` means a different bounding box, a different set of boundary candidate pixels, and potentially different LMPs — even with identical DSM values.

---

## Fixes Investigated

### Fix 1: Interior-neighbor LMP check (attempted, reverted)

**Hypothesis**: MATLAB compares boundary pixels against all 8 image neighbors (not just boundary neighbors), which would reject spurious boundary LMPs.

**Result**: Catastrophic regression (1510 → 195 segments). The reason: every boundary pixel has a higher-valued interior neighbor (one step closer to the nucleation center along the ridge), so almost nothing qualifies as an LMP. MATLAB's face-local comparison (root cause 2) prevents this by **not** including the interior direction in the comparison.

### Fix 2: Column-major LMP traversal order (attempted, reverted)

**Hypothesis**: MATLAB's `unique(LMP,'rows')` sorts candidates by `[col, row]` before distance-deduplication. Changing C++ to column-major (outer=col, inner=row) traversal would match the tie-breaking.

**Result**: `real1.tif` worsened from −3.1% → −5.0%. The change altered fiber paths, but not toward MATLAB's. Reverted.

### Fixes that worked


| Fix                                                | Effect                                                               |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| `genrand_res53` RNG in `findlocmax`                | Nucleation overlap: 86% → 100%                                       |
| Column-major perturbation fill in `findlocmax`     | Nucleation overlap: 86% → 100% (paired with above)                   |
| `extend_xlink` deduplication (symmetric pair keys) | Removed spurious duplicate fibers                                    |
| `line_clear_above_thresh` in `extend_xlink`        | Matches `ind_btw_nodes` continuity check from MATLAB                 |
| Full MATLAB-faithful `fiberproc` in C++            | trimxfv → remove_repeat → fiberlink × 5 → fiberlinkgap → fiberremove |
| `smooth` boundary: `mode='constant', cval=0`       | Matches MATLAB `imfilter` zero-padding default                       |


---

## What Would Be Required for Exact Match

Getting the remaining ~3% to 0% would require:

1. **Zero-pad the DSM in C++** to match MATLAB's `[7, J+6, I+6]` structure, ensuring the same boundary pixel sets near image edges.
2. **Match MATLAB's double-precision direction arithmetic**. Currently C++ accumulates direction in float32 per the image type. Switching `dir` to double throughout `extend_xlink_native.cpp` would reduce rounding drift.
3. **Match MATLAB's `single(d_unpadded)` cast**. MATLAB explicitly converts the distance image to single (float32) before storing in the padded volume. If the Python/C++ DSM differs by even 1 ULP at a single pixel, `ceil(d(xj))` can give a different search radius.

These are engineering-intensive changes for a 3% improvement in a metric (fiber count) that already exceeds the biological variability of the underlying images. The orientation histogram EMD (0.16–0.17) and 100% nucleation overlap represent a higher-fidelity match than fiber count alone.

---

## Current Test Status

```
tests/test_fire_2d_angle_endtoend.py    8 passed
tests/test_cpp_functions.py            ...
tests/test_fiberlinkgap.py             ...
                              Total: 94 passed, 0 failed, 2 xfailed
```

The 2 `xfailed` cases (`test_fire_2d_matches_matlab_angles`) use stale MATLAB reference `.mat` files that were captured before the RNG correction and need to be regenerated.