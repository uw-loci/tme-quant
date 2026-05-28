# CTFire Python Conversion — Developer Notes

Critical conventions and hard-won lessons for working on `src/ctfire_py/`.

---

## Vertex Index Convention

**Policy: the entire codebase uses 0-based vertex indices. No module should subtract 1
before indexing into X, R, or V. If an upstream stage appears to output 1-based indices,
fix that stage — do not patch the callers.**

**Vertex `v` is stored at `X[v]` — direct 0-based numpy index, no subtraction.**

The C++ backend (`extend_xlink_native`, `fiberproc_native`) uses 0-based vertex indices throughout, exactly matching numpy array positions. `extend_xlink` outputs vertices starting from index 0 with `min_v=0`.

After the Python `trimxfv` compacts the array, new indices are also 0-based (position in the sorted `vertices_used` list). All Python code that accesses vertex coordinates must use `X[v]`, not `X[v-1]`.

### Modules that access vertex coordinates

| File | Correct pattern |
|---|---|
| `utils/trimxfv.py` | `X[list(vertices_used)]` — direct index |
| `fiber_processing/curvealign_filter.py` | `v1_idx = int(v)` (no `- 1`) |
| `fiber_processing/fiber2beam.py` | new vertex start = `N_verts` (not `N_verts + 1`) |
| `test_fire_2d.py::plot_fiber_overlay` | `X_arr[v0]` (no `- 1`) |

### All modules now use correct 0-based indexing (as of b5f81f3)

Every module that reads vertex coordinates from X, R, or V uses direct 0-based access —
no `v - 1` offsets remain anywhere. See the incident record below.

---

## `trimxfv.py` — How It Works

`trimxfv` compacts X/F/V after filtering removes some fibers:

1. Collect `vertices_used = sorted(unique v from all remaining fibers)`
2. `X_trimmed = X[list(vertices_used), :]` — picks rows by direct index
3. `old_to_new[old_v] = new_idx` — 0-based renumbering
4. Fiber vertex lists are remapped through `old_to_new`
5. `R_trimmed = R[list(vertices_used)]` — same direct indexing

**The historical bug (now fixed):** the code previously used `X[[v-1 for v in vertices_used]]`. For *consecutive* vertices this accidentally worked (the off-by-one was masked by the phantom at position 0). For *non-consecutive* vertices (e.g., after aggressive filtering), it loaded the wrong row for each vertex, causing 200–600 pixel jumps in fiber centerlines.

---

## Coordinate Layout in Vertex Arrays

`X[:, 0]` = **row** (image y), `X[:, 1]` = **col** (image x), `X[:, 2]` = channel (always 1 for 2D).

This is confirmed empirically: plotting nucleation points as `scatter(x=xlink[:,1], y=xlink[:,0])` (matplotlib convention) places dots on the actual fiber structures. Swapping gives misaligned dots.

---

## Pipeline Data Flow

```
extend_xlink  →  Xz/Fz   (0-based, min_v=0, max_v=len-1)
      ↓
check_danglers → trimxfv  →  Xz2/Fz2  (0-based, compacted)
      ↓
process_fibers (C++)       →  Xa/Fa    (0-based, C++ never renumbers)
      ↓
fiberbreak    → trimxfv   →  Xc/Fc    (0-based, compacted)
      ↓
curvealign_filter → trimxfv →  Xf/Ff  (0-based, final filtered set)
```

`Xf`/`Ff` are the correct inputs for the fiber overlay.

---

## Incident Record: The 91b7218 Wrong-Direction Fix

### What happened

Commit 91b7218 observed that six downstream modules were crashing or producing wrong
coordinates. The diagnosis was that `trimxfv` was outputting 1-based indices. Instead of
fixing trimxfv, the commit added `v - 1` before every array lookup in all six modules:
`fiber_angles.py`, `fiber_stats.py`, `network_stats.py`, `beamproc.py`,
`curvealign_filter.py`, `fiber2beam.py`.

Commit f1191cf (same day, later) correctly fixed trimxfv (the actual source of the
wrong indices) and simultaneously removed the `v - 1` patch from two of those six files
(`curvealign_filter.py` and `fiber2beam.py`). The other four files were not cleaned up
at that time, leaving them with an incorrect `-1` offset relative to the now-fixed pipeline.

Commit b5f81f3 (Dong Woo Lee, next day) completed the cleanup by removing the remaining
`v - 1` offsets from the other four files. The pipeline is now fully 0-based.

### Why this should not have happened

The six-file patch in 91b7218 adapted callers to a broken upstream instead of fixing the
source. Any time a pipeline stage outputs unexpected index values, the correct response is
to fix that stage — not to add offset arithmetic in downstream callers. Patching callers
instead of the source:

- hides the true bug behind compensating hacks
- creates a mixed state where some callers expect corrected behavior and others expect broken behavior
- requires a second cleanup pass (b5f81f3) after the real fix (f1191cf)

### Rule for future changes

Before adding `v - 1` (or any index offset) to array lookups:

1. Verify what the upstream stage actually outputs — read its code, not its commit message.
2. If the upstream is wrong, fix it there.
3. Do not add compensating offsets in downstream callers.

---

## Fiber Overlay (`plot_fiber_overlay` in `test_fire_2d.py`)

- Background: normalize image to `[0, 1]` with `img / img.max()` (do **not** use histogram equalization — it makes the background look unrealistic/saturated).
- Centerlines: 1-pixel-thick Bresenham lines via `skimage.draw.line`.
- Colors: HSV colormap cycling over `n_fibers`.
- Access pattern: `X_arr[v, 0]` = row, `X_arr[v, 1]` = col — no index offset.

---

## C++ Backend Notes

- **`findlocmax_native`**: outputs `xlink[:,0]` = row, `xlink[:,1]` = col (verified empirically). The column naming in the C++ source (`i`=col, `j`=row) is misleading because the flat array is passed row-major from Python, making `i` iterate rows.
- **`fiberproc_native / trimxfv_cpp`**: explicitly does **not** renumber vertices. Unused vertex slots remain; their `V[v].f` is empty.
- **`extend_xlink_native`**: the 2D constructor is called as `ExtendXLink(sizey=J=height, sizez=I=width, ...)`. Image is accessed row-major (`image[p[0]*sizex + p[1]]`).

---

## Test Images

- `tests/test_images/real1.tif` — 512×512 grayscale, range [0, 255]. Bright pixel peak at (row=47, col=481).
- Synthetic image: generated in `test_fire_2d.py::create_synthetic_fiber_image`.

## thresh_im2 Masking — How It Works in MATLAB and Python

MATLAB ctFIRE applies `thresh_im2` to the **original image** (not the reconstruction) to create a
binary foreground mask, then multiplies the reconstruction by that mask before passing to `fire_2D_ang1`:

```matlab
mask_ori = original_Image > p1.thresh_im2;  % mask from original image
CTr = CTr .* mask_ori;                       % apply to reconstruction
im3(1,:,:) = CTr;
data = fire_2D_ang1(p, im3, 0);             % pass thresh_im2=0
```

Python `ct_fire.py` mirrors this exactly:

```python
mask_ori = img > ctfire_params["value"]["thresh_im2"]  # from original (normalized [0,255])
reconstructed_ct = ct_reconstruction(...)
reconstructed_ct = reconstructed_ct * mask_ori
fire2d_params["thresh_im2"] = 0  # prevent double-thresholding inside fire_2d_angle
```

**Why the original image?** The original image clearly delineates fiber (bright) vs background (dark)
in [0, 255] range. The reconstruction can have negative values and large-magnitude artifacts outside fiber
regions; the original-image mask suppresses those without requiring the threshold to know the reconstruction's
unusual scale (typically [-132, 237] for real images).

**Do not** build the mask from the reconstruction — this was tried and leads to very few surviving pixels
(~13% for thresh_im2=5 on raw reconstruction), severely under-detecting fibers.

## Parameters

`thresh_im2=5` gives dense extraction (142 filtered fibers for real1.tif). `thresh_im2=50` is more selective (~25–73 fibers). Very low thresholds include background noise as fiber-like structures.

### Fiber length thresholds

Three separate length thresholds operate at different pipeline stages:

| Name | Default | Stage | Can change without re-running? |
|---|---|---|---|
| `thresh_flen` | 15 px | Inside `fiberremove` (C++): prunes short dangling segments during network construction | No |
| `min_fiber_length` | 30 px | Inside `fire_2d_angle` CurveAlign filter: removes short fibers from `Xf`/`Ff` | No |
| `LL1` | 30 px | CT-FIRE post-processing: selects fibers for CSV output and downstream analysis | **Yes** |

`LL1` is a **top-level key** in `ctfire_params` (not inside `ctfire_params["value"]`).  It can be tuned without re-running the expensive fiber extraction pipeline.

`thresh_flen` and `min_fiber_length` live inside `ctfire_params["value"]` and are passed directly to `fire_2d_angle`.  Changing them requires a full pipeline re-run.

Parameters are saved to `{stem}_params.json` on each run (when `save_images=True`) and can be reloaded with `load_ctfire_params(path)` from `ctfire_py`.

---

## CT Reconstruction Test — Why It Exists

`test_ct_reconstruction_matches_matlab` in `tests/test_ct_fire.py` verifies that the
Python curvelet reconstruction step produces the same intermediate image as MATLAB ctFIRE
**before** fiber extraction begins.

### Motivation

The full CT-FIRE pipeline has two stages: (1) curvelet reconstruction, then (2)
`fire_2d_angle` fiber extraction. Because `fire_2d_angle` already passes its own MATLAB
comparison tests independently, any divergence in the end-to-end `test_ct_fire_*` tests
must come from stage (1). This test isolates stage (1) so that a reconstruction bug cannot
hide behind fiber-extraction noise.

### The `num_scales` / `SS` mapping

`num_scales` in `test_cases_ct_fire.json` is passed as `specific_scales` to
`ct_reconstruction`, which translates directly to `SS` in the MATLAB ctFIRE formula:

```
MATLAB: s = length(C)-SS : length(C)-1   (1-based, excludes finest scale)
Python: range(num_scales - specific_scales - 1, num_scales - 1)  (0-based)
```

For a 512×512 image both Python and MATLAB produce 6 total curvelet scales.
With `SS=3` / `num_scales=3` the selected 1-based scales are **[3, 4, 5]**:

| scale (1-based) | wedges | approx size |
|---|---|---|
| 3 | 32 | 23×35 |
| 4 | 32 | 43×67 |
| 5 | 64 | 43×131 |

Scale 1 (coarse 21×21) and scale 6 (full-resolution 512×512) are always zeroed out.

**`num_scales=4` would be MATLAB `SS=4`, selecting scales [2,3,4,5] — one extra coarser
level. This was the source of the soft-IoU failures before the fix.**
Verified: Python `num_scales=3` achieves Pearson r > 0.999 against the MATLAB reference
image; `num_scales=4` gives only r ≈ 0.88.

### Reference file naming convention

```
recon_img_{image_base}_SS{num_scales}_TH{int(coefficient_percentile*10):02d}.mat
```

Examples: `recon_img_real1_SS3_TH02.mat`, `recon_img_syn2_SS3_TH02.mat`.

To generate from MATLAB ctFIRE (save just after the curvelet reconstruction step,
before `fire_2D_ang1` is called):

```matlab
recon_img = CTRimage;
save('recon_img_real1_SS3_TH02.mat', 'recon_img', '-v7.3');
```

Files live in `tests/test_results/ct_fire_test_files/`. The test skips automatically
if the file is absent, so missing references never block CI.

---

## Running the GUI (`tests/example_process_image.py`) from WSL

### Problem: cv2 import fails with missing `g_pointer_bit_unlock_and_set`

`opencv 4.10.0` (conda-forge Qt5 build) loads `libgobject` at runtime. When
`LD_LIBRARY_PATH` is empty the system's older `libgobject` (`/usr/lib/x86_64-linux-gnu/`)
loads first and lacks the glib 2.79+ symbol. The conda env has glib 2.82.2 which provides
it, but it is never reached.

**Root cause**: `conda activate` in `bash -c` (non-interactive, no TTY) with conda 23.5.2
does **not** source `activate.d/env_vars.sh`, so `LD_LIBRARY_PATH` is never set.

### Fix

Export `LD_LIBRARY_PATH` explicitly before calling python:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curvelops03a
export LD_LIBRARY_PATH=/home/yuming/miniconda3/envs/curvelops03a/lib
cd /mnt/h/GitHub.06.2022/tmequant_ctfire/tme-quant
python tests/example_process_image.py
```

### Qt GUI requires a real display

The script opens a `QMainWindow`. It **must** be run directly in a WSL terminal (with
WSLg providing `DISPLAY`). Running via `wsl -e bash -c "..."` from PowerShell has no
display and the process exits silently with code 1.

### Correct workflow

1. Open a WSL terminal (e.g. Windows Terminal → Ubuntu tab, or wt with WSL profile).
2. Run the four commands above.
3. The GUI window opens. Set **Background Threshold (thresh_im2)** in the "CT-FIRE
   Parameters" group; default is `5` (fluorescence), use `98` for bright-field images.

### `run_example()` (non-GUI headless path)

`run_example()` inside the script uses `thresh_im2=98` by default (hardcoded). To invoke
it without the GUI, comment out `app.exec()` and call `run_example()` directly, or run
from a headless environment where `QApplication` can still be constructed (e.g., with
`QT_QPA_PLATFORM=offscreen`).
