# CTFire Python Conversion — Developer Notes

Critical conventions and hard-won lessons for working on `src/ctfire_py/`.

---

## Vertex Index Convention

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

### Modules with a pre-existing off-by-one (not yet fixed)

These modules subtract 1 before indexing — they were written assuming 1-based indices and have worked only because vertex 0 is a phantom in the `process_fibers` output. They affect statistics and angle calculations, not the overlay:

- `fiber_analysis/network_stats.py` — `v1_idx = v1 - 1`
- `fiber_analysis/fiber_angles.py` — `v1 = fv[0] - 1`

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

## Parameters

`thresh_im2=5` gives dense extraction (142 filtered fibers for real1.tif). `thresh_im2=50` is more selective (~25–73 fibers). Very low thresholds include background noise as fiber-like structures.
