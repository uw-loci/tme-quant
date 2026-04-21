# pycurvelets → tme_quant Integration Status

This file is the **canonical cross-reference** for the ongoing integration of
MATLAB-converted Python functions from `src/pycurvelets/` into the core
`tme_quant` library.

**Update rule:** Update this file in the **same commit** that adds or modifies
an integrated function.  After each batch, also update the memory pointer at
`C:/Users/liu372/.claude/projects/.../memory/project_refactoring_status.md`
so future Claude Code sessions stay in sync.

Path conventions used below:
- **pycurvelets paths** are relative to the git root (`tme-quant/`), e.g.
  `src/pycurvelets/process_fibers.py`
- **tme_quant paths** are relative to the Python package root, e.g.
  `fiber_analysis/utils/fiber_dataframe_utils.py`
  which resolves to
  `src/tme_quant/src/tme_quant/fiber_analysis/utils/fiber_dataframe_utils.py`

---

## Already Integrated

---

#### `circ_r` → `_circ_r`
- **Source:** `src/pycurvelets/utils/math/circ_r.py`
- **Target:** `fiber_analysis/utils/geometry_utils.py`
- **Original:** `circ_r(alpha, w=None, d=0, axis=0)`
- **New:** `_circ_r(alpha, w=None, d=0.0)`
- **Changes:** Private helper (underscore prefix); `axis` param dropped (always 0); return type narrowed to `float`

---

#### `find_outline_slope` → `compute_boundary_tangent_angle`
- **Source:** `src/pycurvelets/utils/math/find_outline_slope.py`
- **Target:** `fiber_analysis/utils/geometry_utils.py`
- **Original:** `find_outline_slope(coords, idx, num=21)`
- **New:** `compute_boundary_tangent_angle(coords, idx, num=21)`
- **Changes:** Renamed for clarity; same signature and return (`float` degrees or `NaN`)

---

#### `find_connected_pts` → `_find_connected_pts`
- **Source:** `src/pycurvelets/utils/connectivity/find_connected_pts.py`
- **Target:** `fiber_analysis/utils/geometry_utils.py`
- **Original:** `find_connected_pts(boundary_coords, idx, num)`
- **New:** `_find_connected_pts(coords, idx, num)`
- **Changes:** Private helper; `boundary_coords` → `coords`; stuck-detection fixed (`nxt == cur` instead of dead `if idx is None`)

---

#### `get_first_neighbor` → `_get_first_neighbor`
- **Source:** `src/pycurvelets/utils/connectivity/get_first_neighbor.py`
- **Target:** `fiber_analysis/utils/geometry_utils.py`
- **Original:** `get_first_neighbor(mask, idx, visited, direction)`
- **New:** `_get_first_neighbor(coords, idx, visited, direction)`
- **Changes:** Private helper; `mask` (2-D bool array) → `coords` ((N,2) int array) — operates on coordinate list instead of image mask

---

#### `get_relative_angles` → `compute_relative_fiber_angles`
- **Source:** `src/pycurvelets/get_relative_angles.py`
- **Target:** `fiber_analysis/utils/geometry_utils.py`
- **Original:** `get_relative_angles(ROI, obj, angle_option=0, fig_flag=False)`
- **New:** `compute_relative_fiber_angles(obj_center, obj_angle, roi_coords, image_size=None, index2object=None, angle_option=0, dense_boundary=False)`
- **Changes:** ROI/obj dataclasses replaced by plain arrays + scalars; `fig_flag` dropped; `dense_boundary` flag added; output keys renamed: `angle_to_boundary_edge` → `angle_to_boundary_tangent`, `angle_to_boundary_center` → `angle_to_roi_orientation`

---

#### `create_3d_curvelet` → `curvelet_transform_3d`
- **Source:** `src/pycurvelets/pycurvelets3D.py`
- **Target:** `fiber_analysis/utils/curvelet_utils.py`
- **Original:** `create_3d_curvelet(folder_path, num_images, nb_scales, nb_angles)`
- **New:** `curvelet_transform_3d(image, n_levels=3, n_angles=8, use_matlab=False, use_curvelops=True)`
- **Changes:** File-based I/O replaced by in-memory `(Z,H,W)` ndarray; returns `(Z,H,W,n_angles)` energy array; **not a port of `new_curv`** — raw FDCT energy only, no thresholding/grouping

---

#### `new_curv` → `extract_curvelet_fiber_candidates`  *(Batch 3)*
- **Source:** `src/pycurvelets/new_curv.py`
- **Target:** `fiber_analysis/utils/curvelet_utils.py`
- **Original:** `new_curv(img, curve_cp: CurveletControlParameters)`
- **New:** `extract_curvelet_fiber_candidates(image, keep=0.05, scale=1, radius=4.0)`
- **Changes:**
  - `CurveletControlParameters.keep/scale/radius` → explicit plain params (per REFACTORING_GUIDE §6)
  - No fallback if `curvelops` absent — raises `ImportError` (approximate backends cannot produce fiber candidates)
  - `fix_angle` nested function extracted as module-private `_fix_angle(angles, inc)`
  - `ac=0` (wavelet mode) and `nbangles_coarse=16` preserved from original
  - Returns identical `(in_curves: DataFrame[center_row, center_col, angle], coefficients, inc)`
  - Also fixed `round_mlab` in `fiber_dataframe_utils.py` to handle numpy arrays (pycurvelets version handles them via `hasattr(__iter__)`; tme_quant previously only handled `list`)

---

#### `get_fire` → `fire_2d`
- **Source:** `src/pycurvelets/get_fire.py`
- **Target:** `fiber_analysis/utils/ctfire_utils.py`
- **Original:** `get_fire(img_name, fire_directory, fiber_mode, feature_cp)`
- **New:** `fire_2d(fiber_mask, image, pixel_size=1.0, ...)`
- **Changes:** File-based `.mat` loading replaced by in-memory mask + image arrays; C++/Python dual backend; returns list of `(N,3)` centerline arrays instead of DataFrame

---

#### `process_fibers` → `compute_fiber_density_and_alignment`  *(Batch 1, commit `e842df0`)*
- **Source:** `src/pycurvelets/process_fibers.py`
- **Target:** `fiber_analysis/utils/fiber_dataframe_utils.py`
- **Original:** `process_fibers(fiber_structure, feature_cp: FeatureControlParameters)`
- **New:** `compute_fiber_density_and_alignment(fiber_structure, params: FiberFeatureParams)`
- **Changes:** `FeatureControlParameters` → `FiberFeatureParams`; `center_1`/`center_2` column aliases accepted; same return type `(density_df, alignment_df)`

---

#### `FeatureControlParameters` → `FiberFeatureParams`  *(Batch 1, commit `e842df0`)*
- **Source:** `src/pycurvelets/models/models.py`
- **Target:** `fiber_analysis/config.py`
- **Original:** `FeatureControlParameters(minimum_nearest_fibers, minimum_box_size, fiber_midpoint_estimate)`
- **New:** `FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32, fiber_midpoint_estimate=1)`
- **Changes:** Renamed; defaults added; `to_dict()` method added

---

#### `get_alignment_to_roi` → `compute_fiber_alignment_to_roi`  *(Batch 2, commit `e842df0`)*
- **Source:** `src/pycurvelets/get_alignment_to_roi.py`
- **Target:** `tme_analysis/utils/alignment_utils.py`
- **Original:** `get_alignment_to_roi(roi_list: ROIList, fiber_structure, distance_threshold=None)`
- **New:** `compute_fiber_alignment_to_roi(roi_coords, img_height, img_width, fiber_structure, distance_threshold=None)`
- **Changes:**
  - `ROIList` dataclass replaced by `(N,2)` ndarray + explicit `img_height`/`img_width` args
  - Output columns renamed: `angle_to_boundary_edge` → `angle_to_boundary_tangent` (90°-complement applied), `angle_to_boundary_center` → `angle_to_roi_orientation`, `angle_to_center_line` → `angle_to_centers_line`
  - Coordinate mixing bug in pycurvelets `angle_to_centers_line` **not replicated**; uses correct formula from `get_relative_angles`
  - **Design limitation:** `roi_coords` must be a dense 8-connected pixel trace (e.g. CurveAlign output). Sparse polygon vertices yield `None` for `angle_to_boundary_tangent`. Use `compute_relative_fiber_angles(dense_boundary=False)` for sparse ROIs.

---

#### `get_tif_boundary` → `extract_tif_boundary`  *(Batch 5)*
- **Source:** `src/pycurvelets/get_tif_boundary.py`
- **Target:** `fiber_analysis/utils/boundary_tif_utils.py` (new file)
- **Original:** `get_tif_boundary(coordinates, img, obj, dist_thresh, min_dist)`
- **New:** `extract_tif_boundary(coordinates, img, fiber_df, dist_thresh, min_dist)`
- **Changes:**
  - `obj` → `fiber_df`; accepts `center_row/center_col` or `center_1/center_2` aliases
  - `coordinates` accepts dict of arrays (same as original) or a pre-stacked ndarray
  - Nested helpers extracted as private module-level functions:
    - `get_segment_pixels` → `_rasterize_line_segment` (uses `round_mlab` from `fiber_dataframe_utils`)
    - `get_points_on_line` → `_get_fiber_line_points`
    - `get_relative_angle` → `_compute_fiber_boundary_relative_angle`
      (uses `compute_boundary_tangent_angle` and `_circ_r` from `geometry_utils`)
  - **Boundary coords swap fix:** boundary CSV is in MATLAB `[col, row]` = `[x, y]` order;
    `boundary_point_row/col` output columns are now correctly labeled as actual row/col
    (the original pycurvelets code stored them in reversed order relative to column names)
  - `extension_point_distance` and `extension_point_angle` columns are **always NaN**;
    the original pycurvelets code computed but never stored these values (bug preserved)
  - `min_dist` falsy-check convention preserved (`if not min_dist:`)
  - Result column names renamed to snake_case tme_quant convention; 7-column structure unchanged
- **Pipeline affiliation:** CurveAlign boundary measurement pipeline; called by `process_image`

---

#### `get_ct` → `build_fiber_structure_from_curvelets`  *(Batch 4, commit `d9b20dc`)*
- **Source:** `src/pycurvelets/get_ct.py`
- **Target:** `fiber_analysis/utils/fiber_dataframe_utils.py`
- **Original:** `get_ct(img, curve_cp: CurveletControlParameters, feature_cp: FeatureControlParameters)`
- **New:** `build_fiber_structure_from_curvelets(image, keep=0.05, scale=1, radius=4.0, feature_params=None)`
- **Changes:**
  - `CurveletControlParameters.keep/scale/radius` → explicit plain params (same pattern as `extract_curvelet_fiber_candidates`)
  - `FeatureControlParameters` → `FiberFeatureParams`; `feature_params=None` defaults to `FiberFeatureParams()`
  - Edge case: empty `fiber_structure` returns `(empty_df, pd.DataFrame(), pd.DataFrame(), coefficients)` instead of bare `return fiber_structure`
  - `curvelet_coefficients` returned as 4th element (unchanged)
- **Pipeline affiliation:** **CurveAlign orientation pipeline** (population-level density and alignment statistics); not CT-FIRE individual fiber extraction

Also added in this batch:

#### `flatten_numeric` → `flatten_numeric`  *(Batch 4, commit `d9b20dc`)*
- **Source:** `src/pycurvelets/utils/math/flatten_numeric.py`
- **Target:** `fiber_analysis/utils/fiber_dataframe_utils.py`
- **Changes:** None — direct port, signature and logic unchanged

#### `draw_curvs` → `draw_curvs`  *(Batch 6)*
- **Source:** `src/pycurvelets/utils/visualization/draw_curvs.py`
- **Target:** `fiber_analysis/visualization/draw_utils.py` (new file + new sub-package)
- **Original:** `draw_curvs(fiber_data, ax, length, color_flag, angles, mark_size, line_width, boundary_measurement)`
- **New:** same signature (function name unchanged)
- **Changes:**
  - `center_1`/`center_2` column aliases now normalised **before both code paths** (original only normalised them for the `centers` array used in the non-boundary branch — the boundary branch would KeyError on alias input)
  - No pycurvelets imports to replace (function never used `circ_r`)
  - matplotlib imported at module top level (core dep in pyproject.toml line 12; Agg backend forced in tests for headless CI)

---

#### `process_image` → `curvealign_pipeline`  *(Batch 9B)*
- **Source:** `src/pycurvelets/process_image.py`
- **Target:** `tme_analysis/pipelines/curvealign_pipeline.py` (new file)
- **Original:** `process_image(image_params, fiber_params, output_params, boundary_params, advanced_options)`
- **New:** `curvealign_pipeline(image, fiber_structure=None, keep, scale, radius, feature_params, coordinates, boundary_img, distance_threshold, tif_boundary, exclude_fibers_in_mask, min_dist)`
- **Changes:**
  - All pycurvelets dataclass params replaced by plain args
  - File I/O, visualization calls, and `multiprocessing.Pool` stripped entirely
  - FIRE path replaced by optional pre-built `fiber_structure` arg (caller supplies CT-FIRE result)
  - `tif_boundary=1/2` (CSV) raises `NotImplementedError` (not yet ported)
  - `polygon2mask` bug fixed: called with `(height, width)` + `(row, col)` coords; original had transposed shape
  - Private helpers: `_analyze_global_boundary`, `_process_single_roi`, `_process_tif_rois`, `_concat_roi_df`, `_build_fiber_features_df`
  - Returns result dict (no file writes); `None` when no fibers detected

---

#### `extract_boundary_coords_from_mask`  *(Batch 9A)*
- **Source:** `src/pycurvelets/process_image.py` (lines 541–582)
- **Target:** `fiber_analysis/utils/boundary_tif_utils.py`
- **Original:** `extract_boundary_coords_from_mask(boundary_img)`
- **New:** same signature; return type `dict[str, ndarray]`
- **Changes:** `print()` calls dropped (library code must not write to stdout); logic preserved exactly

---

#### Exception classes → `core/exceptions.py`  *(Batch 8)*
- **Source:** `src/pycurvelets/models/models.py`
- **Target:** `core/exceptions.py` (new file); re-exported from `core/__init__.py` and `tme_quant/__init__.py`
- **Classes:** `FiberAnalysisError` (base), `ROIProcessingError`, `BoundaryAnalysisError`, `FeatureExtractionError`, `ImageProcessingError`
- **Changes:** Direct port — no logic changes; all classes are pass-through with no `__init__` or fields; scope widened to library-wide (callers can `except tme_quant.FiberAnalysisError`)

---

#### `format_df_to_excel` → `export_dataframe_to_excel`  *(Batch 7)*
- **Source:** `src/pycurvelets/utils/misc/format_df_to_excel.py`
- **Target:** `fiber_analysis/io.py` (module-level function)
- **Original:** `format_df_to_excel(df, filename, sheet_name='Sheet1', mode='w')`
- **New:** `export_dataframe_to_excel(df, filename, sheet_name='Sheet1', mode='w')`
- **Changes:** Renamed for clarity; `from openpyxl import load_workbook` moved inside function body (lazy) because import health check mocks `openpyxl`; all logic preserved exactly

---

#### `draw_map` → `draw_map`  *(Batch 6)*
- **Source:** `src/pycurvelets/utils/visualization/draw_map.py`
- **Target:** `fiber_analysis/visualization/draw_utils.py` (same file as `draw_curvs`)
- **Original:** `draw_map(fiber_structure, angles, img, boundary_measurement, map_params)`
- **New:** same signature (function name unchanged); returns `(rawmap: ndarray[float64], procmap: ndarray[uint8])`
- **Changes:**
  - `from pycurvelets.utils.math import circ_r` → `from ..utils.geometry_utils import _circ_r` (identical call site: `_circ_r(vals * np.pi / 127.5) * 255`)
  - All other logic, variable names, and `map_params` dict keys preserved exactly

---

## Remaining (Future Batches)

---

#### Visualization wrappers from `process_image`  *(Batch 9C, planned)*
- **Source:** `src/pycurvelets/process_image.py` — `generate_overlay`, `generate_heatmap`, `save_histogram`
- **Planned target:** `fiber_analysis/visualization/draw_utils.py` or new helpers alongside it
- **Notes:** These wrap already-ported `draw_curvs` / `draw_map`; strip file I/O (`plt.savefig`) and return figure/array objects instead

---

## Batch Protocol Summary

1. **Plan first** — present a plan and get approval before touching any file.
2. **Max 3–5 files** per batch (new + modified combined).
3. **`pytest tests/`** must be green before closing the batch.
4. **No overlap** — check the "Already Integrated" entries and `REFACTORING_GUIDE.md §5`
   before writing a new function.
5. **No Qt/napari** in `src/tme_quant/` — see `REFACTORING_GUIDE.md §2`.

Full rules: [`REFACTORING_GUIDE.md`](../REFACTORING_GUIDE.md)
- Type mapping (pycurvelets → tme_quant): §8.1
- Two-layer design (util + FiberObject method): §8.2
- Canonical DataFrame column names: §8.3
- Coordinate conventions: §8.4
- FiberObject attribute targets for angles: §8.5
- Multi-ROI → single-ROI pattern: §8.6
