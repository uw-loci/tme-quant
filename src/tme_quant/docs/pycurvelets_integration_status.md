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

## Remaining (Future Batches)

---

#### `new_curv` → `extract_curvelet_fiber_candidates`
- **Source:** `src/pycurvelets/new_curv.py`
- **Planned target:** `fiber_analysis/utils/curvelet_utils.py`
- **Original:** `new_curv(img, curve_cp: CurveletControlParameters)`
- **Proposed:** `extract_curvelet_fiber_candidates(image, keep=0.05, scale=1, radius=4.0)`
- **Notes:** `CurveletControlParameters` fields become explicit params; returns `(in_curves: DataFrame[center_row, center_col, angle], coefficients, inc)`; must include threshold selection, radius-based grouping, `fix_angle` per group, and edge trimming — none of which exist in `curvelet_transform_2d`

---

#### `get_ct` → `build_fiber_structure_from_curvelets`
- **Source:** `src/pycurvelets/get_ct.py`
- **Planned target:** `fiber_analysis/utils/fiber_dataframe_utils.py`
- **Original:** `get_ct(img, curve_cp: CurveletControlParameters, feature_cp: FeatureControlParameters)`
- **Proposed:** `build_fiber_structure_from_curvelets(image, curvelet_params, feature_params)`
- **Notes:** Orchestrates `extract_curvelet_fiber_candidates` + `compute_fiber_density_and_alignment`; returns `(fiber_structure_df, density_df, alignment_df, coefficients)`

---

#### `get_tif_boundary` → `extract_tif_boundary`
- **Source:** `src/pycurvelets/get_tif_boundary.py`
- **Planned target:** `fiber_analysis/utils/boundary_tif_utils.py` (new file)
- **Original:** `get_tif_boundary(coordinates, img, obj, dist_thresh, min_dist)`
- **Proposed:** `extract_tif_boundary(coordinates, img, obj, dist_thresh, min_dist)`
- **Notes:** TIF boundary coordinate extraction + relative angle computation; also contains `get_relative_angle` and `get_points_on_line` helpers

---

#### `draw_curvs` → `draw_utils` (visualization)
- **Source:** `src/pycurvelets/utils/visualization/draw_curvs.py`
- **Planned target:** `fiber_analysis/visualization/draw_utils.py` (new file)
- **Original:** `draw_curvs(fiber_data, ax, length, color_flag, angles, mark_size, line_width, boundary_measurement)`
- **Notes:** No Qt dependency; pure matplotlib

---

#### `draw_map` → `draw_utils` (visualization)
- **Source:** `src/pycurvelets/utils/visualization/draw_map.py`
- **Planned target:** `fiber_analysis/visualization/draw_utils.py` (same new file as `draw_curvs`)
- **Original:** `draw_map(fiber_structure, angles, img, boundary_measurement, map_params)`
- **Notes:** Uses `scipy.ndimage.gaussian_filter`; no Qt dependency

---

#### `format_df_to_excel` → `export_dataframe_to_excel`
- **Source:** `src/pycurvelets/utils/misc/format_df_to_excel.py`
- **Planned target:** `fiber_analysis/io.py` or `tme_analysis/io.py`
- **Original:** `format_df_to_excel(df, filename, sheet_name='Sheet1', mode='w')`
- **Proposed:** `export_dataframe_to_excel(df, filename, sheet_name='Sheet1', mode='w')`
- **Notes:** Thin `openpyxl` wrapper; check which `io.py` already has similar functionality before placing

---

#### `flatten_numeric`
- **Source:** `src/pycurvelets/utils/math/flatten_numeric.py`
- **Planned target:** `fiber_analysis/utils/fiber_dataframe_utils.py` (add to existing module)
- **Original / Proposed:** `flatten_numeric(series)` — unchanged
- **Notes:** One-liner utility; no signature changes needed

---

#### Exception classes
- **Source:** `src/pycurvelets/models/models.py`
- **Planned target:** `fiber_analysis/exceptions.py` (new) or `core/exceptions.py` (new)
- **Classes:** `FiberAnalysisError`, `ROIProcessingError`, `BoundaryAnalysisError`, `FeatureExtractionError`, `ImageProcessingError`
- **Notes:** Decide scope first — fiber-only → `fiber_analysis/`; library-wide → `core/`

---

#### `process_image` — do not integrate directly
- **Source:** `src/pycurvelets/process_image.py`
- **Notes:** 1640-line orchestrator tightly coupled to file I/O and optional GUI widgets. Decompose into sub-functions first; plan separately.

---

## Batch Protocol Summary

1. **Plan first** — present a plan and get approval before touching any file.
2. **Max 3–5 files** per batch (new + modified combined).
3. **`pytest tests/`** must be green before closing the batch.
4. **No overlap** — check the "Already Integrated" entries and `REFACTORING_GUIDE.md §5`
   before writing a new function.
5. **No Qt/napari** in `src/tme_quant/` — see `REFACTORING_GUIDE.md §2`.

Full rules: [`REFACTORING_GUIDE.md`](../REFACTORING_GUIDE.md)
