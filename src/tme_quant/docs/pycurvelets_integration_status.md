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

| pycurvelets source → original signature | tme_quant target → new signature | Signature / naming changes |
|---|---|---|
| `src/pycurvelets/utils/math/circ_r.py`<br>`circ_r(alpha, w=None, d=0, axis=0)` | `fiber_analysis/utils/geometry_utils.py`<br>`_circ_r(alpha, w=None, d=0.0)` | Private helper (underscore prefix); `axis` param dropped (always 0); return type narrowed to `float` |
| `src/pycurvelets/utils/math/find_outline_slope.py`<br>`find_outline_slope(coords, idx, num=21)` | `fiber_analysis/utils/geometry_utils.py`<br>`compute_boundary_tangent_angle(coords, idx, num=21)` | Renamed for clarity; same signature and return (`float` degrees or `NaN`) |
| `src/pycurvelets/utils/connectivity/find_connected_pts.py`<br>`find_connected_pts(boundary_coords, idx, num)` | `fiber_analysis/utils/geometry_utils.py`<br>`_find_connected_pts(coords, idx, num)` | Private helper; parameter renamed `boundary_coords` → `coords` |
| `src/pycurvelets/utils/connectivity/get_first_neighbor.py`<br>`get_first_neighbor(mask, idx, visited, direction)` | `fiber_analysis/utils/geometry_utils.py`<br>`_get_first_neighbor(coords, idx, visited, direction)` | Private helper; `mask` (2-D bool array) → `coords` ((N,2) int array) — operates on coordinate list instead of image mask |
| `src/pycurvelets/get_relative_angles.py`<br>`get_relative_angles(ROI, obj, angle_option=0, fig_flag=False)` | `fiber_analysis/utils/geometry_utils.py`<br>`compute_relative_fiber_angles(obj_center, obj_angle, roi_coords, image_size=None, index2object=None, angle_option=0, dense_boundary=False)` | ROI/obj dataclasses replaced by plain arrays + scalars; `fig_flag` dropped; `dense_boundary` added; output keys renamed: `angle_to_boundary_edge` → `angle_to_boundary_tangent`, `angle_to_boundary_center` → `angle_to_roi_orientation` |
| `src/pycurvelets/new_curv.py`<br>`new_curv(img, curve_cp: CurveletControlParameters)` | `fiber_analysis/utils/curvelet_utils.py`<br>`curvelet_transform_2d(image, n_levels=4, n_angles=8, use_matlab=False, use_curvelops=True)` | `CurveletControlParameters` replaced by explicit params; multi-backend dispatch added (curvelops → MATLAB → NumPy fallback); return shape `(H,W,n_angles)` instead of DataFrame |
| `src/pycurvelets/pycurvelets3D.py`<br>`create_3d_curvelet(folder_path, num_images, nb_scales, nb_angles)` | `fiber_analysis/utils/curvelet_utils.py`<br>`curvelet_transform_3d(image, n_levels=3, n_angles=8, use_matlab=False, use_curvelops=True)` | File-based I/O replaced by in-memory ndarray `(Z,H,W)`; returns `(Z,H,W,n_angles)` energy array |
| `src/pycurvelets/get_fire.py`<br>`get_fire(img_name, fire_directory, fiber_mode, feature_cp)` | `fiber_analysis/utils/ctfire_utils.py`<br>`fire_2d(fiber_mask, image, pixel_size=1.0, ...)` | File-based `.mat` loading replaced by in-memory mask+image arrays; C++/Python dual backend; returns list of `(N,3)` centerline arrays instead of DataFrame |
| `src/pycurvelets/process_fibers.py`<br>`process_fibers(fiber_structure, feature_cp: FeatureControlParameters)` | `fiber_analysis/utils/fiber_dataframe_utils.py`<br>`compute_fiber_density_and_alignment(fiber_structure, params: FiberFeatureParams)` | `FeatureControlParameters` → `FiberFeatureParams`; `center_1`/`center_2` column aliases accepted; same return type `(density_df, alignment_df)` — Batch 1, commit `e842df0` |
| `src/pycurvelets/models/models.py`<br>`FeatureControlParameters(minimum_nearest_fibers, minimum_box_size, fiber_midpoint_estimate)` | `fiber_analysis/config.py`<br>`FiberFeatureParams(minimum_nearest_fibers=2, minimum_box_size=32, fiber_midpoint_estimate=1)` | Renamed; `to_dict()` added — Batch 1, commit `e842df0` |
| `src/pycurvelets/get_alignment_to_roi.py`<br>`get_alignment_to_roi(roi_list: ROIList, fiber_structure, distance_threshold=None)` | `tme_analysis/utils/alignment_utils.py`<br>`compute_fiber_alignment_to_roi(roi_coords, img_height, img_width, fiber_structure, distance_threshold=None)` | `ROIList` dataclass replaced by `(N,2)` ndarray + explicit dims; output columns renamed: `angle_to_boundary_edge` → `angle_to_boundary_tangent` (90°-complement), `angle_to_boundary_center` → `angle_to_roi_orientation`, `angle_to_center_line` → `angle_to_centers_line` — Batch 2, commit `e842df0` |

---

## Remaining (Future Batches)

| pycurvelets source → original signature | tme_quant planned target → proposed signature | Notes |
|---|---|---|
| `src/pycurvelets/get_ct.py`<br>`get_ct(img, curve_cp: CurveletControlParameters, feature_cp: FeatureControlParameters)` | `fiber_analysis/utils/fiber_dataframe_utils.py`<br>`build_fiber_structure_from_curvelets(image, curvelet_params, feature_params)` | Orchestrates `curvelet_transform_2d` + `compute_fiber_density_and_alignment`; returns `(fiber_structure_df, density_df, alignment_df, coefficients)` |
| `src/pycurvelets/get_tif_boundary.py`<br>`get_tif_boundary(coordinates, img, obj, dist_thresh, min_dist)` | `fiber_analysis/utils/boundary_tif_utils.py` (new)<br>`extract_tif_boundary(coordinates, img, obj, dist_thresh, min_dist)` | TIF boundary coordinate extraction + relative angle computation; also contains `get_relative_angle` and `get_points_on_line` helpers |
| `src/pycurvelets/utils/visualization/draw_curvs.py`<br>`draw_curvs(fiber_data, ax, length, color_flag, angles, mark_size, line_width, boundary_measurement)` | `fiber_analysis/visualization/draw_utils.py` (new) | No Qt dependency; pure matplotlib |
| `src/pycurvelets/utils/visualization/draw_map.py`<br>`draw_map(fiber_structure, angles, img, boundary_measurement, map_params)` | `fiber_analysis/visualization/draw_utils.py` (new) | Uses `scipy.ndimage.gaussian_filter`; no Qt dependency |
| `src/pycurvelets/utils/misc/format_df_to_excel.py`<br>`format_df_to_excel(df, filename, sheet_name='Sheet1', mode='w')` | `fiber_analysis/io.py` or `tme_analysis/io.py`<br>`export_dataframe_to_excel(df, filename, sheet_name='Sheet1', mode='w')` | Thin `openpyxl` wrapper; check which io.py already has similar functionality before placing |
| `src/pycurvelets/utils/math/flatten_numeric.py`<br>`flatten_numeric(series)` | `fiber_analysis/utils/fiber_dataframe_utils.py`<br>`flatten_numeric(series)` | One-liner utility; add to existing module |
| `src/pycurvelets/models/models.py` exception classes<br>`FiberAnalysisError`, `ROIProcessingError`, `BoundaryAnalysisError`, `FeatureExtractionError`, `ImageProcessingError` | `fiber_analysis/exceptions.py` (new) or `core/exceptions.py` (new) | Decide scope: fiber-only → `fiber_analysis/`; library-wide → `core/` |
| `src/pycurvelets/process_image.py`<br>`process_image(image_params, fiber_params, output_params, ...)` | **Do not integrate directly** | 1640-line orchestrator tightly coupled to file I/O and optional GUI widgets; decompose into sub-functions first; plan separately |

---

## Batch Protocol Summary

1. **Plan first** — present a plan and get approval before touching any file.
2. **Max 3–5 files** per batch (new + modified combined).
3. **`pytest tests/`** must be green before closing the batch.
4. **No overlap** — check the "Already Integrated" table and `REFACTORING_GUIDE.md §5`
   before writing a new function.
5. **No Qt/napari** in `src/tme_quant/` — see `REFACTORING_GUIDE.md §2`.

Full rules: [`REFACTORING_GUIDE.md`](../REFACTORING_GUIDE.md)
