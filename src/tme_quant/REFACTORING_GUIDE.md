# REFACTORING_GUIDE.md — TMEQuant Integration & Refactoring Rules

This file governs all work that integrates external code (especially MATLAB-converted
Python functions from `pycurvelets`) into the `tme_quant` library, or that performs
library-wide refactoring.  **All contributors and AI assistants must follow these rules
without exception.**

---

## 1. Anti-Hallucination Rules

### 1.1  Never guess external API shapes

Before calling any function from `pycurvelets`, `curvelops`, `sklearn`, `scipy`, or
any other library:

- **Read the source file** or run `help()` / inspect the signature in a REPL.
- Never assume parameter names, return types, or default values from memory.
- If the source is not available, mark the call as `# TODO: verify signature` and
  raise `NotImplementedError` rather than silently passing wrong arguments.

### 1.2  Preserve existing signatures exactly

When adapting a function from `pycurvelets` into `tme_quant`:

1. Keep **every parameter** from the original, even if it appears unused.
2. Keep **return types and shapes** identical (same dtypes, same column names in
   DataFrames, same array shapes).
3. Deviations from the original are allowed **only** when:
   - The original uses a pycurvelets-specific type (e.g. `ROIList`, `Fiber`) that has
     a direct tme_quant equivalent (e.g. `ROIObject`, `FiberProperties`).
   - The CLAUDE.md naming convention explicitly requires a different column name
     (e.g. `angle_to_boundary_tangent` instead of `angle_to_boundary_edge`).
   - In both cases, document the deviation with an inline comment referencing the
     original name.

### 1.3  No invented behaviour

Do not add optional parameters, alternate code paths, or extra output columns that
do not exist in the source function being ported.  Any extension must be a **separate
PR** with a dedicated test.

### 1.4  Verify imports before writing them

Every `import` statement in a new file must correspond to a package that is:

- Listed in `pyproject.toml` (core or optional dep), **or**
- A Python standard-library module, **or**
- Already used elsewhere in the same subpackage.

If the import requires a new dependency, add it to `pyproject.toml` in the same PR.

---

## 2. Core / Plugin Separation

`tme_quant` (core library) and `napari-tme-quant` (plugin) are **separate packages**
with a strict one-way dependency:

```
napari-tme-quant → tme_quant   (plugin depends on library)
tme_quant        ← (no dependency on napari or Qt)
```

**Hard rules:**

- No `import napari`, `import qtpy`, `import PyQt5`, or `import PySide6` anywhere
  inside `src/tme_quant/`.
- No `napari.types`, `magicgui`, or `napari.layers` references anywhere in core code.
- GUI layout, event-handling, and layer management belong in `napari-tme-quant/`.
- If a function needed by the plugin requires Qt, it must live in the plugin package
  and call `tme_quant` APIs — not the other way around.

Violations will break headless (server/HPC) usage of the library and must be caught
in code review before merge.

---

## 3. Incremental Execution (Max 3–5 Files per Batch)

Every integration or refactoring session is divided into **batches**.  Each batch:

1. Touches **at most 5 files** (new + modified combined).
2. Ends with a **full `pytest` run** before starting the next batch.
3. All touched files must pass the import health check (see CLAUDE.md) after the batch.

Batch sizing rationale: small batches make it easy to bisect failures, keep diffs
reviewable, and prevent "big bang" refactors that are impossible to merge safely.

### Batch checklist

Before opening a batch:

- [ ] List the ≤5 files that will be created or changed.
- [ ] Confirm zero overlap with existing tme_quant functions (check `__all__` in the
      relevant `__init__.py` files).
- [ ] State which pycurvelets source function(s) are being adapted.
- [ ] Note any signature deviations and why they are necessary.

After completing a batch:

- [ ] `pytest` passes (green) for the new test file(s).
- [ ] `pytest tests/test_geometry_utils.py` still passes (regression check).
- [ ] Import health check passes for all affected subpackages.
- [ ] Real-dataset tests ported (see §3.1 below).

### 3.1  Real-dataset test porting

For every function ported from pycurvelets, check whether a corresponding
real-dataset test exists in `H:/GitHub.06.2022/tme-quant/tests/`:

```
tests/
├── test_relative_angles.py          → compute_relative_fiber_angles
├── test_get_alignment_to_roi.py     → compute_fiber_alignment_to_roi
├── test_new_curv.py                 → extract_curvelet_fiber_candidates  ✓
├── test_get_ct.py                   → build_fiber_structure_from_curvelets
└── test_results/
    ├── relative_angle_test_files/   → boundary_coords.csv, real1_BoundaryMeasurements.xlsx
    ├── process_image_test_files/    → real1_roi_df.csv, real1_fiber_structure.csv, ...
    ├── new_curv_test_files/         → test_cases_new_curv.json, MATLAB reference CSVs
    └── get_ct_test_files/           → (future)
```

**Rules:**

1. If a real-dataset test file exists in `tme-quant/tests/` for the pycurvelets
   source, add a corresponding `TestXxxRealData` class to the appropriate
   `src/tme_quant/tests/test_*.py` file in the **same batch**.
2. Reference data lives in `tme-quant/tests/test_results/` and is shared —
   **do not copy** it into `src/tme_quant/tests/`.  Resolve the path via
   `Path(__file__).parent.parent.parent.parent / "tests"`.
3. Guard the entire class with
   `@pytest.mark.skipif(not _DATA_DIR.exists(), reason="pycurvelets test data not found")`
   so CI without the pycurvelets subtree still passes.
4. Angle convention conversions required (see §6):
   - `angle2boundaryEdge` (ref) → compare against `90 − angle_to_boundary_tangent` (result)
   - `angle2boundaryCenter` (ref) → compare against `angle_to_roi_orientation`
   - `angle2centersLine` (ref) → compare against `angle_to_centers_line`
     (note: coordinate-mixing bug not replicated; small divergence expected)
5. For curvelops-dependent tests, additionally gate on
   `pytest.importorskip("curvelops")` and run under WSL miniconda
   (see CLAUDE.md "Running tests").

---

## 4. Mandatory Test Suite Runs

After **every batch**, run:

```bash
# From the tme-quant project root (src/tme_quant/)
pytest tests/ -v
```

**No batch is complete until all tests pass.**

If a test fails:

1. Fix the implementation — never skip or xfail a test to make CI green.
2. If the failure is in a pre-existing test unrelated to your change, open a separate
   issue before proceeding.
3. Never modify a test to weaken its assertions in order to pass.

---

## 5. Overlap Check Protocol

Before adding any function to `tme_quant`, confirm it does not duplicate an existing
one by searching:

```bash
grep -r "def <function_name>" src/tme_quant/src/tme_quant/
```

Check also for semantic overlap (same computation, different name) by reading the
docstrings of related modules listed in CLAUDE.md's Module Map.

Known equivalences between pycurvelets and tme_quant (do NOT re-implement these):

| pycurvelets                        | tme_quant                                         |
|------------------------------------|---------------------------------------------------|
| `circ_r`                           | `geometry_utils._circ_r`                         |
| `find_outline_slope`               | `geometry_utils.compute_boundary_tangent_angle`  |
| `find_connected_pts`               | `geometry_utils._find_connected_pts`             |
| `get_first_neighbor`               | `geometry_utils._get_first_neighbor`             |
| `get_relative_angles`              | `geometry_utils.compute_relative_fiber_angles`   |
| `pycurvelets3D` (3D curvelets)     | `curvelet_utils.curvelet_transform_3d`           |
| `get_fire` (FIRE algorithm)        | `ctfire_utils.fire_2d`                           |

All pycurvelets functions from the original decomposition plan have been integrated.
See `docs/pycurvelets_integration_status.md` for the full cross-reference and
`docs/pycurvelets_integration_status.md#remaining` for any new functions added
as future conversion targets.

---

## 6. Naming Conventions

When adapting pycurvelets code, apply the following renames for consistency with
tme_quant conventions:

| pycurvelets column / field       | tme_quant equivalent                    |
|----------------------------------|-----------------------------------------|
| `angle_to_boundary_edge`         | `angle_to_boundary_tangent`             |
| `angle_to_boundary_center`       | `angle_to_roi_orientation`              |
| `angle_to_center_line`           | `angle_to_centers_line`                 |
| `center_1` / `center_2`         | `center_row` / `center_col`             |
| `CurveletControlParameters`      | `CurveAlignParams`                      |
| `FeatureControlParameters`       | `FiberFeatureParams`                    |
| `ROIList`                        | `np.ndarray` coords + explicit dims     |

Note: `angle_to_boundary_edge` in pycurvelets equals `90° − angle_to_boundary_tangent`
in tme_quant TACS convention (pycurvelets returns the complement angle).  Always apply
the conversion; never silently carry over the pycurvelets value with the tme_quant name.

Note: `nearest_relative_boundary_angle` (output of `extract_tif_boundary` /
`_compute_fiber_boundary_relative_angle`) equals `angle_to_boundary_tangent` **directly**
— no conversion needed.  `compute_boundary_tangent_angle` uses `atan2(Δcol, Δrow)`,
which is 90° offset from the fiber-angle convention; this offset cancels the expected
complement, so the raw column value is already the TACS-ready angle_to_tangent.

---

## 7. MATLAB Rounding and Numerical Parity

`pycurvelets` includes `round_mlab` for MATLAB-compatible rounding (round-half-away-
from-zero).  If a ported function relies on MATLAB rounding for numerical parity with
reference data, use `fiber_analysis.utils.fiber_dataframe_utils.round_mlab` (available
after Batch 1) rather than Python's built-in `round()`.

---

## 8. TMEObject Model Integration

### 8.1  Type mapping — pycurvelets → tme_quant

When a pycurvelets function accepts or returns one of the types in the left column,
use the tme_quant equivalent in the right column.  **Never import pycurvelets types
into the library.**

| pycurvelets type                   | tme_quant equivalent                                        | Defined in                                       |
|------------------------------------|-------------------------------------------------------------|--------------------------------------------------|
| `ROIList` (dataclass)              | `(N, 2) ndarray` (row, col) + explicit `img_height`/`img_width` | caller-supplied                           |
| `ROIList.coordinates[i]`           | single `roi_coords` ndarray per call                        | caller-supplied                                  |
| `CurveletControlParameters`        | `CurveAlignParams`                                          | `fiber_analysis/config.py`                       |
| `FeatureControlParameters`         | `FiberFeatureParams`                                        | `fiber_analysis/config.py`                       |
| `Fiber` dataclass (single fiber)   | `FiberObject`                                               | `core/tme_objects/fiber_objects.py`              |
| fiber measurement dict / row       | `FiberProperties`                                           | `fiber_analysis/config.py`                       |
| fiber DataFrame (many fibers)      | `pd.DataFrame` with canonical columns (see §8.3)            | caller-supplied                                  |

### 8.2  Two-layer design: util functions vs. FiberObject methods

Every ported function that operates on fibers lives at **two levels**:

```
Layer 1 — low-level utility (pure function, returns DataFrame or dict)
    fiber_analysis/utils/<module>.py
    → accepts plain ndarrays + DataFrames
    → has no knowledge of TMEObject / hierarchy
    → this is what the tests exercise directly

Layer 2 — FiberObject method (wraps Layer 1, updates self)
    core/tme_objects/fiber_objects.py  (FiberObject)
    → calls the Layer 1 util
    → stores results as FiberObject attributes
    → triggers TACS re-classification when boundary angles change
    → example: FiberObject.compute_boundary_relative_metrics()
               calls compute_relative_fiber_angles() and stores the result
```

**Rules:**

1. Always implement Layer 1 first and test it independently.
2. Layer 2 (the FiberObject method) is added only when the analysis result
   needs to be carried through the hierarchy (e.g., for TACS classification,
   export, or downstream spatial queries).
3. Layer 1 functions must never import from `core/`.  Layer 2 methods import
   from `fiber_analysis/` using relative imports.
4. If a pycurvelets function only computes summary statistics (density, alignment
   scores) and writes them to a DataFrame — not to individual fiber objects —
   Layer 2 is not needed; the DataFrame is the output.

### 8.3  Canonical DataFrame column names

When a ported function returns a per-fiber DataFrame, use these column names
(already established by existing ported functions):

| Column name           | Type    | Description                                        |
|-----------------------|---------|----------------------------------------------------|
| `center_row`          | float   | Fiber centre row coordinate (0-indexed)            |
| `center_col`          | float   | Fiber centre col coordinate (0-indexed)            |
| `angle`               | float   | Fiber orientation degrees [0°, 180°)               |
| `length`              | float   | Arc length in pixels (or µm with pixel_size)       |
| `width`               | float   | Mean fiber width                                   |
| `straightness`        | float   | End-to-end / arc-length ∈ [0, 1]                  |
| `angle_to_boundary_tangent` | float | Acute angle to local boundary tangent [0°, 90°] |
| `angle_to_roi_orientation`  | float | Acute angle to global ROI orientation [0°, 90°]  |
| `angle_to_centers_line`     | float | Acute angle to fiber-ROI centroid line [0°, 90°] |
| `distance`            | float   | Distance to nearest boundary point (pixels)        |
| `boundary_point_row`  | float   | Nearest boundary point row                         |
| `boundary_point_col`  | float   | Nearest boundary point col                         |

Aliases `center_1` / `center_2` (pycurvelets convention) are accepted as
**input** only; always normalise to `center_row` / `center_col` inside the
function (see existing pattern in `alignment_utils.py`).

**Important:** `center_1` from pycurvelets maps to `center_row` (row / Y
direction) and `center_2` maps to `center_col` (col / X direction).  In MATLAB
CurveAlign output, `fibercenterX = center_1` (because MATLAB stores row-first
but labels it X in its own output files).  Do not infer the mapping from the
MATLAB column label alone — always check the actual numeric values.

### 8.4  Coordinate conventions quick-reference

| Context                                         | Convention       | Example                                    |
|-------------------------------------------------|------------------|--------------------------------------------|
| `roi_coords` array passed to util functions     | (row, col)       | skimage / numpy standard                  |
| `obj_center` tuple passed to `compute_relative_fiber_angles` | **(col, row) = (x, y)** | `(fiber_col, fiber_row)` |
| `FiberObject.centerline`                        | (row, col)       | from CT-FIRE / skeleton extraction        |
| `FiberObject.center_point`                      | (x, y) = (col, row) | from `orientation_point` or midpoint   |
| `find_nearest_boundary_index(coords, px, py)`  | px = row, py = col | both in same space as coords            |
| KDTree queries on boundary (alignment_utils)    | (row, col)       | matches `roi_coords` format               |
| `FiberObject.compute_boundary_relative_metrics` | passes `center_point` (x, y) directly as `obj_center` | see fiber_objects.py |

**Critical:** `compute_relative_fiber_angles` takes `obj_center` as `(x, y)` =
`(col, row)`, **not** `(row, col)`.  This differs from every other array in the
pipeline.  The signature is consistent with the `ROI.centroid → (x, y)` output
path used internally.  Always pass `(col_obj, row_obj)` — never `(row_obj, col_obj)`.

### 8.5  FiberObject attribute target for each angle type

When a ported function computes boundary-relative angles, store them on
`FiberObject` using these attributes (do not invent new ones):

| Computed value                     | FiberObject attribute                        |
|------------------------------------|----------------------------------------------|
| `angle_to_boundary_tangent`        | `self.relative_angle_to_boundary_tangent`    |
| `angle_to_roi_orientation`         | `self.angle_to_roi_orientation`              |
| `angle_to_centers_line`            | `self.angle_to_centers_line`                 |
| distance to nearest boundary point | `self.nearest_boundary_distance`             |
| nearest boundary point coords      | `self.nearest_boundary_point` (in µm, x/y)  |

After storing `relative_angle_to_boundary_tangent`, always call
`self._classify_tacs_from_metrics()` to keep `tacs_type` and `tacs_score` in
sync.  `FiberObject.compute_boundary_relative_metrics` does this automatically;
do not bypass it.

### 8.6  ROIList → plain arrays: multi-ROI functions

pycurvelets functions that loop over `ROIList.coordinates` (e.g.
`get_alignment_to_roi`) become tme_quant functions that accept a **single**
`roi_coords` array.  The caller is responsible for iterating over multiple ROIs:

```python
# pycurvelets (multi-ROI loop inside the function)
results = get_alignment_to_roi(roi_list, fiber_structure, distance_threshold=100)

# tme_quant (caller loops)
all_results = []
for roi_coords in roi_coords_list:
    result_df, count = compute_fiber_alignment_to_roi(
        roi_coords, img_height, img_width, fiber_structure, distance_threshold=100
    )
    all_results.append(result_df)
```

This keeps each function testable in isolation and avoids hiding iteration
complexity inside low-level utils.

---

## 9. Exception Classes

pycurvelets defines `FiberAnalysisError`, `ROIProcessingError`,
`BoundaryAnalysisError`, `FeatureExtractionError`, and `ImageProcessingError`
in `src/pycurvelets/models/models.py`.

These have not yet been ported.  Until they are:

- Raise standard Python exceptions (`ValueError`, `RuntimeError`, `ImportError`)
  at tme_quant function boundaries.
- Do not create new custom exception classes without an approved batch plan.
- When porting, place fiber-scoped exceptions in `fiber_analysis/exceptions.py`
  (new file) and library-wide exceptions in `core/exceptions.py` (new file).

---

*Last updated: 2026-04-21 — §5 overlap table updated: all original decomposition targets integrated (Batches 1–9C)*
