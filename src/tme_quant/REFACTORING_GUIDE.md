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

Functions that are **not yet integrated** and are suitable targets for future batches:

| pycurvelets                  | Target in tme_quant                                      |
|------------------------------|-----------------------------------------------------------|
| `process_fibers`             | `fiber_analysis/utils/fiber_dataframe_utils.py`          |
| `get_alignment_to_roi`       | `tme_analysis/utils/alignment_utils.py`                  |
| `new_curv`                   | `fiber_analysis/utils/curvelet_utils.py::extract_curvelet_fiber_candidates` |
| `get_tif_boundary`           | `fiber_analysis/utils/boundary_tif_utils.py`             |
| `draw_curvs`                 | visualization layer (no Qt dependency)                   |
| `draw_map`                   | visualization layer (no Qt dependency)                   |
| `format_df_to_excel`         | `fiber_analysis/io.py` or `tme_analysis/io.py`           |

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

---

## 7. MATLAB Rounding and Numerical Parity

`pycurvelets` includes `round_mlab` for MATLAB-compatible rounding (round-half-away-
from-zero).  If a ported function relies on MATLAB rounding for numerical parity with
reference data, use `fiber_analysis.utils.fiber_dataframe_utils.round_mlab` (available
after Batch 1) rather than Python's built-in `round()`.

---

*Last updated: 2026-04-20 — Batch 0 (REFACTORING_GUIDE and CLAUDE.md bootstrap)*
