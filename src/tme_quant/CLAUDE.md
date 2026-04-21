CLAUDE.md
# CLAUDE.md — TMEQuant Project Guide

This file provides Claude Code with the context, conventions, and rules needed
to work effectively on the TMEQuant codebase. Keep it in sync with the code.

---

## Project Overview

**TMEQuant** (`tme-quant`) is an open-source Python library for quantifying the
tumor microenvironment (TME). It ports and extends the MATLAB-based CurveAlign
and CT-FIRE platforms, adding comprehensive analysis of interactions between
collagen fibers, tumor boundaries, and cell populations.

**Primary capabilities:** fiber extraction (CT-FIRE, skeleton, ridge detection),
fiber orientation analysis (CurveAlign, OrientationJ, structure tensor, gradient),
TACS classification (types 1–3), cell segmentation and classification (StarDist,
Cellpose, thresholding), spatial interaction analysis, multi-modal image
registration (SHG ↔ H&E), 3D volumetric analysis, and a QuPath-style object
hierarchy supporting 2D, 3D, multichannel, and dynamic images.

**Lab:** UW-LOCI, University of Wisconsin–Madison.

---

## Repository Layout

```
tme-quant/
├── pyproject.toml               ← build, deps, tool config (black, ruff, mypy, pytest)
├── src/
│   ├── examples/                ← runnable demo scripts (no real images needed)
│   │   ├── example_ctfire_workflow.py
│   │   ├── example_ctfire_workflow_hierarchy.py  ← CT-FIRE workflow with TMEHierarchy
│   │   ├── example_curvealign_workflow.py
│   │   ├── example_3d_volumetric_workflow.py
│   │   ├── example_analyze_tacs_zone.py
│   │   ├── example_hierarchy_object_analysis.py
│   │   └── roi_curvealign_orientation_example.py
│   └── tme_quant/
│       ├── __init__.py          ← flat public API (~36 exported names)
│       ├── core/
│       ├── fiber_analysis/
│       ├── cell_analysis/
│       ├── tme_analysis/
│       ├── image_registration/
│       └── docs/
│           ├── architecture.md  ← canonical file tree — auto-updated by post-commit hook
│           └── getting_started.md
└── tests/
    ├── test_geometry_utils.py              ← geometry utils + real-dataset tests
    ├── test_alignment_to_roi.py            ← compute_fiber_alignment_to_roi
    ├── test_fiber_dataframe_utils.py       ← compute_fiber_density_and_alignment
    └── test_curvelet_fiber_candidates.py   ← extract_curvelet_fiber_candidates
```

> **Maintenance rule:** After every refactoring, file addition, or file removal,
> update **both** this module map **and** `docs/architecture.md` in the same
> commit. Do not let them drift from the actual directory tree. The post-commit
> hook in `.claude/settings.json` will attempt to auto-update `docs/architecture.md`
> on structural changes, but always verify it captured the intent correctly.

---

## Module Map

### `core/`

- `base_models.py` — `TMEObject` (single base class for all hierarchy nodes),
  `TMEType`, `ObjectType`, `Geometry`, `GeometryType`, `Measurement`,
  `Classification`, `TMEMetadata`, `BoundingBox`, `ROI`
- `hierarchy.py` — `TMEHierarchy` with `HierarchyIndex` (O(1) lookups),
  `spatial_assign()`, `attach_fiber_result()`, `attach_cell_result()`
- `image_entry.py` — `ImageEntry`: per-image metadata + lazy image loader
- `project.py` — `TMEProject`: top-level coordinator; imports `fiber_analysis`
  lazily to avoid circular imports
- `roi_manager.py` — `ROIManager`, `ROIObject`, `ANNOTATION_TYPES`
- `geometry.py` — Shapely-backed utilities: `compute_region_centroid`,
  `point_in_polygon`, `compute_convex_hull`, `buffer_polygon`
- `tme_objects/` — domain object subclasses (all inherit from `TMEObject`):
  - `base_objects.py` — backward-compatibility shim re-exporting from `core/base_models.py`
  - `fiber_objects.py` — `FiberObject`, `FiberPopulation`, `RegionOrientationMap`
  - `cell_objects.py` — `CellObject`, `CellPopulation`, `SegmentationMode`,
    `ClassificationMode`, `SegmentationParams`, `ClassificationParams`
  - `tumor_objects.py` — `Tumor`, `TumorRegion`, `TumorGrade`
  - `stroma_objects.py` — `StromaRegion`, `Stroma`, `ECMComponent`
  - `tissue_objects.py` — `TissueRegion`, `TissueSample`, `TissueZone`
  - `interaction_objects.py` — `Interaction`, `InteractionNetwork`

### `fiber_analysis/`

- `extraction.py` — `BaseExtractionMethod` (ABC) + `FiberExtractionAnalyzer`
- `orientation.py` — `BaseOrientationMethod` (ABC) + `FiberOrientationAnalyzer`
- `config.py` — all parameter dataclasses in one file: `ExtractionParams`,
  `CTFireParams`, `RidgeDetectionParams`, `SkeletonParams`, `ExtractionResult`,
  `FiberProperties`, `OrientationParams`, `CurveAlignParams`, `OrientationJParams`,
  `GradientParams`, `StructureTensorParams`, `OrientationResult`
- `tacs.py` — `classify_fiber_tacs()`, `classify_fiber_segment_tacs_like()`,
  `get_tacs_color()`
- `results.py` — `FiberAnalysisResult`
- `io.py` — `FiberAnalysisExporter` + re-export of `FijiBridge`
- `methods/` — concrete method implementations:
  - `ctfire.py` — `CTFireExtraction` (curvelet preprocessing + FIRE individual fiber extraction)
  - `curvealign.py` — `CurveAlignOrientation` (windowed curvelet orientation/coherency maps;
    planned to use `extract_curvelet_fiber_candidates` output as orientation source once
    `build_fiber_structure_from_curvelets` is integrated)
  - `skeleton.py` — `SkeletonExtractionMethod`
  - `ridge_detection.py` — `RidgeDetectionMethod`
  - `gradient.py` — `GradientOrientationMethod`
  - `structure_tensor.py` — `StructureTensorMethod`
  - `orientationj.py` — `OrientationJMethod`
  - `fiji_bridge.py` — `FijiBridge`, `FijiBackendMixin` (shared Fiji/ImageJ bridge)
- `utils/` — shared low-level utilities:
  - `geometry_utils.py` — `compute_angle_to_boundary_normal`,
    `compute_boundary_tangent_angle`, `compute_fiber_properties`,
    `compute_relative_fiber_angles`, `find_nearest_boundary_index`
  - `ctfire_utils.py` — FIRE algorithm Python impl + C++ wrapper interface;
    includes `_chain_segments_at_junctions()` for iterative multi-pass segment merging
  - `curvelet_utils.py` — curvelet transform with 3-backend dispatch; fallback #3 is
    now a Frangi ridge filter (`skimage.filters.frangi`) replacing the old FFT approximation
  - `fiber_dataframe_utils.py` — `build_fiber_structure_from_curvelets` (CurveAlign
    orchestrator; ported from `pycurvelets/get_ct.py`), `compute_fiber_density_and_alignment`,
    `flatten_numeric`, `round_mlab`; ported from `pycurvelets/process_fibers.py`

### `cell_analysis/`

- `cell_analyzer.py` — `CellAnalyzer` (orchestrator)
- `segmentation.py` — `CellSegmentationAnalyzer`
- `classification.py` — `CellClassificationAnalyzer`
- `quantification.py` — `CellQuantificationAnalyzer`
- `model_loader.py` — `ModelLoader`: downloads and caches StarDist / Cellpose
  models to `~/.tme_quant/models/`
- `config.py` — re-export shim → `tme_objects/cell_objects.py`
- `results.py` — re-export shim → `tme_objects/cell_objects.py`
- `io.py` — `CellAnalysisExporter`
- `methods/` — one file per method class (flat, no sub-packages):
  - `__init__.py` — `MethodRegistry`
  - `base_segmentation.py`, `base_classification.py`
  - `stardist_segmentation.py`, `cellpose_segmentation.py`,
    `threshold_segmentation.py`, `watershed_segmentation.py`
  - `morphology_classification.py`, `marker_classification.py`
- `utils/` — `cell_utils.py`, `postprocessing.py`, `preprocessing.py`,
  `validation.py`

### `tme_analysis/`

- `tme_analyzer.py` — `TMEAnalyzer` (orchestrator)
- `interaction_detector.py` — `CellFiberInteractionDetector`
- `interaction_network.py` — `InteractionNetworkAnalyzer`
- `measurement_engine.py` — `MeasurementEngine`
- `region_manager.py` — `RegionManager`
- `tacs_features.py` — `TACSFeatureExtractor`
- `interaction_features.py` — mechanical / migration / invasive scores
- `spatial_features.py` — `SpatialRelationshipExtractor`
- `config.py` — `TMEAnalysisParams`, `TumorDetectionParams`, `AnalysisMode`,
  `TMEAnalysisResult`
- `io.py` — `export_tme_analysis_results()`
- `pipelines/` — high-level pipelines:
  - `standard_tme_pipeline.py` — `StandardTMEPipeline`
  - `interaction_analysis_pipeline.py` — `InteractionAnalysisPipeline`
  - `tacs_pipeline.py` — `analyze_tacs_zone()`, `plot_tacs_heatmap()`
  - `curvealign_pipeline.py` *(planned)* — full CurveAlign pipeline: curvelet fiber
    extraction, density/alignment, ROI alignment, TACS; port of `pycurvelets/process_image.py`
- `utils/` — `alignment_utils.py` (`compute_fiber_alignment_to_roi`; ported
  from `pycurvelets/get_alignment_to_roi.py`), `distance_utils.py`,
  `orientation_utils.py` (pixel-level boundary-relative orientation,
  `compute_orientation_relative_to_roi`, `discretize_roi_boundary`),
  `statistical_utils.py`, `validation.py`
- `visualization/` — `interaction_visualization.py`

### `image_registration/`

- `registration_manager.py` — `RegistrationManager`
- `config.py` — `RegistrationParams`, `TransformType`
- `preprocessing.py` — 16 merged preprocessing functions
- `transform_handler.py`, `io.py`
- `methods/intensity_based/` — `HESHGRegistration`, cross-correlation, mutual information
- `methods/feature_based/` — SIFT, ORB
- `methods/landmark_based/` — TPS, manual landmarks
- `methods/deep_learning/` — CoMIR, VoxelMorph
- `utils/` — `image_utils.py`, `transform_utils.py`
- `visualization/` — checkerboard, overlay, difference map, interactive viewer

---

## C++ Extension Integration  ⚠️ In Progress

Two computationally intensive algorithms have C++ implementations that are
being integrated via Python bindings. Both are currently stubs with pure-Python
fallbacks; see status flags in each module.

### CT-FIRE FIRE Algorithm (`_ctfire_cpp`)

The FIRE (Fiber Extraction) graph-based tracing algorithm:

- **C++ source (upstream):** `https://github.com/uw-loci/curvelets/tree/master/src/CurveAlign_CT-FIRE/ctFIRE/CPP`
- **C++ source (this repo):** `fiber_analysis/_cpp/ctfire/` — `fire.h`, `fire.cpp`, `fire_bindings.cpp`, `CMakeLists.txt`
- **Python wrapper module:** `_ctfire_cpp` (pybind11); compiled output placed in `fiber_analysis/utils/`
- **Integration point:** `fiber_analysis/utils/ctfire_utils.py` — uses `from . import _ctfire_cpp`
- **Status flag:** `_CPP_AVAILABLE = False` in `ctfire_utils.py`; set to `True`
  when the shared library is built and installed
- **Entry points:**
  - `_fire_cpp_2d(mask, pixel_size, params)` → list of fiber trace dicts
  - `_fire_cpp_3d(mask, voxel_size, params)` → list of 3D fiber trace dicts
- **Build:** `cmake -S fiber_analysis/_cpp/ctfire -B build/ctfire_cpp && cmake --build build/ctfire_cpp && cmake --install build/ctfire_cpp`
- **Fallback:** Pure-Python distance-transform tracer (slower, less accurate
  for touching fibers)
- **3D status:** `CTFireExtraction.supports_3d()` returns `False` until the
  C++ 3D FIRE extension is compiled; `extract_3d` raises `NotImplementedError`
  at the FIRE stage (curvelet and mask steps work)

### Curvelet Transform (`curvelops` / `_curvelet_cpp`)

Three backends in priority order:

1. **curvelops** (recommended) — Python wrapper around C++ FDCT2D/FDCT3D via
   the PyLops framework. Genuine 2D and 3D curvelet transforms.
   `pip install curvelops`
2. **MATLAB Engine** (legacy) — calls original CT-FIRE MATLAB curvelet functions.
   Requires licensed MATLAB + MATLAB Engine for Python.
3. **Frangi ridge-filter fallback** (always available) — Hessian-based multi-scale
   ridge detector via `skimage.filters.frangi`. Approximates curvelet fiber responses
   without directional decomposition. Replaced the old FFT approximation (commit `46fe7ae`)
   because the FFT directional decomposition was unreliable on real SHG data.
   3D fallback is slice-by-slice, not a true volumetric transform.

Backend selection is automatic via `_has_curvelops()` / `_has_matlab_engine()`
probes in `curvelet_utils.py`. Check `ctfire_backend_status()` at runtime.

**When adding new C++ wrappers**, follow the same pattern:
1. Add an `_EXTENSION_AVAILABLE` flag at module top
2. Probe availability lazily (inside the function, not at import time)
3. Provide a documented pure-Python fallback
4. Expose backend status via a `*_backend_status()` function

---

## TACS Angle Convention  ⚠️ Critical

All TACS functions in `fiber_analysis/tacs.py` and `tme_analysis/utils/orientation_utils.py`
expect **`angle_to_tangent`**: the acute angle (0–90°) between the fiber
orientation and the **local boundary tangent line**.

```
angle_to_tangent =  0–30°                          → TACS-2 (parallel)
angle_to_tangent = 60–90°                          → TACS-3 (perpendicular)
angle_to_tangent = 30–60°, or straightness < 0.7  → TACS-1
```

If you have the angle to the boundary **normal**, convert first:
```python
angle_to_tangent = 90.0 - angle_to_normal
```

The two TACS analysis paths (pixel-map and per-fiber) are unified in
`tme_analysis/pipelines/tacs_pipeline.py::analyze_tacs_zone()`. Use this
function rather than calling `compute_orientation_relative_to_roi` and
`compute_relative_fiber_angles` separately.

The straightness threshold `0.7` applies to CT-FIRE fibers only. Do not
change it without reviewing `tacs.py` and `measurement_engine.py` together.

---

## Architecture Rules

> **When integrating MATLAB-converted functions or performing library-wide refactors,
> strictly follow the rules in [@REFACTORING_GUIDE.md](REFACTORING_GUIDE.md).**
> That document defines anti-hallucination rules, mandatory test-suite runs (pytest),
> incremental batch limits (max 3–5 files), overlap-check protocol, and naming
> conventions for adapted functions.

### Core / Plugin separation

`tme_quant` (this library) and `napari-tme-quant` (the GUI plugin) are **separate
packages** governed by a strict one-way dependency:

```
napari-tme-quant → tme_quant   (plugin depends on library)
tme_quant        ← (no dependency on napari, Qt, or magicgui)
```

- **No Qt/napari imports** (`napari`, `qtpy`, `PyQt5`, `PySide2`, `PySide6`,
  `magicgui`) anywhere inside `src/tme_quant/`.
- GUI logic, event-handling, and layer management live exclusively in the plugin
  package (documented in [CLAUDE_NAPARI.md](CLAUDE_NAPARI.md)).
- Plugin code must call `tme_quant` APIs — never the reverse.
- Analysis logic that a plugin widget needs must be added to the **library**,
  then called from the plugin controller layer.

### Object model
- Every domain object inherits from `TMEObject` in `core/base_models.py`.
- `TMEObject` is a **regular class** (not a dataclass) — deep inheritance with
  dataclasses causes MRO and default-field conflicts. Do not convert it.
- `object_id` is the primary key (string, caller-supplied or UUID). Never use
  integer indices as primary keys.
- Manage parent/child relationships through `TMEObject.add_child()` /
  `TMEObject.detach()`. If you bypass these, call `hierarchy.rebuild_index()`.
- `core/tme_objects/base_objects.py` is a **shim** that re-exports from
  `core/base_models.py`. Do not add new logic there.

### Import rules
- **Cross-subpackage** imports must use **absolute** imports:
  `from tme_quant.core.tme_objects.cell_objects import ...`
- **Within a subpackage**, use relative imports keyed to file depth:
  - `tme_quant/<pkg>/file.py` (depth 2) → `from ..core...`
  - `tme_quant/<pkg>/utils/file.py` (depth 3) → `from ...core...`
  - `tme_quant/<pkg>/methods/file.py` (depth 3) → `from ...core...`
- Never import `fiber_analysis` at module top level inside `core/`. That
  import must remain lazy (inside `__init__()`) to prevent circular imports.
- Optional deps (`torch`, `stardist`, `cellpose`, `curvelops`, `pyimagej`,
  `matlab`, `sklearn`, `plotly`, `community`) must be imported **lazily**
  (inside the function that uses them) so `import tme_quant` succeeds without
  them. Probe availability with `_has_X()` helper functions.
- `shapely`, `cv2`, `tifffile`, `imageio`, `scipy`, `skimage` are core deps
  and may be imported at module top level.

### Method dispatch
- `FiberExtractionAnalyzer` and `FiberOrientationAnalyzer` use a plain
  `Dict[Mode, Type[BaseMethod]]`. Do not introduce a `MethodRegistry` in
  `fiber_analysis/`.
- `CellSegmentationAnalyzer` and `CellClassificationAnalyzer` use
  `MethodRegistry` from `cell_analysis/methods/__init__.py`.

### Config / params
- All params are `@dataclass` with `to_dict()`. One `config.py` per subpackage.
  Do not split a config file into sub-modules.
- `cell_analysis/config.py` and `cell_analysis/results.py` are thin re-export
  shims pointing to `tme_objects/cell_objects.py`. Keep params defined there.

### Backwards compatibility
- `FiberAnalyzer = FiberExtractionAnalyzer` alias in `fiber_analysis/__init__.py`.
  Keep it — examples use both names.

---

## Development Setup

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[fiber]"        # + curvelops, pyimagej
pip install -e ".[dl]"           # + torch, stardist, cellpose
pip install -e ".[network]"      # + networkx, plotly, python-louvain
pip install -e ".[examples]"     # everything needed for all examples
pip install -e ".[napari]"       # napari GUI plugin stack
```

Python ≥ 3.10. Line length: **100** (black + ruff). Use `uv pip install` for
faster dependency resolution.

Models for StarDist / Cellpose are auto-downloaded on first use and cached at
`~/.tme_quant/models/` via `cell_analysis/model_loader.py`.

### Building C++ Extensions (optional)

The C++ extensions are optional — the package imports and runs without them.
Build them to get the full-speed FIRE algorithm instead of the Python fallback.

**Prerequisites:** cmake ≥ 3.18, a C++17 compiler (MSVC 2019+, GCC 9+, or
Clang 10+), and pybind11:

```bash
pip install pybind11
```

**Build `_ctfire_cpp`** (FIRE fiber extraction):

```bash
# From the repo root (tme-quant/)
cmake -S src/tme_quant/src/tme_quant/fiber_analysis/_cpp/ctfire \
      -B build/ctfire_cpp \
      -DCMAKE_BUILD_TYPE=Release
cmake --build  build/ctfire_cpp
cmake --install build/ctfire_cpp
# installs _ctfire_cpp.pyd/.so into fiber_analysis/utils/
```

**Enable in Python** — after a successful build, flip the flag in
`fiber_analysis/utils/ctfire_utils.py`:

```python
_CPP_AVAILABLE: bool = True   # was False
```

**Verify:**

```python
from tme_quant.fiber_analysis.utils.ctfire_utils import ctfire_backend_status
print(ctfire_backend_status())   # 'cpp_available' should be True
```

---

## Running Examples and Tests

```bash
# These 3 examples require real image files (place in data/ at the project root):
PYTHONPATH=src python src/examples/example_ctfire_workflow.py
PYTHONPATH=src python src/examples/example_ctfire_workflow_hierarchy.py
PYTHONPATH=src python src/examples/example_curvealign_workflow.py
# These examples are self-contained (synthetic data, no real images needed):
PYTHONPATH=src python src/examples/example_3d_volumetric_workflow.py
PYTHONPATH=src python src/examples/example_analyze_tacs_zone.py
PYTHONPATH=src python src/examples/example_hierarchy_object_analysis.py

# Standard test suite (171 passed, 7 skipped as of 2026-04-20)
# Run from src/tme_quant/ — curvelops integration tests are skipped automatically
# when curvelops is not installed.
pytest tests/ -v

# Full test suite WITH curvelops (requires curvelops installed)
# On this machine curvelops lives in WSL miniconda — run from WSL:
#
#   wsl bash -c "cd /mnt/h/GitHub.06.2022/tme-quant/src/tme_quant && \
#       ~/miniconda3/bin/python -m pytest tests/ -v"
#
# Expected: 171 passed, 7 skipped (MATLAB parity checks disabled by default)

# Strict MATLAB-reference parity assertions (needs TMEQ_VALIDATE_MATLAB=1):
#   TMEQ_VALIDATE_MATLAB=1 pytest tests/test_curvelet_fiber_candidates.py -v
# or from WSL:
#   wsl bash -c "cd /mnt/h/GitHub.06.2022/tme-quant/src/tme_quant && \
#       TMEQ_VALIDATE_MATLAB=1 ~/miniconda3/bin/python -m pytest \
#       tests/test_curvelet_fiber_candidates.py -v"

# Linting and type checking
black src/ && ruff check src/ && mypy src/tme_quant/
```

### Installing / locating curvelops

**General install instructions:** `docs/getting_started.md` Step 2 and Step 6.

**On this machine:** curvelops 0.23 is installed in WSL at
`/home/yuming/miniconda3/`.  FFTW 2.1.5 and CurveLab 2.1.3 are pre-built
(ELF/Linux) under `H:/GitHub.06.2022/utils/` — the Windows `.venv` cannot
link against them.  Always use the WSL miniconda Python for curvelops-dependent
tests (see the WSL command above).

---

## Import Health Check

Run after any structural change:

```python
import sys, importlib, unittest.mock as mock
sys.path.insert(0, "src")

for lib in ["shapely", "shapely.geometry", "shapely.ops", "cv2", "stardist",
            "cellpose", "torch", "torch.nn", "voxelmorph", "imageio", "tifffile",
            "napari", "plotly", "openpyxl", "imagej", "matlab", "curvelops",
            "community", "comir", "sklearn", "sklearn.cluster",
            "sklearn.neighbors", "sklearn.preprocessing", "sklearn.metrics",
            "scipy.stats"]:
    sys.modules[lib] = mock.MagicMock()

for m in ["tme_quant.core", "tme_quant.fiber_analysis", "tme_quant.cell_analysis",
          "tme_quant.tme_analysis", "tme_quant.image_registration", "tme_quant"]:
    try:
        importlib.import_module(m)
        print(f"OK  {m}")
    except Exception as e:
        print(f"ERR {m}: {e}")
```

Expected: all `OK`. The `scipy.stats` mock is only needed when `torch` is absent.

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `from __future__` SyntaxError mid-file | Duplicate header after module concatenation | Keep only one at line 1 |
| `ModuleNotFoundError: ...config.extraction_params` | Stale dotted path after config consolidation | `from .config import ...` |
| `attempted relative import beyond top-level package` | Too many dots for file depth | depth-2 → `..`, depth-3 → `...`, depth-4 → absolute |
| `ImportError: cannot import name X from partially initialized module` | Circular import at load time | Lazy import inside `__init__()` or `TYPE_CHECKING` guard |
| `issubclass() arg 2 must be a class` in tests | Mocked `torch` breaks `scipy.stats` | Also mock `scipy.stats` |
| `CTFireExtraction.supports_3d()` returns False | C++ 3D FIRE extension not yet compiled | Expected; use 2D or wait for C++ build |
| Wrong TACS type classification | Passing angle-to-normal instead of angle-to-tangent | Convert: `angle_to_tangent = 90 - angle_to_normal` |