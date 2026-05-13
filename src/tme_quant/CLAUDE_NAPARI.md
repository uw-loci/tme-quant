CLAUDE_NAPARI.md
# CLAUDE_NAPARI.md — napari-TMEQuant Plugin Guide

This file provides Claude Code with context and rules for the **napari-TMEQuant**
plugin subproject. Read `CLAUDE.md` first for the core `tme_quant` library —
this file covers only plugin-specific concerns.

> **Maintenance rule:** Keep this file in sync with `CLAUDE.md` and
> `docs/architecture.md`. Update the widget list, signal table, and file map
> whenever a new widget or controller is added.

---

## What This Plugin Is

**napari-TMEQuant** is a napari plugin that exposes the `tme_quant` library
through an interactive GUI. It is the primary interface for researchers who
want to run TME analysis without writing Python code.

It is a **separate package** (`napari-tme-quant`) that depends on `tme-quant`
as a library. It lives alongside the core library:

```
tme-quant/                    ← core library (tme_quant/)
napari-tme-quant/             ← this plugin
    pyproject.toml
    src/
        napari_tme_quant/
            __init__.py
            _main_widget.py           ← TMEQuantDockWidget (QTabWidget container)
            widgets/                  ← one file per dock panel (all independently registerable)
                project_widget.py
                image_widget.py
                roi_manager_widget.py
                registration_widget.py
                fiber_analysis_widget.py
                cell_analysis_widget.py
                tme_pipeline_widget.py
                measurements_widget.py
                visualization_widget.py
                io_widget.py
                log_widget.py
            controllers/
                state.py              ← PluginState singleton
                project_controller.py
                roi_controller.py
                analysis_controller.py
                visualization_controller.py
                measurements_controller.py
                log_controller.py
            utils/
                layer_utils.py        ← napari layer helpers
                export_utils.py       ← figure and table export
                coord_utils.py        ← (row,col) ↔ (y,x) translations
        napari.yaml                   ← plugin manifest (napari ≥ 0.4.17)
    tests/
        conftest.py                   ← make_napari_viewer fixture
        test_project_widget.py
        test_roi_controller.py
        test_analysis_controller.py
        test_visualization_controller.py
```

---

## Guiding Principles / UX Philosophy

1. **Workflow-ordered tabs.** The tab bar follows the natural analysis sequence
   so users rarely need to go "backwards": Project → Image → ROI Manager →
   Analysis → Results → Log.
2. **Each step is independent.** Every analysis sub-widget can run standalone.
   No step requires another to have been run first; inputs can be loaded from
   disk at any point.
3. **ROI Manager is optional.** No pipeline requires the ROI Manager to be
   populated. Analysis outputs are written there only when the user explicitly
   opts in.
4. **Results in `PluginState` first, hierarchy second.** Analysis results are
   stored as raw objects in `PluginState` immediately on completion. They are
   attached to `TMEHierarchy` only when the user presses "Commit to Hierarchy",
   making the attachment explicit and reversible.
5. **Three-tier visualization.** Spatial overlays go in napari layers. Summary
   plots (heatmaps, histograms, charts) go in a docked `napari-matplotlib`
   panel. Exported PNGs can be opened in a lightweight `QDialog` viewer.
6. **No analysis logic in the plugin.** All computation belongs in `tme_quant`.
   If it needs a new library function, add it there.

---

## Tab Bar Layout

The plugin presents a **flat top-level tab bar** with six tabs. The
**Analysis** tab has its own internal sub-tabs:

```
┌─────────┬───────┬─────────────┬──────────┬─────────┬─────┐
│ Project │ Image │ ROI Manager │ Analysis │ Results │ Log │
└─────────┴───────┴─────────────┴──────────┴─────────┴─────┘

Analysis sub-tabs:
┌──────────────┬───────┬──────┬─────┐
│ Registration │ Fiber │ Cell │ TME │
└──────────────┴───────┴──────┴─────┘
```

All 11 widgets are independently registered in `napari.yaml` so expert users
can float any panel as a standalone dock widget. The `TMEQuantDockWidget`
(`_main_widget.py`) is the primary entry point that assembles them into the
tab bar via a `QTabWidget`.

---

## Widget Inventory

### Tab 1 — Project

**File:** `widgets/project_widget.py`

Manages the image list and image pairing for the entire project.

**Image type assignment.** When a user adds an image, a type-selector dialog
appears with auto-pre-selection based on filename keywords:

| Type | Auto-detected from |
|------|--------------------|
| `Fiber` | filename contains `SHG`, `shg`, `collagen` |
| `Cell` | filename contains `HE`, `he`, `DAPI`, `dapi`, `DAB` |
| `Mask` | filename contains `mask`, `boundary`, `annotation` |
| `Pre-registered 2-channel` | filename contains `2ch`, `merged`, `combined` |
| `Unknown` | fallback |

User confirms or changes the type before the image is added to the project.

**Image pairing.** Fiber and cell images are paired by matching filename stems
across folders (e.g. `SHG/SHG_001.tif` auto-pairs with `HE/HE_001.tif`).
The Project Widget shows a pairing table; the user can override any auto-pair
or leave an image unpaired (valid for pure-fiber or pure-cell pipelines).

```
┌───────────────────┬──────────────┬──────────────┬───────────┐
│ File              │ Type         │ Paired With  │ Status    │
├───────────────────┼──────────────┼──────────────┼───────────┤
│ SHG_001.tif       │ Fiber        │ HE_001.tif ✓ │ ● active  │
│ HE_001.tif        │ Cell         │ SHG_001.tif  │           │
│ SHG_001_2ch.tif   │ 2-channel    │ —            │ merged    │
│ tumor_mask.tif    │ Mask         │ SHG_001.tif  │           │
│ SHG_002.tif       │ Fiber        │ — (none)     │           │
└───────────────────┴──────────────┴──────────────┴───────────┘
[ + Add Image... ]   [ Auto-Pair ]   [ Merge to 2-channel ]
```

- **"Merge to 2-channel"** (on-demand): creates a new project entry
  (ch1 = fiber, ch2 = cell); does not remove the originals. Requires a
  confirmed pair. If the images are not yet registered, the button is
  greyed out with tooltip "Register first in Analysis → Registration".
- **"Register & Merge"** lives in the Registration sub-tab; on completion
  it adds the merged 2-channel image to the project automatically.
- Selecting an image here emits `image_selected(image_id)`, which triggers
  `VisualizationController` to show only layers with the matching
  `[image_id] ::` prefix in their name.

---

### Tab 2 — Image

**File:** `widgets/image_widget.py`

Details and controls for the currently selected image.

- Channel names and roles (editable)
- Pixel size (µm/px) — pre-filled from TIFF metadata if available
- Z-slice navigator and max-projection toggle — **hidden** unless image is 3D
- Time-point slider — **hidden** unless image has a T dimension
- "Set as active fiber image" / "Set as active cell image" shortcuts

The widget reads `PluginState.active_image_id` and updates whenever
`image_selected` fires.

---

### Tab 3 — ROI Manager

**File:** `widgets/roi_manager_widget.py`

The ROI Manager is the spatial annotation hub. It is **optional** for all
pipelines — no analysis step requires it to be populated.

**ROI list.** Each row shows:

```
◎ [Name          ] [Type ▾   ] [Source image ▾] [Scope ▾     ]
  Tumor_Boundary_1  generic    SHG_001.tif       This image
  Stroma_Zone_1     stroma     SHG_001.tif       All images
```

- **Type** is optional (generic by default). Options: `tumor` / `fiber` /
  `stroma` / `cell` / `generic`. Setting a type changes which "detect
  associated objects" shortcuts are shown in the context menu.
- **Scope**: "This image" (default) or "All images". "All images" copies the
  ROI geometry to every `ImageEntry`'s `ROIManager`; per-image offsets are
  allowed.

**Toolbar buttons:**
- `[ + Draw ROI ]` — activates napari Shapes layer for drawing; on commit
  the shape is added to the list. `ROIController` debounces napari
  `layer.events.data` (300 ms) before syncing to the library's `ROIManager`.
- `[ Import from file ]` — loads GeoJSON / mask TIFF / CSV annotation files
  directly into the ROI list without running any analysis.
- `[ Refresh from Analysis ]` — pulls auto-detected boundaries (tumor regions,
  etc.) from `PluginState` into the ROI list. Only available when a matching
  analysis result exists in memory.

**Context menu** (right-click on a row):
- *Detect fibers in this ROI* → runs Fiber Analysis scoped to this ROI
- *Detect cells in this ROI* → runs Cell Analysis scoped to this ROI
- *Run interaction analysis for this ROI* → pre-fills TME sub-tab with this
  ROI as the analysis boundary and switches to Analysis → TME
- *Promote TMEObject to ROI* → converts a selected object from the Measurements
  tree into an ROI boundary
- *Apply to all images* → copies this ROI to all open `ImageEntry` nodes

**napari ↔ ROI sync:** `ROIController` is the single source of truth for both
the widget list and the napari Shapes layer. Both views observe the same
controller signals; there is no second ROI Manager instance.

---

### Tab 4 — Analysis

Internal `QTabWidget` with four sub-tabs. Each sub-tab follows this layout:

```
▼ [Method / Section name]          [status chip]
  Preset:  [Name  ▾]   [ Load ]   [ Save ]
  ─────────────────────────────────────────────
  [~8 key parameters inline]
                          [ Advanced Settings... ]
  ─────────────────────────────────────────────
  Input:  [image or result selector]
          ○ not run   ◑ cached (disk)   ● in memory
          [ Load from disk ]
  ─────────────────────────────────────────────
  ☐ Write results to ROI Manager
  ─────────────────────────────────────────────
  [ Run ]        [ Commit to Hierarchy ]
```

**Status chips:** `○ not run` / `◑ cached (disk)` / `● computed (memory)`.
`AnalysisController` checks `PluginState` before running: if a result is in
memory, it is used directly; if only a saved file exists, the user is prompted
to load it; otherwise the step runs fresh.

**"Advanced Settings..." dialog.** Opens a `QDialog` with a `QFormLayout`
containing all remaining parameters, organised into collapsible groups. This
pattern applies to every method that has more parameters than fit inline
(CT-FIRE has ~30 total; ~8 inline, ~22 in the dialog). Tooltips on every field.

**"Commit to Hierarchy" button.** Attaches the step's raw result from
`PluginState` to `TMEHierarchy` under the correct `ImageEntry` node by calling
`hierarchy.attach_fiber_result()`, `hierarchy.attach_cell_result()`, or
`hierarchy.add_object()`. Emits `committed_to_hierarchy(image_id, object_type)`.
Nothing is attached until this button is pressed.

#### Registration sub-tab

**File:** `widgets/registration_widget.py`

- Input: a fiber image and a cell image (or any two images to align)
- Method: `[Affine ▾]` / Rigid / Deformable (maps to `TransformType` enum)
- Key params inline: pyramid levels, iterations, multiresolution toggle
- No "Write to ROI Manager" checkbox (registration produces an image, not an
  annotation)
- Output options:
  - "Add registered image to project" (adds the warped image as a new
    `Cell` type entry)
  - "Register & Merge into 2-channel" (adds a merged `Pre-registered 2-channel`
    entry to the project)
- Covers workflow step 2: H&E → SHG registration via `HESHGRegistration`

#### Fiber Analysis sub-tab

**File:** `widgets/fiber_analysis_widget.py`

- Input image: locked to `ImageType.FIBER` or `ImageType.TWO_CHANNEL` (ch1).
  Selector shows only valid images from the project.
- Method selector: `CT-FIRE | CurveAlign | Skeleton | Ridge Detection`
- **CT-FIRE inline params (~8):** threshold, min/max fiber length, pixel size,
  spur prune length, mask closing radius, n levels, n angles
- **CT-FIRE Advanced dialog groups:** Transform, Tracing, Filters, Measurements
- 3D CT-FIRE toggle: disabled (greyed) until `ctfire_backend_status()` returns
  `cpp_available=True` and `3d_supported=True`
- Covers workflow step 3: CT-FIRE fiber extraction

#### Cell Analysis sub-tab

**File:** `widgets/cell_analysis_widget.py`

- Input image: locked to `ImageType.CELL` or `ImageType.TWO_CHANNEL` (ch2)
- Method selector: `StarDist | Cellpose | Threshold | Watershed`
- **StarDist inline params (~8):** model name, prob threshold, NMS threshold,
  pixel size, min/max cell size
- Includes a **Tumor Boundary Detection** section (collapsible):
  - Method: `DBSCAN | Convex Hull | Manual mask`
  - Key params: eps, min samples, min tumor area, smooth boundary toggle
  - "Run Tumor Detection" button (independent from cell segmentation)
  - "Write tumor boundaries to ROI Manager" checkbox (off by default)
- Covers workflow steps 4 (cell segmentation) and 5 (tumor detection)

#### TME Analysis Pipelines sub-tab

**File:** `widgets/tme_pipeline_widget.py`

The hub for all relational multi-object analysis. Interaction detection logic
lives **here**, not in the ROI Manager.

- **Mode toggle:** `◉ Single Image  ○ Batch` (Batch greyed in v1; architecture
  in place via `AnalysisController.run_batch()` stub)
- **Recipe system:** `[ Load JSON ]  [ Save JSON ]` — serialises/deserialises
  the current `TMEAnalysisParams` dataclass. One JSON recipe per pipeline
  configuration.
- Three collapsible sections:

  **Interactions section** (covers workflow step 6 + 7):
  ```
  ☑ Cell–Fiber   ☑ Tumor–Fiber   ☐ Fiber–Fiber
  Zone width: [100 µm]   Contact threshold: [5 µm]
  ☑ Compute TACS   Straightness threshold: [0.7]
  ```

  **Per-fiber Metrics section** (covers workflow step 8):
  ```
  K neighbours: [10]   BBox size: [100 µm]
  ☑ Alignment   ☑ Density   ☑ Distance to tumor
  ```

  **Network Analysis section** (covers workflow step 9):
  ```
  Weight by: [distance ▾]   Community: [greedy ▾]
  Top N hubs: [10]
  ```

- **Input selectors** (all independent; each can load from memory or disk):
  - Fiber result: `● computed (memory)  [ Load from disk ]`
  - Cell result: `◑ cached (disk)  [ Load from disk ]`
  - Tumor mask / boundaries: `[ From ROI Manager ]  [ Load mask file ]`
- **"Run Full Pipeline"** runs all enabled sections in sequence.
- **"Run Interaction Analysis for ROI"** — pre-filled by the ROI Manager
  context-menu shortcut; runs the pipeline scoped to the selected ROI.

---

### Tab 5 — Results

#### Measurements sub-widget

**File:** `widgets/measurements_widget.py`

Two-panel layout: hierarchy tree on the left, data table on the right.

```
┌─────────────────────────┬──────────────────────────────┐
│ Hierarchy               │ Table                        │
│ ▼ Project               │ ┌──────────┬────────┬──────┐ │
│   ▼ SHG_001 (ImageEntry)│ │ fiber_id │ length │ tacs │ │
│     ● Fibers (277)      │ │ f_001    │ 45.2   │ 3    │ │
│     ● Cells (1204)      │ │ f_002    │ 31.7   │ 2    │ │
│     ▼ TumorRegion_001   │ └──────────┴────────┴──────┘ │
│       ● TACS zone       │ Filters:                      │
│   ▷ SHG_002 (ImageEntry)│ Type [Fiber ▾] TACS [3 ▾]   │
│                         │ ☐ Global query (all images)  │
│                         │ [ Export CSV ] [ Export XLSX ]│
└─────────────────────────┴──────────────────────────────┘
```

- Clicking a hierarchy node filters the table to that node's descendants.
- Dropdowns refine within the current scope: object type / TACS type / ROI.
- "Global query" toggle: default is per-image scope; when enabled, the query
  runs across all `ImageEntry` nodes in the hierarchy.
- Selecting a table row highlights the corresponding napari layer (bidirectional
  with `VisualizationController` via `measurement_row_selected` signal).
- Only objects committed to the hierarchy appear in the tree. Raw results in
  `PluginState` are not shown here.

#### Visualization sub-widget

**File:** `widgets/visualization_widget.py`

Three-tier output strategy:

| Output type | Display | Controls |
|-------------|---------|----------|
| Fiber centerlines, ROI boundaries, TACS zones | napari `Shapes` / `Points` layers | Overlay checkboxes |
| Heatmaps, histograms, TACS charts, network graphs | Docked `napari-matplotlib` panel (auto-shown on `analysis_complete`) | Plot selector dropdown |
| Exported PNGs (`generate_fiber_overlay`, `generate_fiber_heatmap`) | `QDialog` with `QLabel` on demand | "View in window" button |

Controls:
```
Overlays:   ☑ Fibers  ☑ ROIs  ☑ TACS zones  ☐ Cells
Color by:   [TACS type ▾]
────────────────────────────────────────────────
Plots (docked napari-matplotlib):
  [ Orientation Heatmap   ▾ ] [ Show ]
  [ TACS Distribution     ▾ ] [ Show ]
  [ Angle Histogram       ▾ ] [ Show ]
  [ Network Graph         ▾ ] [ Show ]
────────────────────────────────────────────────
Exported figures:
  [ View Overlay PNG in window    ]
  [ View Density Heatmap in window]
```

`VisualizationController` manages all layer visibility. It never removes layers;
it toggles `layer.visible`. A `ColorStrategy` enum controls layer colour:
`BY_OBJECT_TYPE | BY_TACS_TYPE | BY_ROI | UNIFORM`.

#### I/O sub-widget

**File:** `widgets/io_widget.py`

```
[ Load Project ]         [ Save Project ]
[ Import QuPath GeoJSON ]
[ Export GeoJSON ]       (for QuPath annotation import)
[ Export Fiber Metrics CSV / Excel ]
[ Export TME Analysis JSON ]
[ Load Parameter Preset ]  [ Save Parameter Preset ]
```

Project save/load uses `tme_quant.core.io.save_project` /
`load_project`. GeoJSON export uses the planned `qupath_bridge.py`
integration. Parameter presets are plain JSON (serialised dataclasses via
`.to_dict()`).

---

### Tab 6 — Log

**File:** `widgets/log_widget.py`

```
[ ■ Clear ]
────────────────────────────────────────────
[INFO]  CT-FIRE backend: Python (C++ not built)
[INFO]  Loaded SHG_001.tif  512×512  1.0 µm/px
[INFO]  Registration MI: 0.4832
[INFO]  Extracted 277 fibers
[WARN]  No calibration metadata — pixel size defaulted to 1.0 µm/px
[INFO]  Segmented 1,204 cells
[INFO]  2 tumor regions detected
[INFO]  Pipeline complete  14.3 s
```

`LogController` subscribes to all `analysis_complete` and `batch_progress`
signals. The log is a `QTextEdit` in read-only mode; ANSI colour codes map to
`INFO` (black) / `WARN` (orange) / `ERROR` (red).

---

## Architecture

### Three-layer separation

```
Widget layer       napari_tme_quant/widgets/
                   Pure Qt UI. No direct tme_quant calls.
                       ↕  (Qt signals / slots)
Controller layer   napari_tme_quant/controllers/
                   Translates widget state → tme_quant API calls.
                   Owns threading. Mutates PluginState on Qt main thread only.
                       ↕  (function calls)
Library layer      tme_quant/
                   All analysis logic. Zero knowledge of the plugin.
```

Never put analysis logic in widgets or controllers. If a computation belongs in
the library, add it there first.

### Threading rule

All `tme_quant` calls run in a `@thread_worker`; `PluginState` is mutated only
in the `returned` signal callback (Qt main thread):

```python
from napari.qt.threading import thread_worker

@thread_worker(connect={"returned": self._on_fiber_result, "errored": self._on_error})
def _run_fiber_extraction(self, image: np.ndarray, params: CTFireParams):
    return FiberAnalyzer().extract_2d(image, params)

def _on_fiber_result(self, result: FiberAnalysisResult) -> None:
    # Called on Qt main thread — safe to mutate PluginState and update UI
    self._state.fiber_results[self._state.active_image_id] = result
    self._state.emit("analysis_complete", step="fiber", result=result)
```

**GIL hazard:** C++ extensions (CT-FIRE via pybind11) must use
`py::gil_scoped_release`. Pure-Python loops over thousands of objects should
yield occasionally (`time.sleep(0.001)`) to keep the Qt event loop responsive.

### State management

`PluginState` (`controllers/state.py`) is the single source of truth:

```python
@dataclass
class PluginState:
    project: Optional[TMEProject] = None
    hierarchy: TMEHierarchy = field(default_factory=TMEHierarchy)

    # Per-image raw results (before hierarchy commit)
    fiber_results: Dict[str, FiberAnalysisResult] = field(default_factory=dict)
    cell_results:  Dict[str, CellAnalysisResult]  = field(default_factory=dict)
    tme_results:   Dict[str, TMEAnalysisResult]   = field(default_factory=dict)

    # Image metadata
    image_pairs:   Dict[str, str]    = field(default_factory=dict)  # fiber_id → cell_id
    image_types:   Dict[str, ImageType] = field(default_factory=dict)
    active_image_id: Optional[str]   = None

    # napari layer map: layer_name → TMEObject.object_id
    layer_map: Dict[str, str] = field(default_factory=dict)

    # Active parameter presets (one per analysis step)
    presets: Dict[str, dict] = field(default_factory=dict)
```

Widgets read and write state **only through controllers**. No widget holds a
reference to `PluginState` directly.

### Data flow: raw results → hierarchy → napari layers

```
AnalysisController._on_fiber_result()
    → PluginState.fiber_results[image_id] = result   # step A: store raw
    → emit analysis_complete("fiber", image_id)

[User clicks "Commit to Hierarchy"]
    → AnalysisController.commit_fiber_result(image_id)
        → hierarchy.attach_fiber_result(image_entry, result)  # step B: attach
        → emit committed_to_hierarchy(image_id, "fiber")

VisualizationController.on_committed(image_id, "fiber")
    → create napari Shapes layer named "SHG_001 :: Fibers :: all"  # step C: visualise
    → PluginState.layer_map["SHG_001 :: Fibers :: all"] = image_id
    → for each TACS type present, create filtered layer
        "SHG_001 :: Fibers :: TACS-3"

MeasurementsController.on_committed(image_id, "fiber")
    → refresh hierarchy tree                         # step D: results tab
    → refresh table to show fiber rows
```

Nothing in step B or C happens without the user pressing "Commit to Hierarchy".

### napari layer ↔ TMEHierarchy sync rules

1. **Layer naming:** `[image_id] :: [ObjectType] :: [detail]`
   e.g. `SHG_001 :: Fibers :: TACS-3`, `HE_001 :: Cells :: StarDist`,
   `SHG_001 :: ROI :: Tumor_Boundary_1`
2. **Layer visibility:** `VisualizationController` toggles `layer.visible`.
   Layers are never removed when switching images.
3. **Layer creation:** only on "Commit to Hierarchy". Raw results in
   `PluginState` do not produce layers.
4. **Bidirectional selection:** clicking a layer highlights the matching
   Measurements table row (via `layer_selected` → `measurement_row_selected`),
   and vice versa.
5. **Coordinate translation:** `coord_utils.py` handles `(row, col)` ↔ `(y, x)`
   conversions at the controller boundary. The library always uses `(row, col)`.
   napari uses `(y, x)` for 2D, `(z, y, x)` for 3D.

### ROI Manager sync rules

- `ROIController` is the single source of truth for both the ROI Manager widget
  and the napari Shapes layer.
- napari `layer.events.data` is debounced (300 ms) before `ROIController` syncs
  to `tme_quant.ROIManager`.
- When "Refresh from Analysis" is pressed, `ROIController` reads the relevant
  objects from `PluginState` (not from the hierarchy, which may not yet be
  committed) and adds them to the ROI list.

### Signal / event table

| Signal | Emitted by | Handled by |
|--------|-----------|------------|
| `image_selected(image_id)` | `ProjectController` | `VisualizationController` (show layers with `[image_id] ::` prefix), `ROIController` (filter ROI list to image) |
| `image_type_changed(image_id, type)` | `ProjectController` | `AnalysisController` (update input image selectors) |
| `pair_changed(fiber_id, cell_id)` | `ProjectController` | `AnalysisController` (invalidate cached results for that pair) |
| `roi_drawn(roi_id)` | napari Shapes callback (debounced 300 ms) | `ROIController` (sync to `tme_quant.ROIManager`) |
| `roi_changed(roi_id)` | `ROIController` | `AnalysisController` (re-run scoped step if auto-run enabled) |
| `analysis_complete(step, image_id, result)` | `AnalysisController` | `PluginState` (store raw result), `LogController` (log message), status chip update on widget |
| `committed_to_hierarchy(image_id, obj_type)` | `AnalysisController` | `MeasurementsController` (refresh tree + table), `VisualizationController` (create/update layers) |
| `layer_selected(layer_name)` | napari layer click | `MeasurementsController` (highlight table row) |
| `measurement_row_selected(object_id)` | `MeasurementsController` | `VisualizationController` (highlight layer) |
| `global_query_toggled(enabled)` | `MeasurementsController` | `MeasurementsController` (re-run query across all images vs. active image) |
| `batch_progress(step, n, total)` | `AnalysisController` | `LogController` (update progress bar in Log tab) |

---

## Workflow Step Coverage

The following maps all 12 steps of `example_ctfire_workflow_hierarchy.py` to
the widget(s) that provide them:

| Workflow step | Widget(s) |
|---------------|-----------|
| 1. Load H&E and SHG images | Project → Add Image |
| 2. Register H&E → SHG | Analysis → Registration sub-tab |
| 3. CT-FIRE fiber extraction | Analysis → Fiber sub-tab |
| 4. Cell segmentation (StarDist) | Analysis → Cell sub-tab |
| 5. Tumor boundary detection (DBSCAN) | Analysis → Cell sub-tab (Tumor section) |
| 6. TME analysis (fiber–tumor interactions) | Analysis → TME sub-tab (Interactions section) |
| 7. InteractionAnalysisPipeline | Analysis → TME sub-tab (Interactions section) |
| 8. Per-fiber metrics (K-NN alignment, density) | Analysis → TME sub-tab (Per-fiber Metrics section) |
| 9. Network analysis | Analysis → TME sub-tab (Network Analysis section) |
| 10. Generate heatmaps | Results → Visualization (docked matplotlib + "View in window") |
| 11. Create TACS overlay | Results → Visualization (napari Shapes layers) |
| 12. Export results | Results → I/O |

---

## 3D / Dynamic Extensibility Hooks

The v1 plugin is **pure 2D**. The following hooks are in place so that
adding 3D and time-series support does not require architectural changes:

- **Image Widget:** Z-slice navigator and T-slider are present in the widget
  code but hidden (`widget.setVisible(False)`) unless `image.ndim >= 3` or
  `image.n_timepoints > 1`. Unhide them when 3D/4D support is added.
- **Controller interfaces:** `AnalysisController._run_fiber_extraction` accepts
  an `ndim: int` parameter and routes to `extract_2d` or `extract_3d` based on
  its value. In v1, `ndim=2` always. The routing logic is already in place.
- **3D CT-FIRE gate:** the 3D CT-FIRE toggle in the Fiber sub-tab reads
  `ctfire_backend_status()["3d_supported"]` and stays disabled until the C++
  3D FIRE extension is compiled.
- **Hierarchy:** `TMEHierarchy` and `TMEObject` already support 3D coordinates
  (centerline shape `(N, 3)`). No plugin changes needed for hierarchy storage.
- **Napari layers:** `VisualizationController` checks `ndim` before creating
  layers; 3D fibers will be rendered as `Tracks` or `Shapes` layers with Z
  coordinates when the time comes.

---

## File Naming Conventions

| Path | Contents |
|------|----------|
| `_main_widget.py` | `TMEQuantDockWidget` — top-level `QTabWidget` container |
| `widgets/project_widget.py` | Project tab: image list, type, pairing |
| `widgets/image_widget.py` | Image tab: active image details, channel mapping |
| `widgets/roi_manager_widget.py` | ROI Manager tab: annotations, context menus |
| `widgets/registration_widget.py` | Analysis → Registration sub-tab |
| `widgets/fiber_analysis_widget.py` | Analysis → Fiber sub-tab |
| `widgets/cell_analysis_widget.py` | Analysis → Cell sub-tab (incl. tumor detection) |
| `widgets/tme_pipeline_widget.py` | Analysis → TME sub-tab |
| `widgets/measurements_widget.py` | Results: hierarchy tree + data table |
| `widgets/visualization_widget.py` | Results: overlay controls + docked plots |
| `widgets/io_widget.py` | Results: project/parameter import–export |
| `widgets/log_widget.py` | Log tab |
| `controllers/state.py` | `PluginState` dataclass + `ImageType` enum |
| `controllers/project_controller.py` | Image list, pairing, type assignment |
| `controllers/roi_controller.py` | ROI Manager ↔ `tme_quant.ROIManager` sync |
| `controllers/analysis_controller.py` | All analysis step dispatch + threading |
| `controllers/visualization_controller.py` | Layer visibility, naming, colour |
| `controllers/measurements_controller.py` | Hierarchy tree + table queries |
| `controllers/log_controller.py` | Log message routing |
| `utils/layer_utils.py` | Layer name construction / parsing helpers |
| `utils/export_utils.py` | Figure `QDialog` viewer, CSV/Excel export wrappers |
| `utils/coord_utils.py` | `(row,col)` ↔ `(y,x)` coordinate translation |

---

## napari Plugin Registration

All 11 widgets are registered individually in `napari.yaml` so expert users
can open any panel as a standalone floating dock widget. The `TMEQuantDockWidget`
container is also registered as the primary entry point.

```yaml
name: napari-tme-quant
display_name: TMEQuant
contributions:
  commands:
    - id: napari-tme-quant.main_widget
      title: TMEQuant (all panels)
      python_name: napari_tme_quant._main_widget:TMEQuantDockWidget
    - id: napari-tme-quant.project_widget
      title: TMEQuant — Project
      python_name: napari_tme_quant.widgets.project_widget:ProjectWidget
    - id: napari-tme-quant.image_widget
      title: TMEQuant — Image
      python_name: napari_tme_quant.widgets.image_widget:ImageWidget
    - id: napari-tme-quant.roi_manager_widget
      title: TMEQuant — ROI Manager
      python_name: napari_tme_quant.widgets.roi_manager_widget:ROIManagerWidget
    - id: napari-tme-quant.registration_widget
      title: TMEQuant — Registration
      python_name: napari_tme_quant.widgets.registration_widget:RegistrationWidget
    - id: napari-tme-quant.fiber_analysis_widget
      title: TMEQuant — Fiber Analysis
      python_name: napari_tme_quant.widgets.fiber_analysis_widget:FiberAnalysisWidget
    - id: napari-tme-quant.cell_analysis_widget
      title: TMEQuant — Cell Analysis
      python_name: napari_tme_quant.widgets.cell_analysis_widget:CellAnalysisWidget
    - id: napari-tme-quant.tme_pipeline_widget
      title: TMEQuant — TME Pipelines
      python_name: napari_tme_quant.widgets.tme_pipeline_widget:TMEPipelineWidget
    - id: napari-tme-quant.measurements_widget
      title: TMEQuant — Measurements
      python_name: napari_tme_quant.widgets.measurements_widget:MeasurementsWidget
    - id: napari-tme-quant.visualization_widget
      title: TMEQuant — Visualization
      python_name: napari_tme_quant.widgets.visualization_widget:VisualizationWidget
    - id: napari-tme-quant.io_widget
      title: TMEQuant — I/O
      python_name: napari_tme_quant.widgets.io_widget:IOWidget
    - id: napari-tme-quant.log_widget
      title: TMEQuant — Log
      python_name: napari_tme_quant.widgets.log_widget:LogWidget
  widgets:
    - command: napari-tme-quant.main_widget
      display_name: TMEQuant
      autogenerate: false
    - command: napari-tme-quant.project_widget
      display_name: TMEQuant — Project
      autogenerate: false
    - command: napari-tme-quant.roi_manager_widget
      display_name: TMEQuant — ROI Manager
      autogenerate: false
    - command: napari-tme-quant.measurements_widget
      display_name: TMEQuant — Measurements
      autogenerate: false
    - command: napari-tme-quant.log_widget
      display_name: TMEQuant — Log
      autogenerate: false
    # ... (register all remaining 6 widgets the same way)
```

---

## Dependencies

`napari-tme-quant/pyproject.toml` declares only GUI-specific deps:

```toml
dependencies = [
    "tme-quant>=0.2.0",       # all analysis deps inherited transitively
    "napari>=0.4.17",
    "napari-matplotlib>=0.2", # docked non-spatial plots
    "qtpy>=2.3.0",
]
```

Do not duplicate any analysis dependency here.

---

## Development Setup

```bash
pip install -e "../tme-quant[dev]"
pip install -e ".[dev]"
napari   # plugin loads automatically from editable install
```

Widget tests use napari's `make_napari_viewer` fixture (headless via `pytest-qt`).

---

## Key Rules

1. **Analysis logic belongs in `tme_quant`, not in the plugin.** Add library
   functions first; call them from controllers.
2. **All analysis runs in `@thread_worker`.** `PluginState` is mutated only in
   the `returned` callback on the Qt main thread.
3. **Raw results live in `PluginState`; hierarchy attachment is on-demand.**
   Nothing is attached to `TMEHierarchy` until "Commit to Hierarchy" is pressed.
4. **Napari layers are created only in `VisualizationController`**, never in
   widgets or analysis controllers.
5. **Layer names follow `[image_id] :: [ObjectType] :: [detail]`.**
   `PluginState.layer_map` is the only source of truth for layer ↔ object mapping.
6. **ROI Manager is optional for all pipelines.** "Write results to ROI Manager"
   is off by default on every analysis step.
7. **All 11 sub-widgets are independently registered in `napari.yaml`.**
8. **Coordinate translation is `coord_utils.py`'s job.** The library uses
   `(row, col)`; napari uses `(y, x)` / `(z, y, x)`. Never translate inline.
9. **`TMEHierarchy` is persisted via `save_project` / `load_project`.**
   GeoJSON export for QuPath is a separate I/O action.
10. **Disable 3D CT-FIRE until `ctfire_backend_status()["3d_supported"]` is
    `True`.** All other 3D hooks are present but hidden until needed.
11. **Do not put interaction detection logic in the ROI Manager.** The ROI
    Manager may have a context-menu shortcut that pre-fills and switches to the
    TME Pipelines sub-tab, but the logic runs there.