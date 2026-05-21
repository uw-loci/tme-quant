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
as a library. For prototyping it is co-located inside the core library's source
tree (tracked in the same git repo):

```
tme-quant/
└── src/
    └── tme_quant/            ← core library
        ├── core/  fiber_analysis/  ...
        └── napari-tme-quant/ ← this plugin (separate installable package)
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

Manages the image list, type assignment, and project save/load operations.

**`ImageType.FIBER` covers any microscopic fiber image** — not just SHG. This
includes birefringent images from polarised-light microscopes (PLM), bright-field
tissue images with collagen-specific stains, and any other modality where local
fiber orientations are the analytical target.

**Supported file formats:** TIFF (default), PNG, JPEG.
File dialog filter: `Images (*.tif *.tiff *.png *.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*)`.

**Image type assignment.** When a user adds an image, a type-selector dialog
appears with auto-pre-selection based on filename keywords (case-insensitive):

| Type | Auto-detected from filename keywords |
|------|--------------------------------------|
| `Fiber` | `shg`, `collagen`, `fiber`, `fibre`, `biref`, `plm`, `brightfield` |
| `Cell` | `he`, `dapi`, `dab`, `cell`, `nuclei` |
| `Mask` | `mask`, `boundary`, `annotation`, `label`, `seg` |
| `Pre-registered 2-channel` | `2ch`, `merged`, `combined` |
| `Unknown` | fallback |

User confirms or changes the type before the image is added to the project.
`ProjectController.add_image(path, image_type, viewer)` loads the file, adds
it as a napari layer, stores the array in `PluginState.images`, and records the
absolute path in `PluginState.image_paths` for save/restore.

**Mask provenance note.** In the full plugin the ROI Manager is the hub for
boundary masks: they can be loaded from file, drawn manually in a napari Shapes
or Labels layer, or auto-detected from the Cell Analysis tumor-detection step.
In the current prototype:
- Add mask TIFFs via `[ + Add Image... ]` → type = Mask
- Or draw / import a Shapes/Labels layer directly in napari; the CurveAlign TACS
  pipeline mask selector lists **all napari layers** (not just project-registered ones)

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

- **"Merge to 2-channel"** — future batch; greyed out in v1.
- **"Register & Merge"** lives in the Registration sub-tab; on completion
  it adds the merged 2-channel image to the project automatically.
- Selecting an image emits `image_selected(image_id)`, which triggers
  `VisualizationController.on_image_selected()` to show only layers with the
  matching `[image_id] ::` prefix in their name.
- Double-clicking a row opens a type-change dialog.

**Project operations** (bottom of the Project tab):
```
[ New ]   [ Save Project... ]   [ Load Project... ]
```
- `[ New ]` — clears all project state and napari layers after confirmation.
- `[ Save Project... ]` — selects a save directory → calls
  `io_utils.save_plugin_state(state, save_dir)` which writes:
  - `plugin_state.json` — image types, paths, per-image params
  - `results/<image_id>/` — `fiber_features.csv`, `roi_summary.csv`,
    `fiber_structure.csv`, numpy arrays, `params.json`
- `[ Load Project... ]` — selects a directory → calls
  `io_utils.load_plugin_state(state, load_dir, viewer)` which restores
  metadata, reloads image files as napari layers, and reconstructs
  `CurveAlignPipelineResult` objects from the saved CSVs/npy files.

---

### Tab 2 — Image

**File:** `widgets/image_widget.py`

Details, controls, and analysis status for the currently selected image.

```
┌─ Image metadata ────────────────────────────────────────────┐
│  File:        fiber_001                                     │
│  Path:        /data/SHG_001.tif                             │
│  Dimensions:  512 × 512  (float32)                          │
│  Type:        Fiber                                         │
│  Pixel size:  [ 1.0 ▴▾ ] µm/px                              │
└─────────────────────────────────────────────────────────────┘
┌─ Analysis status ───────────────────────────────────────────┐
│  CT-FIRE:         ○ not run                                 │
│  CurveAlign TACS: ● computed (memory)                       │
└─────────────────────────────────────────────────────────────┘
```

- **Pixel size** is editable; stored in
  `PluginState.per_image_params[image_id]["image"]["pixel_size"]`.
- **Status chips** refresh on every `analysis_complete` signal.
- Z-slice navigator and T-slider are present in code but hidden unless
  `image.ndim >= 3` or `image.n_timepoints > 1` (future 3D/4D support).
- The widget reads `PluginState.active_image_id` and updates whenever
  `image_selected` fires from `ProjectController`.

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
- Method selector: `CT-FIRE | CurveAlign (curvelets mode)`
  Switching the selector shows/hides the matching `QGroupBox` of controls.

##### CT-FIRE section (default)

- **Inline params (~8):** threshold, min/max fiber length, pixel size,
  spur prune length, mask closing radius, n levels, n angles
- **Advanced dialog groups:** Transform / Tracing / Filters / Measurements
- 3D CT-FIRE toggle: disabled (greyed) until `ctfire_backend_status()` returns
  `cpp_available=True` and `3d_supported=True`
- Covers workflow step 3: CT-FIRE fiber extraction

##### CurveAlign (curvelets mode) section

Runs `curvealign_curvelets_mode_pipeline()` and stores a
`CurveAlignPipelineResult` in `PluginState.curvealign_pipeline_results`.

```
┌─ CurveAlign (curvelets mode) ──────────────────────────────────┐
│  ☐ Use pre-computed fiber_structure  [Load CSV/XLSX...]        │
│                                                                │
│  ☐ Analyze boundary alignment                                  │
│     Mask layer: [ Select mask layer ▾ ]                        │
│     Zone width: [50.0  µm]                                     │
│                                                                │
│  ─── Curvelet params ───────────────────────────────────────── │
│  Keep:      [0.05 ]   Scale:  [1   ]   Radius:   [4.0  ]      │
│  Pixel size:[1.0  ]   Dist threshold:  [50.0 ]                 │
│  Min fiber weight: [0.0]   ☐ Exclude fibers inside mask        │
│                                                                │
│                            [ Advanced... ]  [ Run CurveAlign ] │
└────────────────────────────────────────────────────────────────┘
```

- `☐ Use pre-computed fiber_structure` — when checked, the Load button becomes
  active and the resulting DataFrame is passed as `fiber_structure=` to the
  pipeline, bypassing `curvelops` extraction (covers the optional pre-computed
  input pattern, generalisation rule G6).
- `☐ Analyze boundary alignment` — enables mask selector + zone width; when
  unchecked, `boundary_img=None` and `tif_boundary=0` are passed and the TACS
  sub-section in the visualisation panel stays hidden.
- **Advanced dialog groups:** Transform / Boundary / Features / Output
- Covers the CurveAlign pipeline workflow (see workflow table in §Workflow Step
  Coverage below).

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

The hub for all multi-object TME analysis that combines fiber data with tumor
context (boundary, cell populations). Two pipeline families live here:

**CurveAlign TACS Pipeline** (implemented in v1) — combines curvelet-based local
fiber orientation analysis with a pre-computed tumor boundary mask to classify
fibers as TACS-1/2/3. The boundary mask is assumed to have been derived from
cell/tumor image analysis (e.g. DBSCAN tumor boundary detection in the Cell tab).

```
┌─ CurveAlign TACS Pipeline ─────────────────────────────────────┐
│  Fiber image:   [ fiber_001 (Fiber) ▾ ]                        │
│  Boundary mask: [ Select napari layer ▾ ] [ Load from file... ]│
│  Zone width:    [ 50.0 µm ]                                    │
│                                                                │
│  ─── Curvelet params ──────────────────────────────────────    │
│  Keep: [0.05]  Scale: [1]  Radius: [4.0]  Pixel size: [1.0]   │
│  Dist threshold: [50.0]  ☐ Exclude fibers inside mask          │
│                                          [ Advanced... ]       │
│                                                                │
│  Run on:  ◉ Current image  ○ All images in project             │
│  [ Copy params to all images ]                                 │
│  [████████░░]  2/4  Extracting curvelet fiber structure…       │
│                                                                │
│  [ Run CurveAlign TACS ]      [ Commit to Hierarchy ]          │
│  ○ not run                                                     │
└────────────────────────────────────────────────────────────────┘
```

Key behaviours:
- **Fiber image selector** — populated from `PluginState.image_types` filtered to
  `ImageType.FIBER`; auto-selects the current active image when it is a Fiber type.
- **Boundary mask selector** — lists ALL napari layers (Image, Labels, Shapes).
  Shapes layers are rasterised to a binary mask via
  `layer.to_labels(labels_shape=image.shape[:2])`. Also includes project-registered
  MASK images. `[ Load from file... ]` adds a mask file to the project as
  `ImageType.MASK` and auto-selects it.
- **Per-image params** — parameters are stored per image in
  `PluginState.per_image_params[image_id]["curvealign_tacs"]`; automatically
  saved on Run and restored when the active image changes.
- **Progress bar** — steps through the pipeline's 4 progress messages; hidden at
  rest, shown during the run.
- **tif_boundary** — always passes `tif_boundary=3` to the pipeline when a mask
  is selected (the only supported mode for binary mask input). `tif_boundary=0`
  when no mask is selected.

**Standard TME Pipeline** (stub in v1) — uses CT-FIRE fibers + StarDist cells +
tumor detection; architecture in place via stub sections. Mode toggle, recipe
JSON, and collapsible sections below are future additions.

- **Mode toggle:** `◉ Single Image  ○ Batch` (Batch greyed in v1)
- **Recipe system:** `[ Load JSON ]  [ Save JSON ]` — future addition.
- Three collapsible sections (Standard Pipeline only):

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
| Orientation heatmap (`procmap` from `generate_fiber_heatmap`) | napari `Image` layer `[id] :: Heatmap :: orientation` (inferno, 60 % opacity) — added on Commit | Overlay toggle |
| Heatmaps, histograms, TACS charts, network graphs | Docked `napari-matplotlib` panel | Plot selector dropdown |
| Overlay / heatmap figures (`generate_fiber_overlay`, `generate_fiber_heatmap`) | `FigureDialog` (`QDialog` with `QLabel`) on demand | "View … in window" buttons |

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

#### CurveAlign TACS View

Shown only when `PluginState.curvealign_pipeline_results` is non-empty for the
active image. Docked below the main overlay controls.

```
┌─ CurveAlign TACS View ─────────────────────────────────────────┐
│  ROI:   [ All ROIs              ▾ ]                            │
│  TACS:  [ All zones             ▾ ]                            │
│          All zones                                             │
│          TACS-3 (high alignment)                               │
│          TACS-2 (intermediate)                                 │
│          TACS-1 (low alignment)                                │
│          Outside zone                                          │
│  ☐ Show boundary association lines                             │
│                                                                │
│  ┌─────────┬──────────┬───────────┬───────────┬──────────┐    │
│  │fiber_key│center_row│center_col │abs_angle  │tacs_class│    │
│  ├─────────┼──────────┼───────────┼───────────┼──────────┤    │
│  │    0    │  127.4   │   88.1    │  43.2°    │  TACS-3  │    │
│  │   ...   │   ...    │    ...    │   ...     │   ...    │    │
│  └─────────┴──────────┴───────────┴───────────┴──────────┘    │
│                                                                │
│  ─── Save results ─────────────────────────────────────────    │
│  [ Save fiber features CSV ]  [ Save fiber features XLSX ]     │
│  [ Save overlay PNG ]         [ Save heatmap PNG ]             │
└────────────────────────────────────────────────────────────────┘
```

**ROI filter** — dropdown populated from the `roi_summary_df` index of the
active `CurveAlignPipelineResult`. "All ROIs" passes `roi_id=None` to the
visualisation helper.

**TACS filter** — filters the fiber table and updates the layer immediately.
No re-run of the pipeline.

**Fiber table** — reads `fiber_features_df` columns directly (not
`FiberObject.to_dict()`; the CurveAlign result may not yet be committed to the
hierarchy). Bidirectional row ↔ layer selection applies:
- Clicking a table row selects the corresponding point in the
  `[image_id] :: Fibers :: curvealign` Points layer.
- Clicking a point in the layer scrolls the table to the matching row.

**TACS color coding:**

| TACS class | napari color |
|------------|-------------|
| TACS-3 (high alignment, inside zone) | `red` |
| TACS-2 (intermediate) | `limegreen` |
| TACS-1 (low alignment) | `dodgerblue` |
| Outside zone | `lightgray` |

Color is computed from `in_curvs_flag` + alignment statistics in
`fiber_features_df` using the same thresholds as `_tacs_hierarchy_integration`
in the example script.

**`☐ Show boundary association lines`** — lazily creates a
`[image_id] :: Associations :: boundary` Shapes layer with dashed lines
connecting each fiber centroid to its nearest boundary point
(`boundary_point_col`, `boundary_point_row` columns of `fiber_features_df`).
The layer is hidden (not removed) when the checkbox is unchecked.

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
# controllers/state.py imports
from tme_quant.tme_analysis.pipelines.curvealign_curveletsMode_pipeline import (
    CurveAlignPipelineResult,
)

@dataclass
class PluginState:
    project: Optional[TMEProject] = None
    hierarchy: TMEHierarchy = field(default_factory=TMEHierarchy)

    # Per-image raw results (before hierarchy commit)
    fiber_results: Dict[str, FiberAnalysisResult] = field(default_factory=dict)
    cell_results:  Dict[str, CellAnalysisResult]  = field(default_factory=dict)
    tme_results:   Dict[str, TMEAnalysisResult]   = field(default_factory=dict)

    # CurveAlign full-pipeline results (separate slot — typed result, not FiberAnalysisResult)
    curvealign_pipeline_results: Dict[str, CurveAlignPipelineResult] = field(default_factory=dict)

    # Image metadata
    image_pairs:   Dict[str, str]    = field(default_factory=dict)  # fiber_id → cell_id
    image_types:   Dict[str, ImageType] = field(default_factory=dict)
    active_image_id: Optional[str]   = None

    # napari layer map: layer_name → TMEObject.object_id
    layer_map: Dict[str, str] = field(default_factory=dict)

    # Active parameter presets (one per analysis step)
    presets: Dict[str, dict] = field(default_factory=dict)

    # Raw image arrays keyed by image_id (populated by ProjectController.add_image).
    # Used by VisualizationController for heatmap/overlay generation without
    # re-reading the napari layer.
    images: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-image parameter snapshots: image_id → {step → params_dict}
    # e.g. state.per_image_params["fiber_001"]["curvealign_tacs"] = {"keep": 0.05, ...}
    # Preserved across reset() so re-runs use the same settings.
    per_image_params: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    # Absolute file paths for project save/restore: image_id → path string
    image_paths: Dict[str, str] = field(default_factory=dict)
```

`reset()` clears analysis results and `layer_map` but **preserves** `images`,
`image_types`, `image_paths`, and `per_image_params` so a re-run uses the same
images and settings without needing to reload from disk.

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

## Analysis Widget Generalization Rules

These rules govern how any new analysis method is added to the plugin. They
apply uniformly across all sub-tabs of Tab 4 (Analysis) and ensure that adding
a new method does not require changes to any other layer.

**G1 — Taxonomy: Extraction vs. Pipeline**
Analysis widgets fall into two families:
- *Extraction* widgets (e.g. CT-FIRE, StarDist) call a single-step library
  function and produce one result type (`FiberAnalysisResult`,
  `CellAnalysisResult`). Result stored in `PluginState.fiber_results` or
  `cell_results`.
- *Pipeline* widgets (e.g. CurveAlign curvelets mode) call a multi-step
  library function and produce a composite result type. Result stored in a
  dedicated slot (e.g. `PluginState.curvealign_pipeline_results`). Never mix
  Pipeline results into Extraction slots.

**G2 — Param layout: ≤8 inline + lazy Advanced QDialog**
Each method exposes at most 8 parameters inline (enough to fit the sub-tab
without scrolling at 1080 p). All remaining params live in a `QDialog` opened
by an `[ Advanced... ]` button. The dialog is built lazily on first open.
Group advanced params into named `QGroupBox` sections (e.g.
Transform / Boundary / Features / Output).

**G3 — Method selector → QGroupBox show/hide**
The method `QComboBox` (or `QButtonGroup`) must map each selection to a single
`QGroupBox` that is shown/hidden with `setVisible()`. No other widget outside
that group box changes visibility on method switch.

**G4 — Result Type Map**

| Method | Library call | PluginState slot | Result type |
|--------|-------------|-----------------|-------------|
| CT-FIRE | `run_ctfire_analysis()` | `fiber_results` | `FiberAnalysisResult` |
| CurveAlign (curvelets mode) | `curvealign_curvelets_mode_pipeline()` | `curvealign_pipeline_results` | `CurveAlignPipelineResult` |
| Cell methods (StarDist, etc.) | `run_cell_analysis()` | `cell_results` | `CellAnalysisResult` |
| TME pipeline | `StandardTMEPipeline.run()` | `tme_results` | `TMEAnalysisResult` |

**G5 — `progress_callback` binding pattern**

```python
@thread_worker(connect={"returned": self._on_result, "errored": self._on_error})
def _worker(self, **kwargs):
    def _cb(step, total, msg):
        worker.signals.progressed.emit(int(step / total * 100))
        LogController.append(msg)
    return library_pipeline_function(**kwargs, progress_callback=_cb)
```

The `progress_callback` always has the signature `(step: int, total: int,
msg: str) -> None`. The worker updates both the progress bar and the log from
the same callback.

**G6 — Optional pre-computed input**
When a pipeline accepts a pre-computed intermediate (e.g. `fiber_structure=`
in `curvealign_curvelets_mode_pipeline`), the widget exposes:
```
☐ Use pre-computed <input_name>   [Load CSV/XLSX...]
```
The checkbox enables the Load button; the loaded DataFrame is validated
against the expected schema before passing it to the pipeline.

**G7 — Result adapter: never store raw dicts in PluginState**
If a library function returns a raw `dict`, wrap it in a typed dataclass
before storing (see `CurveAlignPipelineResult`). The PluginState contract
requires that all result slots are typed; untyped dicts in slots are a
maintenance hazard and break downstream serialisation.

**G8 — Visualisation hints per result type**

| Result type | Auto-created layers on commit |
|-------------|-------------------------------|
| `FiberAnalysisResult` | `Shapes` (fibers), `Image` (orientation map, optional) |
| `CurveAlignPipelineResult` | `Points` (fiber centroids), `Image` (density/alignment heatmap), optional TACS color-coded Points |
| `CellAnalysisResult` | `Labels` (cell mask) |
| `TMEAnalysisResult` | `Shapes` (interaction lines), `Labels` (TACS zones) |

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

### CurveAlign TACS Pipeline coverage (Scenario 3)

This pipeline is a **TME analysis** that combines curvelet fiber orientations with
a pre-computed tumor boundary mask. `Run CurveAlign TACS` is in
**Analysis → TME sub-tab → CurveAlign TACS Pipeline section**.

| Pipeline step | progress_callback call | Widget(s) |
|---------------|----------------------|-----------|
| 0. Load fiber image | *(Project tab)* | Project → `[ + Add Image... ]` (type = Fiber) |
| 0. Load / draw boundary mask | *(Project tab or napari)* | Project → `[ + Add Image... ]` (type = Mask) OR draw Shapes/Labels in napari |
| 0. Select active image | *(Project tab)* | Project → click row → `ProjectController.select_image()` |
| 1. Curvelet fiber extraction | `"Extracting curvelet fiber structure…"` | Analysis → TME → CurveAlign TACS (keep, scale, radius params) |
| 2. Density + alignment statistics | *(internal, no progress msg)* | Analysis → TME → CurveAlign TACS (inline params) |
| 3. ROI boundary coordinate extraction | `"Extracting boundary coordinates…"` | Analysis → TME → CurveAlign TACS (boundary mask selector, zone width) |
| 4. Global boundary alignment analysis | `"Computing boundary alignment…"` | Analysis → TME → CurveAlign TACS (dist threshold) |
| 4b. Feature table assembly | `"Assembling fiber feature table…"` | *(automatic, no widget)* |
| 5. FiberObject node creation | — | `[ Commit to Hierarchy ]` button in CurveAlign TACS section |
| 5. TMEHierarchy commit | — | `[ Commit to Hierarchy ]` button |
| 5. Orientation heatmap layer | — | `VisualizationController._create_heatmap_layer()` → `[id] :: Heatmap :: orientation` |
| 6. TACS visualisation | — | Results → Visualization → CurveAlign TACS View (auto-shown after commit) |
| 7. Save results | — | CurveAlign TACS View → Save results group (CSV/XLSX/PNG) |
| 8. Save project | — | Project → `[ Save Project... ]` |

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

---

## Troubleshooting — Known Bugs and Hard-Won Fixes

This section documents runtime bugs encountered during development. Read before
implementing new features in the same areas to avoid reintroducing known problems.

### Qt / napari threading

**RecursionError: itemSelectionChanged → selectRow infinite loop**
Calling `self._table.selectRow(row)` from inside an `itemSelectionChanged` slot fires
the signal again, causing infinite recursion. Fix: always wrap programmatic row
selection with `blockSignals(True/False)`:
```python
self._table.blockSignals(True)
self._table.selectRow(row)
self._table.blockSignals(False)
```

**vispy RecursionError when setting `layer.visible`**
In napari 0.7.0 + vispy 0.16.x, setting `layer.visible` synchronously inside napari's
event cycle triggers `_reorder_layers_in_the_same_view` → vispy recursion. Always
defer with:
```python
from qtpy.QtCore import QTimer
QTimer.singleShot(0, lambda: setattr(layer, "visible", value))
```

**OpenGL "Cannot make QOpenGLContext current in a different thread"**
Any Qt widget update (progress bar, status label, layer creation) called from inside a
`@thread_worker` worker body crashes napari on MSYS2. Fix: defer ALL GUI updates to
the main thread via `QTimer.singleShot(0, ...)` in the progress callback:
```python
def _cb(step, total, msg):
    print(f"[{step}/{total}] {msg}", flush=True)  # terminal: safe from any thread
    from qtpy.QtCore import QTimer
    QTimer.singleShot(0, lambda s=step, t=total, m=msg:
                      self._emit_batch_progress(s, t, m))
```

**LogController.__init__ wrong argument order**
`LogController(state, log_widget)` — state is the first arg, widget second. Passing
`LogController(log_widget)` silently puts the widget in the `state` slot and crashes
later with a missing-arg error. The None-guards in `info/warn/error` prevent crashes if
`_widget` is None.

### Signal wiring (_main_widget.py)

**viz_widget never gets active image id**
`_on_image_selected` must call `viz_widget.set_active_image(image_id)` explicitly.
This is easy to forget because the viz widget is wired separately from the other widgets.
Without it `_active_image_id` stays `None` and all view buttons silently return early.

**`if result` is False for CurveAlignPipelineResult**
Dataclass instances with DataFrame fields can evaluate to `False` in boolean context
(pandas ambiguous-truth-value). Always use `if result is not None` when guarding
callbacks that receive a result object.

**Commit to Hierarchy fails when image selected via dropdown (not project table)**
`_commit_curvealign` uses `self._active_image_id` set by the project selection signal.
If the user picks an image from the fiber dropdown without clicking the project table
row, `_active_image_id` stays `None`. Always fall back:
```python
image_id = self._active_image_id or self._fiber_selector.currentData()
```

### Coordinate conventions

**`boundary_point_row`/`boundary_point_col` are MATLAB-swapped**
In `fiber_features_df`, the columns from `extract_tif_boundary` follow MATLAB (x,y)
convention: `boundary_point_row` stores the **column** (x) value and
`boundary_point_col` stores the **row** (y) value. When building napari shapes
(row, col order), always swap:
```python
# WRONG:
[[center_row, center_col], [row["boundary_point_row"], row["boundary_point_col"]]]
# CORRECT (napari row,col):
[[center_row, center_col], [row["boundary_point_col"], row["boundary_point_row"]]]
```

**`fiber_df_to_napari_shapes` had swapped dy/dx**
The original implementation used `dy = cos(angle), dx = sin(angle)` — wrong.
`draw_curvs` in `draw_utils.py` uses col-direction = cos(angle), row-direction = sin(angle).
Correct formula:
```python
dy = half_len * np.sin(angle_rad)   # row
dx = half_len * np.cos(angle_rad)   # col
```

### Visualization / draw_utils

**`generate_fiber_overlay` crashes when `coordinates=None`**
When called without pre-computed ROI boundary coords, `coordinates=None` causes
`for roi_coords in coordinates.values()` to crash. Guard:
```python
if coordinates is not None:
    for roi_coords in coordinates.values():
        ...
```
Also: `CurveAlignPipelineResult` now stores `roi_coordinates` so callers can pass it
instead of `None`.

**Fiber overlay always uses absolute angle; heatmap uses relative when boundary available**
- Overlay orientation LINES → always `fiber_structure["angle"]` (absolute, 0–180°)
- Orientation heatmap COLOR → `nearest_angles` (relative to boundary tangent, 0–90°)
  when boundary was run; `fiber_structure["angle"]` (absolute) when no boundary.

**`generate_fiber_heatmap` crashes when `tif_boundary=0`**
With no boundary mask, `in_curvs_flag` and `angles` (nearest_angles) are both `None`.
`fiber_structure[None]` and `angles[None]` raise TypeError. Guard:
```python
if in_curvs_flag is not None:
    map_fibers = fiber_structure[in_curvs_flag]
    map_angles = angles[in_curvs_flag]
else:
    map_fibers = fiber_structure
    map_angles = fiber_structure["angle"].values  # absolute angles
```

### MSYS2 / Windows environment

**vispy `freetype.dll` not found in MSYS2**
`freetype-py` (used by vispy for text rendering) looks for `freetype.dll` but MSYS2
names the library `libfreetype-6.dll`. Fix:
```bash
cp /c/msys64/ucrt64/bin/libfreetype-6.dll /c/msys64/ucrt64/bin/freetype.dll
```

**napari deps build from source under MSYS2 GCC 15**
GCC 15 has a `mkdtemp` overload ambiguity that breaks the `ninja` build tool, which
blocks numpy/vispy source builds. Pre-install C-extension deps via pacman; for vispy
specifically install from git tag with `--no-build-isolation` and patch the
`vispy-0.0.0.dist-info` METADATA version to `0.16.2` (hatch-vcs reports 0.0.0 without
git tag context). See CLAUDE.md "Installing / locating curvelops" for the full sequence.

**Qt SVG DLL missing in MSYS2**
napari requires `QtSvg` for layer icons. Install:
```bash
pacman -S --needed mingw-w64-ucrt-x86_64-qt6-svg
```

### QTableWidget performance

**Slow ROI/TACS filter in TACS View table**
`_populate_tacs_table` with `iterrows()` rebuilds all QTableWidgetItem objects on every
filter change — O(n) Qt creation, visibly slow for 1000+ rows. Use `setRowHidden`:
```python
# Build once; filter by hiding/showing rows
for row_idx in range(self._tacs_table.rowCount()):
    hide = want_tacs and row_data["tacs_class"] != want_tacs
    self._tacs_table.setRowHidden(row_idx, hide)
```