CLAUDE_NAPARI.md
# CLAUDE_NAPARI.md — napari-TMEQuant Plugin Guide

This file provides Claude Code with context and rules for the **napari-TMEQuant**
plugin subproject. Read `CLAUDE.md` first for the core `tme_quant` library —
this file covers only plugin-specific concerns.

---

## What This Plugin Is

**napari-TMEQuant** is a napari plugin that exposes the `tme_quant` library
through an interactive GUI. It is the primary interface for researchers who
want to run TME analysis without writing Python code.

It is a **separate package** (`napari-tme-quant`) that depends on `tme-quant`
as a library. It lives alongside the core library:

```
tme-quant/              ← core library
napari-tme-quant/       ← this plugin
    pyproject.toml
    src/
        napari_tme_quant/
            __init__.py
            _widget.py         ← napari widget entry points
            widgets/           ← one file per dock panel
            controllers/       ← business logic bridging widgets ↔ tme_quant
            utils/             ← plugin-specific helpers only
        napari.yaml            ← plugin manifest (napari ≥ 0.4.17)
    tests/
```

---

## Capabilities

### 1. Image and ROI management
- Load single images or batch-load a project folder (TIFF, OME-TIFF, PNG, JPEG)
- View 2D images, Z-stacks, multichannel, and time series in the napari canvas
- Draw, edit, name, and delete ROI annotations via napari Shapes layers
- Assign ROI types: tumor boundary, stroma, custom region
- Import/export ROI annotations (JSON, CSV, QuPath-compatible GeoJSON)
- Sync ROIs automatically to `tme_quant`'s `ROIManager` / `TMEHierarchy`

### 2. Analysis pipeline control

**Full pipelines (single image or batch):**
- `StandardTMEPipeline` — full TME quantification
- `InteractionAnalysisPipeline` — fiber–cell interaction analysis
- `analyze_tacs_zone()` — TACS zone analysis (pixel-map + per-fiber paths)

**Individual modules:**
- Fiber extraction — method selector (CT-FIRE, skeleton, ridge detection);
  form controls for `ExtractionParams`; shows C++ backend status
- Fiber orientation — method selector (CurveAlign, OrientationJ, gradient,
  structure tensor); controls for `OrientationParams`
- TACS classification — runs after orientation; results overlaid on image
- Cell segmentation — method selector (StarDist, Cellpose, thresholding,
  watershed); controls for `SegmentationParams`; model download via `ModelLoader`
- Cell classification — morphology or marker-based; `ClassificationParams`
- Interaction analysis — fiber–cell and fiber–boundary; distance threshold controls

**3D/volumetric:** All controls adapt to Z-stack inputs. CT-FIRE 3D is shown
as unavailable until the C++ FIRE extension is compiled (see `CLAUDE.md`).

**Parameter presets:** Save, load, and share named parameter sets as JSON.

### 3. Visualization
- Fiber overlay: color-coded by TACS type, orientation angle, or straightness
- Cell overlay: color-coded by cell type or classification score
- Tumor boundary overlay with configurable buffer zones
- Orientation heatmap layer (mean orientation per tile)
- TACS zone heatmap from `plot_tacs_heatmap()`
- Interaction network visualization (requires `[network]` extras)
- All overlay layers togglable independently of analysis state

### 4. Results and export
- Summary table (per-image, per-region): fiber count, mean orientation,
  alignment score, TACS distribution, cell density
- Per-fiber and per-cell data tables (scrollable, sortable)
- Export: CSV, Excel (`.xlsx`), JSON
- Export figures (PNG, SVG) at publication resolution

---

## Architecture

### Three-layer separation

```
Widget layer       napari_tme_quant/widgets/
                   Pure UI: Qt widgets, napari layers, user input.
                   No direct calls to tme_quant analysis functions.
                       ↕
Controller layer   napari_tme_quant/controllers/
                   Translates widget state → tme_quant API calls.
                   Manages threading. Holds TMEProject / TMEHierarchy state.
                       ↕
Library layer      tme_quant/
                   All analysis logic. Never reimplemented in the plugin.
```

**Never put analysis logic in widgets or controllers.** If a computation
belongs in the library, add it there and call it from the controller.

### Threading rule

All `tme_quant` calls must run in a worker thread, never on the Qt main thread:

```python
from napari.qt.threading import thread_worker

@thread_worker(connect={"returned": self._on_result, "errored": self._on_error})
def _run_extraction(self, image, params):
    return self.analyzer.extract_2d(image, params)
```

Never block on I/O or run heavy computation in a widget method or Qt slot.

### State management

A single `PluginState` object (`controllers/state.py`) owns:
- The current `TMEProject` instance
- The mapping from napari layer name → `TMEObject.object_id`
- The active parameter presets

Widgets read and write state only through controllers, never directly.

### napari layer ↔ TMEHierarchy sync

When a user edits a Shapes layer the controller must update the corresponding
`ROIObject`. When analysis produces new objects (fibers, cells) the controller
must create corresponding napari layers. Keep the two representations in sync.

---

## File Naming Conventions

| Path | Contents |
|------|----------|
| `widgets/image_loader_widget.py` | Image / project loading panel |
| `widgets/roi_widget.py` | ROI annotation and management panel |
| `widgets/fiber_widget.py` | Fiber analysis parameter controls |
| `widgets/cell_widget.py` | Cell analysis parameter controls |
| `widgets/tme_widget.py` | TME / interaction analysis controls |
| `widgets/results_widget.py` | Results table and export panel |
| `widgets/visualization_widget.py` | Overlay and colormap controls |
| `controllers/analysis_controller.py` | Orchestrates all analysis calls |
| `controllers/roi_controller.py` | ROI ↔ TMEHierarchy sync |
| `controllers/visualization_controller.py` | Layer creation and updates |
| `controllers/state.py` | `PluginState` singleton |
| `utils/layer_utils.py` | napari layer helpers |
| `utils/export_utils.py` | Figure and table export helpers |

---

## napari Plugin Registration

Register widgets in `napari.yaml` (napari ≥ 0.4.17 manifest format):

```yaml
name: napari-tme-quant
display_name: TMEQuant
contributions:
  commands:
    - id: napari-tme-quant.fiber_widget
      title: Fiber Analysis
      python_name: napari_tme_quant._widget:FiberWidget
  widgets:
    - command: napari-tme-quant.fiber_widget
      display_name: Fiber Analysis
      autogenerate: false
```

Register each dock widget separately so users can open only the panels they need.

---

## Dependencies

`napari-tme-quant/pyproject.toml` should declare only GUI-specific deps:

```toml
dependencies = [
    "tme-quant>=0.2.0",     # all analysis deps are inherited transitively
    "napari>=0.4.17",
    "magicgui>=0.7.0",
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

- Analysis logic belongs in `tme_quant`, not in the plugin.
- All analysis runs in a worker thread via `@thread_worker`.
- Widgets communicate with `tme_quant` only through controllers.
- Keep napari layers and `TMEHierarchy` in sync at all times.
- One widget file per functional panel; one controller per concern.
- Register all widgets in `napari.yaml`.
- Reflect CT-FIRE 3D availability from `ctfire_backend_status()` — disable
  the 3D CT-FIRE option in the UI until the C++ extension is available.