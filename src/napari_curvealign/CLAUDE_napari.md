# CLAUDE_napari.md — napari-curvealign Plugin Guide

This file provides Claude Code with the context, conventions, and rules for
working on the **napari-curvealign** plugin (`src/napari_curvealign/`).

---

## Architecture Overview

### Plugin entry point

```
napari.yaml
  → __init__.py : napari_experimental_provide_dock_widget()
    → widgets/curve_align_widget.py : CurveAlignWidget
```

`napari.yaml` registers a single npe2 command (`tme-quant.make_widget`) that
calls `napari_experimental_provide_dock_widget`, which lazily imports and
returns a `CurveAlignWidget` instance.

### Widget structure

`CurveAlignWidget` is a `QTabWidget` with five tabs:

| Tab | Purpose |
|-----|---------|
| **Main** | Image loading, curvelet analysis, results display |
| **Preprocessing** | Thresholding, filtering, Bio-Formats I/O |
| **Segmentation** | Cell/region segmentation (threshold, Cellpose, StarDist) |
| **ROI Manager** | Draw, import/export, and manage ROI annotations |
| **Post-Processing** | Post-analysis metrics and visualization |

### Module map

```
src/napari_curvealign/
├── __init__.py                  ← plugin entry point (npe2 hook)
├── napari.yaml                  ← plugin manifest
├── curvelet_analysis_run.py     ← curvelet/FDCT analysis pipeline
├── preprocessing.py             ← image preprocessing + Bio-Formats
├── segmentation.py              ← segmentation method dispatch
├── roi_manager.py               ← ROI creation, editing, persistence
├── tacs.py                      ← TACS classification + boundary metrics
├── widgets/
│   ├── __init__.py              ← exports CurveAlignWidget
│   ├── curve_align_widget.py    ← main dock widget (5-tab UI)
│   ├── dialogs.py               ← modal dialogs (advanced params, results, metrics)
│   └── widget_utils.py          ← shared helpers (grayscale conversion, save filters)
└── imagej/
    ├── __init__.py              ← exports FijiBridge, get_fiji_bridge
    └── client.py                ← Fiji/ImageJ bridge via napari-imagej
```

### Separation of concerns

Analysis logic lives in `pycurvelets` (the core library). This plugin is
strictly UI — it binds widgets to library calls and displays results.
Do not reimplement analysis here.

---

## Coding Rules

### Qt imports — use `qtpy` only

All Qt imports must go through `qtpy`. Never import from `PyQt6`, `PyQt5`,
or `PySide2` directly:

```python
# correct
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from qtpy.QtCore import Qt
from qtpy import QtCore

# wrong — do not use
from PyQt6.QtWidgets import QWidget
```

The codebase already follows this consistently.

### Type hints — `__future__` annotations and `TYPE_CHECKING`

Use `from __future__ import annotations` at the top of new files.
Guard heavy imports (like `napari.viewer`) behind `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari.viewer
```

### Optional dependency imports — `HAS_*` flags

Optional backends use a try/except pattern with a `HAS_*` boolean flag:

```python
try:
    import cellpose
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False
```

Current flags across the codebase:

| Flag | Module | Optional dependency |
|------|--------|---------------------|
| `HAS_AICSIO` | preprocessing.py | aicsimageio (Bio-Formats) |
| `HAS_IMAGEJ` | preprocessing.py, imagej/client.py | napari-imagej |
| `HAS_SKIMAGE` | segmentation.py, roi_manager.py | scikit-image |
| `HAS_CELLPOSE` | segmentation.py | cellpose |
| `HAS_CELLCAST` | segmentation.py | cellcast/StarDist |
| `HAS_NAPARI` | roi_manager.py | napari.layers |
| `HAS_PYCURVELETS` | roi_manager.py, curvelet_analysis_run.py | pycurvelets |
| `HAS_ROIFILE` | roi_manager.py | roifile |

When adding a new optional feature, follow this same pattern.

### Enum classes for option sets

Use `Enum` subclasses for fixed option sets, not bare strings:

| Enum | Module | Values |
|------|--------|--------|
| `ThresholdMethod` | preprocessing.py | OTSU, TRIANGLE, ISODATA, MEAN, MINIMUM, LI, YEN, MANUAL |
| `SegmentationMethod` | segmentation.py | THRESHOLD, CELLPOSE_CYTO, CELLPOSE_NUCLEI, STARDIST, CUSTOM_MASK |
| `BoundaryType` | widgets/dialogs.py | NO_BOUNDARY, TIFF_BOUNDARY |
| `ROIShape` | roi_manager.py | RECTANGLE, FREEHAND, ELLIPSE, POLYGON |
| `ROIAnalysisMethod` | roi_manager.py | CURVELETS, CTFIRE, POST_ANALYSIS |

### Docstrings and tests

- Use NumPy docstring format.
- Tests use `tempfile.TemporaryDirectory()` — never write to the repo root.
- Headless testing: set `QT_QPA_PLATFORM=offscreen` for any test that
  creates Qt widgets.

### Linting

Run `uv run ruff check .` before committing. The Makefile target `make check`
does this.

---

## Claude Code Hooks (Suggested Configuration)

These hooks can be added to `.claude/settings.json` to automate lint checks
and testing reminders during development sessions.

### PreToolUse — auto-lint on file changes

Run `ruff check` automatically when editing Python files:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hook": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q '\\.py'; then uv run ruff check --fix $(echo \"$CLAUDE_TOOL_INPUT\" | grep -o '\"file_path\":\"[^\"]*\"' | cut -d'\"' -f4) 2>/dev/null; fi"
      }
    ]
  }
}
```

### PostToolUse — remind about headless testing

After test runs, remind about the offscreen platform if GUI tests fail:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hook": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'pytest'; then echo 'Reminder: GUI tests need QT_QPA_PLATFORM=offscreen'; fi"
      }
    ]
  }
}
```

These are suggestions — adopt or modify as needed.

---

## Guidelines for Claude

- **Keep the tab structure intact.** The 5-tab layout (Main, Preprocessing,
  Segmentation, ROI Manager, Post-Processing) is the user-facing organization.
  Do not merge, reorder, or remove tabs without explicit direction.

- **Always use `qtpy`.** Never add direct `PyQt6` or `PySide2` imports.

- **Follow the `HAS_*` flag pattern** when adding optional feature support.
  Check the flag before calling the optional dependency.

- **Keep analysis logic in `pycurvelets`**, UI binding in `napari_curvealign`.
  If you need new analysis functionality, it belongs in the library, not here.

- **Platform-specific issues** — refer to
  `doc/MACOS_INTEL_TROUBLESHOOTING.md` for macOS Intel torch/Qt issues.

- **Run tests** with `make test` or `uv run pytest -v`. For GUI widget tests,
  ensure `QT_QPA_PLATFORM=offscreen` is set.

- **Run linting** with `make check` or `uv run ruff check .`.
