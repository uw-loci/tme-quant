# tme_quant — Package Architecture

```
examples/                              ← runnable demo scripts
├── example_3d_volumetric_workflow.py  ← 3-D z-stack volumetric analysis
├── example_analyze_tacs_zone.py       ← standalone TACS zone analysis demo
├── example_ctfire_workflow.py         ← CT-FIRE individual fiber + TACS workflow
├── example_ctfire_workflow_hierarchy.py ← CT-FIRE workflow with TMEHierarchy
├── example_curvealign_workflow.py     ← CurveAlign segment orientation + TACS workflow
├── example_curvealign_curvelets_mode_pipeline.py ← curvealign curvelets-mode + TACS + hierarchy
├── example_curvealign_ctfire_pipeline.py         ← CT-FIRE pipeline: CT-FIRE mode (curvelets→FIRE) and FIRE-only mode (no curvelops needed), TACS viewer, saves overlay/heatmap/xlsx
├── example_hierarchy_object_analysis.py
└── roi_curvealign_orientation_example.py

src/tme_quant/
├── __init__.py                    ← flat public API
│
├── core/
│   ├── __init__.py
│   ├── base_models.py             ← TMEObject, ObjectType, TMEType, Geometry, …
│   ├── geometry.py                ← BoundingBox, ROI + Shapely utilities
│   ├── hierarchy.py               ← TMEHierarchy
│   ├── image_entry.py             ← ImageEntry
│   ├── project.py                 ← TMEProject
│   ├── roi_manager.py             ← ROIManager
│   └── tme_objects/               ← canonical domain object definitions
│       ├── __init__.py            ← re-exports CellObject, FiberObject, Tumor, …
│       ├── base_objects.py        ← re-export shim → core/base_models.py
│       ├── cell_objects.py        ← CellObject, CellType, SegmentationParams, …
│       ├── fiber_objects.py       ← FiberObject, FiberPopulation, OrientationResult, …
│       ├── interaction_objects.py ← Interaction, InteractionNetwork
│       ├── stroma_objects.py      ← StromaRegion, Stroma, ECMComponent
│       ├── tissue_objects.py      ← TissueRegion, TissueSample, TissueZone
│       └── tumor_objects.py       ← TumorRegion, Tumor, TumorGrade
│
├── fiber_analysis/
│   ├── _cpp/                      ← C++ source for pybind11 extension modules
│   │   └── ctfire/
│   │       ├── CMakeLists.txt     ← build config (pybind11 + cmake)
│   │       ├── fire.h             ← FIRE algorithm declarations
│   │       ├── fire.cpp           ← FIRE algorithm implementation (TODO)
│   │       └── fire_bindings.cpp  ← pybind11 bindings → _ctfire_cpp.pyd/.so
│   ├── __init__.py                ← public API; re-exports all key symbols
│   ├── config.py                  ← ExtractionParams, OrientationParams, CurveAlignAnalysisMode (CURVELETS/WINDOWED/FULL), CurveAlignParams + all sub-params
│   ├── extraction.py              ← BaseExtractionMethod + FiberExtractionAnalyzer
│   ├── orientation.py             ← BaseOrientationMethod + FiberOrientationAnalyzer
│   ├── io.py                      ← FiberAnalysisExporter (+ FijiBridge re-export)
│   ├── results.py                 ← FiberAnalysisResult
│   ├── tacs.py                    ← classify_fiber_tacs(), get_tacs_color()
│   ├── methods/                   ← concrete method implementations
│   │   ├── __init__.py            ← re-exports all method classes
│   │   ├── ctfire.py              ← CT-FIRE curvelet fiber extraction
│   │   ├── curvealign.py          ← CurveAlign orientation analysis
│   │   ├── fiji_bridge.py         ← unified Fiji/ImageJ subprocess bridge
│   │   ├── gradient.py            ← Sobel/Scharr gradient orientation
│   │   ├── orientationj.py        ← OrientationJ Fiji plugin wrapper
│   │   ├── ridge_detection.py     ← Ridge Detection Fiji plugin wrapper
│   │   ├── skeleton.py            ← skeletonization-based extraction
│   │   └── structure_tensor.py    ← windowed structure tensor orientation
│   └── utils/
│       ├── __init__.py
│       ├── ctfire_utils.py
│       ├── curvelet_utils.py
│       ├── fiber_dataframe_utils.py ← compute_fiber_density_and_alignment, round_mlab
│       ├── geometry_utils.py
│       └── _ctfire_cpp.pyd/.so    ← compiled output (not in VCS; built from _cpp/ctfire/)
│
├── cell_analysis/
│   ├── __init__.py
│   ├── cell_analyzer.py           ← CellAnalyzer (orchestrator)
│   ├── classification.py          ← CellClassificationAnalyzer
│   ├── config.py                  ← re-export shim → tme_objects/cell_objects
│   ├── io.py                      ← CellAnalysisExporter
│   ├── model_loader.py            ← ModelLoader (StarDist / Cellpose)
│   ├── quantification.py          ← CellQuantificationAnalyzer
│   ├── results.py                 ← re-export shim → tme_objects/cell_objects
│   ├── segmentation.py            ← CellSegmentationAnalyzer
│   ├── methods/                   ← flat; all method classes at this level
│   │   ├── __init__.py            ← MethodRegistry
│   │   ├── base_classification.py
│   │   ├── base_segmentation.py
│   │   ├── cellpose_segmentation.py
│   │   ├── marker_classification.py
│   │   ├── morphology_classification.py
│   │   ├── stardist_segmentation.py
│   │   ├── threshold_segmentation.py
│   │   └── watershed_segmentation.py
│   └── utils/
│       ├── __init__.py
│       ├── cell_utils.py
│       ├── postprocessing.py
│       ├── preprocessing.py
│       └── validation.py
│
├── tme_analysis/
│   ├── __init__.py
│   ├── config.py
│   ├── interaction_detector.py    ← CellFiberInteractionDetector
│   ├── interaction_features.py    ← mechanical / migration / invasive scores
│   ├── interaction_network.py     ← InteractionNetworkAnalyzer
│   ├── io.py
│   ├── measurement_engine.py      ← MeasurementEngine
│   ├── region_manager.py          ← RegionManager
│   ├── spatial_features.py        ← SpatialRelationshipExtractor
│   ├── tacs_features.py           ← TACSFeatureExtractor
│   ├── tme_analyzer.py            ← TMEAnalyzer (orchestrator)
│   ├── pipelines/
│   │   ├── __init__.py                    ← re-exports analyze_tacs_zone
│   │   ├── interaction_analysis_pipeline.py
│   │   ├── standard_tme_pipeline.py
│   │   ├── tacs_pipeline.py               ← analyze_tacs_zone(), plot_tacs_heatmap()
│   │   ├── curvealign_curveletsMode_pipeline.py  ← curvealign_curvelets_mode_pipeline(): in-memory CurveAlign curvelets-mode pipeline (curvealign_pipeline is a deprecated alias)
│   │   └── curvealign_ctfireMode_pipeline.py     ← curvealign_ctfire_mode_pipeline(): in-memory CT-FIRE individual-fiber pipeline; returns CTFirePipelineResult with full morphology; coordinates-only boundary mode (no boundary_img) populates roi_measurements_df but skips nearest_angles/in_curvs_flag
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── alignment_utils.py     ← compute_fiber_alignment_to_roi
│   │   ├── distance_utils.py
│   │   ├── orientation_utils.py
│   │   ├── statistical_utils.py
│   │   └── validation.py
│   └── visualization/
│       ├── __init__.py
│       └── interaction_visualization.py
│
├── image_registration/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── preprocessing.py           ← 16 functions (merged from 3 files)
│   ├── registration_manager.py
│   ├── transform_handler.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── base_registration.py
│   │   ├── deep_learning/
│   │   │   ├── __init__.py
│   │   │   ├── comir_registration.py
│   │   │   └── voxelmorph_registration.py
│   │   ├── feature_based/
│   │   │   ├── __init__.py
│   │   │   ├── orb_registration.py
│   │   │   └── sift_registration.py
│   │   ├── intensity_based/
│   │   │   ├── __init__.py
│   │   │   ├── cross_correlation.py
│   │   │   ├── he_shg_registration_python.py
│   │   │   └── mutual_information.py
│   │   └── landmark_based/
│   │       ├── __init__.py
│   │       ├── manual_landmarks.py
│   │       └── thin_plate_spline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── transform_utils.py
│   └── visualization/
│       ├── checkerboard.py
│       ├── difference_map.py
│       ├── interactive_viewer.py
│       └── overlay.py
│
└── napari-tme-quant/              ← napari plugin (separate installable package; co-located for prototyping)

docs/
├── architecture.md                    ← canonical file tree (this file)
├── getting_started.md                 ← install and first-run guide
└── fire_only_windows_setup.md         ← FIRE-only pipeline setup on Windows MSYS2 UCRT64; explains two-level src structure, .pth file approach, and flat-layout alternative

src/ctfire_py/                         ← CT-FIRE Python module; tracked here, changes made only in 32-convert-ctfire branch
├── __init__.py
├── ct_fire.py                         ← main entry point: ct_fire()
├── ct_reconstruction.py               ← curvelet image preprocessing (requires curvelops)
├── fire_2d_angle.py                   ← FIRE 2D angle computation (no curvelops needed)
├── parameter_mapping.py               ← MATLAB→Python parameter mapping
├── fiber_backend.*.pyd/.so            ← pre-built C++ extension (platform-specific)
├── fiber_analysis/                    ← fiber geometry and statistics
│   ├── fiber_angles.py
│   ├── fiber_stats.py
│   └── network_stats.py
├── fiber_processing/                  ← FIRE graph-tracing pipeline stages
│   ├── beamproc.py, check_danglers.py, curvealign_filter.py
│   ├── fiber2beam.py, fiberbreak.py, fiberlink_py.py
│   ├── fiberlinkgap_py.py, fiberproc.py, fiberremove.py
│   └── remove_repeat.py
├── utils/                             ← shared utilities
│   ├── fiber_helpers.py, remove_repeat.py, trimxfv.py
└── CPP/                               ← C++ FIRE backend source
    ├── Makefile                       ← macOS (Apple Silicon / M-chips)
    ├── Makefile.linux                 ← WSL / Linux
    ├── Makefile.ucrt64                ← Windows MSYS2 UCRT64
    └── extend_xlink_native.cpp, fiberproc_native.cpp, findlocmax_native.cpp, …
Inter-module note: ctfire_py imports from pycurvelets (round_mlab); pycurvelets must
be installed separately (it lives in the 32-convert-ctfire repo, not in tme-quant).
    ├── pyproject.toml             ← declares napari-tme-quant distribution
    ├── src/
    │   └── napari_tme_quant/      ← plugin Python package (NOT part of tme_quant namespace)
    │       ├── __init__.py
    │       ├── _main_widget.py    ← TMEQuantDockWidget (QTabWidget container)
    │       ├── napari.yaml        ← plugin manifest (napari ≥ 0.4.17)
    │       ├── controllers/       ← PluginState + per-tab controllers
    │       ├── utils/             ← layer helpers, export utils, coord utils
    │       └── widgets/           ← one file per dock panel
    └── tests/

tests/
└── test_geometry_utils.py             ← 55 unit tests for fiber_analysis/utils/geometry_utils.py
                                         covers: _angle_between_orientations,
                                         compute_angle_to_boundary_normal{,_simplified},
                                         _circ_r, find_nearest_boundary_index,
                                         _get_first_neighbor, _find_connected_pts,
                                         compute_boundary_tangent_angle,
                                         compute_fiber_properties,
                                         compute_relative_fiber_angles
```

---

## Verification Status (2026-04-19)

- [PASS] All .py files parse without syntax errors
- [PASS] All single-dot relative imports resolve to existing files
- [PASS] `pytest tests/test_geometry_utils.py` — 55 passed
- [PASS] All two-dot relative imports resolve to existing files
- [PASS] All three-dot relative imports resolve to existing files
- [PASS] All absolute `tme_quant.X` imports in example files resolve
- [PASS] No stale module path references in live import statements
- [PASS] pyproject.toml package discovery unchanged (src/ layout preserved)

## Statistics

| | Count |
|---|---|
| .py files | 100 |
| Max folder depth | 8 |
