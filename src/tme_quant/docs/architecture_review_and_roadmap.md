# TMEQuant — Architecture Review, Code-Quality Workflow & ECM-Interaction Roadmap

This document reviews the TMEQuant object architecture, explains how to combine
**manual tests** and **automated agent workflows** for code-quality control, and
lays out a roadmap for extending the library toward **state-of-the-art
ECM/collagen ↔ fiber ↔ cell ↔ tumor interaction analysis**, dynamic imaging, and
3-D.

> Path convention (same as `pycurvelets_integration_status.md`): tme_quant paths
> are relative to the Python package root, e.g. `fiber_analysis/tacs.py` resolves
> to `src/tme_quant/src/tme_quant/fiber_analysis/tacs.py`.

> Status: **advisory** — review + roadmap. Nothing here is implemented yet. Any
> implementation must follow `REFACTORING_GUIDE.md` (≤5 files/batch, pytest green
> after each batch, overlap check, two-layer porting).

---

## 1. Architecture review

### 1.1 The hierarchy models interactions in two complementary layers

This is the design's core strength: **spatial nesting** and **interaction edges**
are kept in separate layers.

**(a) Containment tree.** Every domain object subclasses one base class,
`TMEObject` (`core/base_models.py`) — a Composite node with `parent`/`children`
and `add_child()` / `detach()`. `TMEHierarchy` (`core/hierarchy.py`) wraps the tree
with an O(1) `_HierarchyIndex` (`by_id`, `by_tme_type`, `by_object_type`).
`spatial_assign()` nests objects under regions by point-in-polygon containment.
This layer answers *"is located in / is part of"*: fiber ∈ tumor region ∈ image ∈
project.

**(b) Interaction graph (cross-cutting edges).** Interactions are deliberately
**not** tree nodes; they are edges overlaid on the tree:

- `InteractionDetector` (`tme_analysis/interaction_detector.py`) builds
  `InteractionPair` records via KDTree queries (cell–fiber, fiber–fiber,
  fiber–tumor, cell–tumor).
- `interaction_features.py` annotates each pair with mechanical / migration /
  invasive scores.
- `MeasurementEngine` aggregates pairs into TACS / spatial / prognostic features.
- `InteractionNetworkAnalyzer` (`tme_analysis/interaction_network.py`) lifts pairs
  into a `networkx` graph for centrality / community / critical-edge analysis.
- Objects also carry denormalized adjacency (`CellObject.interacting_fiber_ids`,
  `neighbor_cell_ids`).

Using the tree for "is-part-of" and a separate edge/graph layer for "interacts-with"
is the right abstraction for the TME: one fiber can participate in many interactions
without distorting its place in the spatial hierarchy, and graph-theoretic analysis
falls out naturally.

### 1.2 Extensibility / modularity / maintainability

**Strong:**
- Clean subpackage boundaries with documented import-depth rules.
- Strict one-way dependency (no Qt/napari in the library) → headless/HPC-safe,
  fully testable.
- Genuinely pluggable *analysis methods*: ABC + dict-dispatch for fiber
  extraction/orientation; `MethodRegistry` for cell segmentation/classification.
- Robust optional-deps via lazy `_has_X()` probes + documented fallbacks
  (curvelops → MATLAB → Frangi).
- Schema-flexible nodes: `TMEObject.properties` / `metadata` dicts let new
  measurements attach without class changes.

**Friction points — these are exactly what limit the requested expansions:**
- **TACS classification is hard-coded**, contradicting the otherwise-pluggable
  design. `classify_fiber_tacs()` (`fiber_analysis/tacs.py`) is an `if/elif` with
  baked thresholds (60/30/90°, straightness 0.7); TACS types are bare strings;
  `get_tacs_color()`, the feature counts in `tme_analysis/tacs_features.py`
  (`tacs_types.count('TACS-1')`), GeoJSON colors (`fiber_objects.py`), and IO all
  hard-code the three labels. Adding TACS-4 or an alternative scheme touches
  ~10–15 scattered sites.
- **Dead config:** `TMEAnalysisParams.tacs_angle_threshold_*` exist but are **not
  wired** into the canonical classifier (thresholds live as function defaults).
- **Two parallel interaction models** that can drift: enum-typed `Interaction`
  (`core/tme_objects/interaction_objects.py`) vs. string-typed `InteractionPair`
  (`tme_analysis/config.py`).
- **No temporal dimension:** `Interaction.temporal_duration` exists but is never
  populated; no `timepoint` / `track_id` on objects.
- **3-D is structurally anticipated but computationally 2-D:** `Geometry.is_3d`,
  `Nx3` centerlines, `GeometryType.MESH/CUBOID`, `OrientationResult.dimension="3D"`
  all exist, yet TACS angle/tangent/distance math is 2-D only and
  `CTFireExtraction.supports_3d()` returns `False`.

---

## 2. Code-quality control: manual tests + automated agent workflows

TMEQuant already has a strong QC substrate (pytest, real-data and MATLAB-parity
tests, import health check, black/ruff/mypy, example scripts, a post-commit
architecture-doc hook, and the batch discipline in `REFACTORING_GUIDE.md`). The
goal of this section is to define **how to combine human/manual verification with
automated agent workflows** into a layered, repeatable gate.

### 2.1 The four QC layers (run in order, fail fast)

| Layer | What | Tooling in this repo | When |
|-------|------|----------------------|------|
| L1 — Static | Format, lint, types, import health | `black src/`, `ruff check src/`, `mypy src/tme_quant/`, the Import Health Check snippet in `CLAUDE.md` | Every change, before commit |
| L2 — Automated tests | Unit + real-dataset + parity | `pytest tests/ -v`; real-data `TestXxxRealData` classes; `TMEQ_VALIDATE_MATLAB=1` parity | After every ≤5-file batch (`REFACTORING_GUIDE.md §3–4`) |
| L3 — Manual / visual | Pipelines & visual output a human must eyeball | `examples/*.py` that save overlay / heatmap / `.xlsx` (e.g. `example_curvealign_ctfire_pipeline.py`) | Anything touching visualization, pipelines, or numeric output |
| L4 — Agent review | Bug-hunt, simplification, security | `/code-review` (low→ultra), `/simplify`, `/security-review`; `Explore`/`Plan` subagents | Before merge / PR |

### 2.2 Manual tests — what humans verify that machines should not

Reserve manual effort for what automation cannot cheaply assert:
- **Visual correctness** of fiber overlays, TACS color maps, and angle heatmaps
  produced by the example pipelines. A green pytest does not prove an overlay is
  biologically sensible — open the saved PNG/`.xlsx`.
- **Numeric plausibility on real SHG data**, especially across the dual-venv
  reality: the Frangi fallback (no curvelops) and the curvelops backend can give
  materially different fiber structure. Spot-check both.
- **New TACS / interaction taxonomies**: confirm that category boundaries match the
  intended biology on annotated reference images before trusting the counts.

Manual-test hygiene:
1. Keep example scripts runnable and self-documenting; the three real-image
   examples (`example_ctfire_workflow*.py`, `example_curvealign_workflow.py`) need
   data in `data/` — note that in the PR description.
2. When a manual check finds a defect, **promote it to an automated test** (golden
   reference, see §2.4) so it never regresses silently.

### 2.3 Automated agent workflows — where they add the most value

- **`/code-review` (medium/high)** on every PR diff for correctness bugs + reuse/
  simplification. Use **ultra** for large refactors (e.g. the TACS-registry work in
  §1.2 / the roadmap) where multi-agent depth pays off.
- **`/simplify`** after a feature lands to remove duplication introduced during
  development — relevant given the two parallel interaction models.
- **`/security-review`** when touching IO, subprocess bridges (`integrations/
  fiji_bridge.py`), model downloads (`cell_analysis/model_loader.py`), or any new
  external-tool surface.
- **`Explore` / `Plan` subagents** for scoping a change before writing it — the
  overlap-check protocol in `REFACTORING_GUIDE.md §5` is exactly an `Explore` task.
- **Post-commit hook** already auto-updates `docs/architecture.md` on structural
  changes — always verify it captured intent (per the CLAUDE.md maintenance rule).

### 2.4 Recommended additions (close the current gaps)

1. **Golden-output regression for pipelines.** The example pipelines already emit
   overlay/heatmap/`.xlsx`. Capture small reference outputs (fiber-structure
   DataFrame, TACS counts, summary stats) as committed fixtures and add a pytest
   that re-runs the pipeline on a tiny fixed image and asserts equality within
   tolerance. This converts today's manual eyeballing into a CI gate.
2. **Dual-backend CI matrix.** curvelops lives only in `.venv-curvelops` (MSYS2
   UCRT64). Document/automate two runs: default (Frangi fallback, 206 passed/2
   skipped) and curvelops (run from `.venv-curvelops`, 239 passed/7 skipped). A
   backend-divergence test that asserts both produce *qualitatively* consistent
   fiber counts guards the fallback.
3. **Parity-as-gate for numeric ports.** Keep `TMEQ_VALIDATE_MATLAB=1` runs in the
   PR checklist for any function ported from `pycurvelets`/MATLAB; never weaken an
   assertion to go green (`REFACTORING_GUIDE.md §4`).
4. **Pre-merge agent gate.** Make "`/code-review` clean + pytest green (both
   backends) + import health check" the definition of done for a batch.

### 2.5 One-line "done" definition for a batch

> L1 clean → L2 green (default **and** `.venv-curvelops`) → L3 visual check on any
> touched pipeline/visual output → L4 `/code-review` (and `/security-review` if IO/
> subprocess/model surface changed) clean → `docs/architecture.md` + module map in
> `CLAUDE.md` updated in the same commit.

---

## 3. Expansion roadmap — state-of-the-art ECM/collagen ↔ cell/tumor interaction analysis

The current library models **one** collagen-signature scheme (TACS-1/2/3,
Provenzano 2006 / Conklin 2011) on **2-D static** images. Below is a capability-
oriented roadmap toward the modern ECM/TME analysis state of the art, each item
mapped to a concrete extension point in the existing architecture so it is
actionable. The unifying theme: **turn the hard-coded classification/feature paths
into pluggable registries**, then build the new science on top of those seams.

### 3.1 Foundational refactor — pluggable classification & feature layers *(prerequisite)*

Nearly every item below re-multiplies the hard-coded TACS sites unless this lands
first.

- **`TACSClassifier` protocol + registry**, mirroring `FiberExtractionAnalyzer`
  dict-dispatch / `MethodRegistry`. Replace bare-string types with a registry of
  `TACSCategory` descriptors `(name, color, predicate/angle-range, references)` so
  counting, coloring, IO, and GeoJSON discover types **dynamically**.
  - Touch points to converge: `fiber_analysis/tacs.py`,
    `core/tme_objects/fiber_objects.py::_classify_tacs_from_metrics`,
    `tme_analysis/tacs_features.py`, `tme_analysis/measurement_engine.py`, export/IO.
  - Wire the existing `tacs_angle_threshold_*` config into the classifier (remove
    dead config; thresholds become runtime params).
- **Unify the interaction model** (`Interaction` + `InteractionPair` → one typed
  edge) and make `interaction_type` an open enum/registry like TACS.
- **Registry of interaction-feature scorers** (mechanical / migration / invasive →
  pluggable) so new biomechanical scores plug in like methods.

### 3.2 Extended & learned collagen-signature taxonomies

- **Extended TACS categories.** The registry from §3.1 lets you add the finer-
  grained TACS subtypes proposed in recent multiphoton/SHG-microscopy literature
  (e.g. splitting reorganization/alignment/invasion stages into additional
  categories) as descriptors — no code changes outside the registry.
- **ML/learned TACS.** Register a classifier backed by a model
  (`cell_analysis/model_loader.py` already provides a download/cache pattern):
  a feature-based classifier on fiber morphology + boundary-relative angles, or a
  CNN on SHG patches. The registry contract returns a category label + score, so
  rule-based and learned classifiers are interchangeable.
- **Continuous signatures, not just discrete classes.** Expose an ordinal
  "TACS progression score" and per-fiber soft memberships (the
  `tacs_score`/`_tacs_score()` machinery already computes a 0–1 confidence) for
  downstream gradient/regression analyses.

### 3.3 Spatial-statistics & graph methods for fiber ↔ cell ↔ tumor interaction

Move beyond pairwise nearest-neighbor detection to population-level spatial
statistics — the modern standard in spatial biology:

- **Point-process statistics** in a new `tme_analysis/spatial_statistics.py`:
  Ripley's K / L, pair-correlation function (PCF), cross-type nearest-neighbor,
  and mark-correlation functions for cell–cell, cell–fiber, and fiber–fiber
  organization. Inputs come straight from `CellObject`/`FiberObject` positions.
- **Cellular-neighborhood / niche analysis** layered on the existing `networkx`
  interaction graph (`InteractionNetworkAnalyzer`): cluster nodes by local
  composition to define TME niches at the tumor margin.
- **Distance-to-boundary radial profiling.** Generalize the TACS-zone idea into a
  first-class radial-profile API: bin any fiber/cell feature vs. signed distance to
  the tumor boundary to quantify invasion gradients (`RegionManager.generate_tumor_zones`
  already produces core/margin/stroma rings — profile across them).
- **(Optional) Graph neural networks** on the interaction graph for learned
  interaction phenotypes; gate behind a lazy optional dep like the other DL stacks.

### 3.4 Richer ECM/collagen biophysics

- **Alignment/anisotropy descriptors** beyond per-fiber angle: orientation order
  parameter, coherency, and von-Mises concentration over a region (some pieces
  exist in `interaction_features.compute_alignment_heterogeneity` and
  `RegionOrientationMap` — consolidate into a regional ECM-organization summary).
- **Density / matrix-pattern metrics** (gap/lacunarity, fiber-end density, branch/
  crosslink density) — the TWOMBLI-style ECM descriptors — as new fiber-structure
  features, complementing CT-FIRE morphology.
- **Mechanobiology proxies.** Fiber-tension/stiffness proxies and, where the
  modality supports it, forward/backward SHG ratio (FSHG/BSHG) as a crosslinking/
  maturity indicator. These attach as `Measurement`s on `FiberObject`/region nodes.

### 3.5 Multimodal & multiplexed integration

- Extend `image_registration/` (already SHG↔H&E) to register SHG collagen with
  **multiplexed cell phenotyping** (IF / IMC / CODEX). Phenotyped cells become
  `CellObject`s with `marker_expression`, enabling collagen-signature ↔ immune-niche
  co-analysis at the margin.
- **Interop:** keep the QuPath GeoJSON export path first-class (objects already
  emit `to_geojson_feature`), and consider an `anndata`/`spatialdata` exporter so
  TMEQuant outputs drop into the Python spatial-omics ecosystem (squidpy/Giotto).

### 3.6 Dynamic (live / longitudinal) imaging

- Add `timepoint` / `frame_index` (+ optional `track_id` / `lineage_parent_id`) to
  fiber/cell objects; add a `Track` / `TimeSeries` container (sibling to
  `FiberPopulation` / `CellPopulation`) linking one physical object across frames.
- `ImageEntry` already documents `(T,H,W,C)` arrays — add a time accessor and a
  project-level frame coordinator.
- Populate the existing `temporal_duration` (+ onset/offset) on interactions →
  interaction-dynamics metrics: formation/dissolution rates, contact-guidance
  migration speed along fibers, collagen remodeling over time.
- **Dynamic TACS:** classify TACS *transitions* (e.g. TACS-2 → TACS-3 progression),
  the biologically meaningful invasion-onset signal.

### 3.7 Volumetric (3-D) analysis

- Represent 3-D boundaries as surface meshes (`GeometryType.MESH` exists) and
  compute surface tangent-plane / normal so the angle convention generalizes
  (fiber vector vs. local boundary **tangent plane**). **The 2-D angle-to-tangent
  definition must be explicitly redefined for 3-D** before TACS can be computed
  volumetrically.
- 3-D distance-to-boundary via 3-D KDTree / EDT; generalize
  `compute_relative_fiber_angles` / `compute_boundary_tangent_angle` to accept
  `(z, row, col)`.
- Finish the pending C++ 3-D FIRE extension so `CTFireExtraction.supports_3d()`
  returns `True`.

---

## 4. Prioritized roadmap summary

| Pri | Theme | Outcome | Primary touch points |
|-----|-------|---------|----------------------|
| P1 | Pluggable TACS + interaction registries (§3.1) | New taxonomies & learned classifiers become drop-ins; dead config removed | `fiber_analysis/tacs.py`, `tme_analysis/{tacs_features,measurement_engine,config}.py`, `core/tme_objects/{fiber,interaction}_objects.py` |
| P2 | Spatial-statistics & graph methods (§3.3) | Population-level fiber↔cell↔tumor organization, invasion gradients, niches | new `tme_analysis/spatial_statistics.py`, `interaction_network.py` |
| P3 | Extended/learned signatures + ECM biophysics (§3.2, §3.4) | TACS-4+, ML-TACS, TWOMBLI-style ECM & mechano proxies | TACS registry, `fiber_analysis/utils/fiber_dataframe_utils.py`, `cell_analysis/model_loader.py` |
| P4 | Multimodal/multiplex integration (§3.5) | SHG↔multiplex cell phenotyping; spatial-omics export | `image_registration/`, new exporters |
| P5 | Dynamic imaging (§3.6) | Tracks, interaction lifetimes, dynamic TACS transitions | `core` object fields, new `Track` container, `image_entry.py` |
| P6 | Volumetric 3-D (§3.7) | 3-D TACS, surface-relative angles, 3-D FIRE | geometry/angle utils, C++ FIRE extension |

Sequencing note: **P1 is a prerequisite for P3/P5/P6** — without the pluggable
classification/interaction seams, every new taxonomy and the dynamic/3-D work each
re-multiply the hard-coded sites identified in §1.2.

---

## 5. Verification (when any roadmap item is implemented)

- Per ≤5-file batch: `pytest tests/ -v` from `src/tme_quant/` (default **and**
  `.venv-curvelops`), plus the Import Health Check (`CLAUDE.md`).
- Add real-dataset `TestXxxRealData` classes where reference data exists
  (`REFACTORING_GUIDE.md §3.1`); resolve shared fixtures from
  `tests/test_results/`.
- **TACS-registry parity test:** assert the default registry reproduces today's
  `classify_fiber_tacs()` labels on existing fixtures *before* adding new types.
- **Golden-output regression** (§2.4) for any touched pipeline.
- `/code-review` (ultra for the P1 refactor) + `/security-review` for new IO/model
  surfaces.
- Update `docs/architecture.md` and the `CLAUDE.md` module map in the same commit
  as any structural change.
