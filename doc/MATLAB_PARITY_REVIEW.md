# CT-FIRE MATLAB parity review

Author: Andrew Kim
Date: October 12, 2026

I worked with Codex to review the differences in our output and that of the original implementation of CT-FIRE. I analyzed both the MATLAB source code as well as the several variables from a runtime test on `real1.tif`, and compared them to the old parity analysis file to see what we had missed before. This document is a list of findings and recommended approaches to get our implementation as close to the original as possible.

## Conclusion

Substantially closer parity is achievable. The remaining differences include reproducible algorithm and indexing mismatches, rather than only floating-point drift. None of the identified issues establishes a fundamental barrier to reproducing MATLAB's fiber geometry and connectivity on a fixed reference environment.

Exact final paths and total path length have now been demonstrated for the captured `real1` configuration, including reconstruction from its raw TIFF (see the reconstruction follow-up below). This does not establish equality of every output field or universal parity. Individual fixes can interact, and their effect must be measured using matching intermediate inputs. Universal byte-for-byte agreement across platforms is a harder target than reproducing fiber paths, connectivity, and measurements for a defined MATLAB version.

Fiber count agreement is not a percentage of fiber parity. Two networks can have similar counts while containing different paths and connections.

## Implementation progress — October 19, 2026

The original measurements below remain a historical baseline. The first implementation pass now addresses the following issues:

- **Issue 1:** native nucleation detection perturbs a private working buffer, preserving the caller's DSM, including read-only inputs.
- **Issue 2:** distance smoothing retains the singleton depth axis. Residual convolution differences in issue 3 remain.
- **Issues 4 and 5:** initial and continuation LMP traversal follow MATLAB's `[x,y,z]` order; initial distance pruning retains discarded candidates for subsequent pair comparisons. Fractional `thresh_LMPdist` support remains outstanding.
- **Issue 6:** the 2D boundary check follows MATLAB's face and z-running-edge neighbor lists, including its asymmetric corner offsets. This reduction targets the positive LMP thresholds used by the fixtures; it does not establish parity for zero-threshold padded-volume candidates or 3D extraction.
- **Issue 10:** crosslink search uses MATLAB's column-first image-plane order.
- **Issue 13:** the final-endpoint sample position in `fiberproc` is corrected.

With the same runtime capture, one OpenMP thread, and a freshly rebuilt backend:

| Metric | Before these changes | After these changes | MATLAB reference |
| --- | ---: | ---: | ---: |
| DSM maximum absolute error | 0.0459185 | 7.15256e-7 | 0 |
| DSM pixels changed by nucleation detection | 262,143 | 0 | 0 |
| Shared nucleation points | 358 | 358 | 358 |
| Extension fibers on exact MATLAB inputs | 844 | 868 | 873 |
| Exact shared extension paths on those inputs | 585 | 868 | 873 |
| Processed fibers on MATLAB extension output | 192 | 189 | 191 |
| Exact shared processed paths on that input | 175 | 182 | 191 |
| Full captured-input extraction: final fibers | 182 | 189 | 191 |
| Full captured-input extraction: exact shared final paths | 5 | 179 | 191 |
| Full captured-input extraction: total length, pixels | 7,511.64 | 7,851.97 | 7,921.06 |

Every extension path produced in the exact-input experiment now matches a MATLAB path, allowing whole-path reversal, but five MATLAB paths are missing. Early coordinate merging and duplicate removal remain candidates for those missing paths; their cause has not yet been isolated. Full captured-input length error falls from 5.17% to 0.87%, with 179 of 191 exact final paths shared. These are fixture-specific measurements, not a claim of complete parity.

Regression coverage includes immutable/repeatable nucleation inputs, singleton-axis attenuation, non-greedy LMP pruning, face-neighbor selection near a corner, and final-endpoint direction sampling. The targeted suite completed with **46 passed, 10 skipped**; the skips include unavailable MATLAB function references. This is not a full-suite result. No tolerance was loosened. Ruff checks on the new Python files and `git diff --check` also passed.

Reproduce the runtime measurements after building the native backend:

```bash
make -C src/ctfire_py/CPP
OMP_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python \
  scripts/compare_matlab_runtime.py /path/to/runtime_reference.mat \
  --output /tmp/parity-result.json
```

The comparison command requires `h5py`, uses the capture's effective parameters, and records the input checksum and loaded backend path. `--backend-dir /path/to/build` selects an alternate compiled backend. For the targeted tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
  tests/test_parity_preprocessing.py tests/test_cpp_functions.py \
  tests/test_fire_2d_angle.py -q --disable-warnings
```

The subsequent full-suite pass also addresses issue 18's reconstruction cutoff
index and the default scale count for non-power-of-two images, and passes the
explicitly converted float64 image into the native transform. Three focused
tests exercise coefficient-rank selection, the zero/full-retention endpoints,
finest-scale exclusion, and transform parameters without requiring a MATLAB
fixture. These source fixes do not by themselves establish reconstruction parity
for images whose MATLAB reconstruction references are absent.

Remaining work includes floating-point arithmetic, vertex identity/deduplication timing, gap-link candidate semantics, processing tie order and hashing, numerical reconstruction comparisons, and validation on additional captures.

### Full-suite validation

After installing both `curvelops` and `gui` extras, the full suite completed with
**147 passed, 32 skipped, 0 failed** on October 19, 2026. Curvelet tests were
enabled, the napari import smoke test passed, and all 11 newly added regression
cases ran. No test tolerance was loosened and no new skip or xfail was added.
The targeted Ruff checks for the new Python files and reconstruction changes,
relative documentation links, and `git diff --check` passed.

The 32 skips are missing fixtures, not missing installed backends:

| Cases skipped | Missing input/reference |
| --- | --- |
| 4 | `tests/test_images/2B_D9_ROI1.tif` |
| 8 | Six per-function MATLAB references under `cpp_test_files`; two additional pipeline tests reuse the missing nucleation references |
| 4 | MATLAB reconstructed-image references under `ct_fire_test_files` |
| 16 | Four final CT-FIRE MATLAB references, each used by four comparison cases |

Exact command used (with `MPLCONFIGDIR` pointing to a writable cache):

```bash
TMEQ_RUN_CURVELETS=1 TMEQ_VALIDATE_MATLAB=1 QT_QPA_PLATFORM=offscreen \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tme-parity-mpl OMP_NUM_THREADS=1 \
  uv run --extra curvelops --extra gui pytest tests/ -q -rs --disable-warnings
```

`TMEQ_VALIDATE_MATLAB=1` has no effect on this branch because no implementation
reads it. Available reference tests already run; missing references remain
unvalidated. The final run used normal pytest plugin loading and access to uv
and napari caches. An earlier sandbox-only run skipped napari because the GUI
extra was not installed at collection time; that skip is absent from the final
result.

## Vertex-identity investigation and fix — October 19, 2026

This follow-up addresses the first remaining fix discussed after the initial
implementation pass: vertex identity and endpoint deduplication (issue 11 in the
inventory below). The earlier measurements are retained as historical baselines.

### Trace and confirmed failure point

An instrumented temporary build logged every generated branch before duplicate
removal, then logged every branch discarded by the endpoint-pixel key. It used
the capture's exact DSM and ordered nucleation points, isolating extension from
preprocessing. C++ generated 1,005 raw branches and discarded 137, leaving 868.
Every missing MATLAB path was present in the raw branches and was removed at
that duplicate-removal step. No direction-arithmetic or radius change was needed
to recover them.

| Missing MATLAB fiber, one-based | Start `(x,y)` | Free-tip `(x,y)` | Lost length, pixels |
| --- | --- | --- | ---: |
| 71 | (35,67) | (24,57) | 16.0944 |
| 109 | (54,437) | (43,403) | 38.0554 |
| 141 | (71,477) | (82,499) | 26.3939 |
| 165 | (84,28) | (68,1) | 33.5202 |
| 721 | (443,265) | (443,260) | 5.5765 |
| **Total** | | | **119.6404** |

The total exactly accounts for the extension-length shortfall:
`9299.590353665404 - 9179.94994398473 = 119.640409680674` pixels, within
floating-point summation precision.

For example, MATLAB fibers 70 and 71 take different routes between the same
endpoint pixels:

```text
70: (35,67) -> (31,63) -> (30,61) -> (28,58) -> (24,57)
71: (35,67) -> (34,63) -> (31,61) -> (28,58) -> (24,57)
```

MATLAB retains endpoint ID pairs `(32,671)` and `(32,675)`. The distinct free-tip
IDs occupy the same pixel, so MATLAB's vertex-ID check keeps both. C++ previously
reduced both pairs to the same pixel-offset key and deleted fiber 71. All five
missing fibers have this pattern, and none of their free tips is a nucleation
point. A temporary diagnostic probe retained those branches and recovered all
873 MATLAB paths while discarding only the other 132 duplicate branches.

The next construction step also merged every coincident coordinate through
`index_map`. MATLAB's 4,122 vertex records occupy only 4,029 distinct pixels;
coordinate equality therefore cannot substitute for vertex identity at this
stage. Keeping five extra paths alone would not correct the graph.

### Implementation

[extend_xlink_native.cpp](../src/ctfire_py/CPP/extend_xlink_native.cpp) now:

1. Records explicit crosslink termination on each generated branch.
2. Initializes vertex IDs from the ordered nucleation list and allocates a new
   vertex for every extension step, including coincident pixels.
3. Reuses an existing nucleation ID only for an explicit crosslink termination.
4. Removes duplicate fibers using unordered endpoint **vertex-ID** pairs,
   retaining the original serial branch order.
5. Compacts unused vertices in their original order and remaps fiber indices,
   matching `trimxfv` without merging coordinates.
6. Rebuilds radii and vertex/fiber connectivity. Fiber-neighbor lists concatenate
   MATLAB's per-vertex `setdiff` result, preserving repeated contributions across
   vertices rather than globally deduplicating them.

The pixel index map and intermediate `F_init` reconstruction are removed.
Coincident-coordinate merging remains the responsibility of the later processing
stage. Reverse duplicates between genuinely shared nucleation vertices are still
removed.

### Capture validation and scope

The comparison script now checks ordered graph arrays in addition to whole-path
multisets. Its HDF5 reader also handles MATLAB's empty-array encoding; without
that check, empty endpoint lists decode incorrectly as `[-1,-1]` and report false
connectivity differences.

| Exact-input extension metric | Before identity fix | After identity fix | MATLAB |
| --- | ---: | ---: | ---: |
| Fibers | 868 | **873** | 873 |
| Exact shared paths | 868 | **873** | 873 |
| Vertex records | Prematurely merged by pixel | **4,122** | 4,122 |
| Total length, pixels | 9,179.949944 | **9,299.590354** | 9,299.590354 |

After axis and index conversion, the following compare exactly, including order:

- Vertex coordinates (`Xz`).
- Every fiber's vertex IDs (`Fz.v`).
- Radius values (`Rz`).
- Vertex membership, endpoint, and neighbor lists (`Vz.f`, `Vz.fe`, `Vz.vall`).
- Fiber connectivity lists (`Fz.f`).

This establishes exact extension output for the captured inputs, not universal
or full-pipeline parity. On the captured FIRE input with Python preprocessing,
the final output changes from 179 to **180 of 191** exact shared paths, retains
**189 fibers**, and has total length **7,872.836608** pixels versus MATLAB's
7,921.061988 (about **0.61%** error). Preprocessing still has its documented small
DSM differences. Independently, `fiberproc` on MATLAB's exact extension output
still produces 189 fibers with 182 exact shared final paths; this fix does not
address those processing discrepancies.

Four portable regression cases cover distinct vertex IDs at coincident free
tips, two different routes sharing both endpoint pixels, shared nucleation IDs
with reverse duplicate removal, and removal of unused nucleation vertices.
These tests do not depend on the external runtime capture.

Validation for this follow-up: the full suite completed with **150 passed,
32 missing-fixture skips, and 0 failures**. The fourth regression case was added
after full-suite collection; the complete four-case regression module then
passed separately. Targeted Ruff checks and `git diff --check` passed. No test
tolerance or skip condition was changed.

## Fiber-processing merge-order investigation and fix — October 19, 2026

This follow-up addresses the next remaining parity problem: `fiberproc` produced
189 fibers with 182 of MATLAB's 191 final paths, even when given MATLAB's exact
extension graph. It resolves the endpoint-list ordering portion of issue 15;
other potential ordering and numerical differences remain relevant to broader
validation.

### Direct MATLAB stage tracing

The installed MATLAB R2026a was run in batch mode on a temporary instrumented
copy of the original `fiberproc.m`. The run restored the runtime capture's
MATLAB path, explicitly selected the temporary copy, and used the captured
`Xz`, `Fz`, `Rz`, image size, and parameters. The resolved `calc_fiberlen` was
the CT-FIRE `FIRE/fiberproc` version. The original source files were not edited.
The replayed MATLAB final paths agree with the saved capture.

Both implementations saved graphs after initial trimming, initial overlap
cleanup, each intersection-linking and overlap-cleanup pass, gap linking, and
short-fiber removal. Individual merge decisions were also logged. Graph
comparisons account for C++ retaining unused vertices: active vertex IDs are
renumbered in their original order before comparison.

| Stage | MATLAB fibers | C++ before fix | Exact shared paths before fix |
| --- | ---: | ---: | ---: |
| Initial trim | 873 | 873 | 873 |
| Initial `remove_repeat` | 863 | 863 | 863 |
| First `fiberlink` pass | 534 | 534 | **524** |
| After five `fiberlink`/`remove_repeat` passes | 533 | 533 | **523** |
| `fiberlinkgap` | 476 | 477 | 464 |
| Final `fiberremove` | 191 | 189 | 182 |

The first path discrepancy is therefore in intersection linking, not initial
overlap cleanup or gap linking. Equal fiber counts concealed it.

### Root cause and specific decisions

MATLAB's `mergefiber` explicitly sorts endpoint membership lists after replacing
the removed fiber ID. C++ removed IDs and appended the survivor without sorting.
The representation first diverged after the first merge, at pixel `(20,340)`:
MATLAB retained endpoint list `[1,41]`, while C++ stored `[41,1]`.

After merging fibers 6 and 7, pixel `(10,32)` similarly had `[6,31]` in MATLAB
and `[31,6]` in C++. At the twelfth merge, MATLAB selected `(6,31)` and retained
fiber 6; C++ selected `(31,6)` and retained fiber 31. IDs in this example refer
to the fiber list after initial cleanup; coordinates and IDs are one-based as
in MATLAB.

The pair selection is not just a numbering detail. The survivor, fiber
orientation, and position in the subsequent traversal can change. Later
direction estimates and merge decisions then join different branches. The
gap-linking count discrepancy was downstream of these earlier decisions.

A temporary C++ probe sorting the updated endpoint lists restored all ordered
paths at every captured processing stage, including 191 of 191 final paths.
It also matched active coordinates, renumbered fiber vertex IDs, and active
vertex membership and endpoint lists. This isolates a concrete cause rather
than attributing the difference to floating-point drift.

### Implemented behavior and regression

[fiberproc_native.cpp](../src/ctfire_py/CPP/fiberproc_native.cpp) now reproduces:

- `sort(setdiff(V(vm).fe,[f1 f2]))` at the merged intersection.
- `sort([setdiff(V(ve).fe,f2) f1])` at the transferred endpoint. The `setdiff`
  contribution is deduplicated before appending, preserving MATLAB's treatment
  of an already-present survivor ID, including closed-loop cases.
- Sorted unique fiber membership lists, matching MATLAB's `unique` update.

The portable regression in
[test_fiberproc_merge_order.py](../tests/test_fiberproc_merge_order.py) joins
three segments `A--B`, `B--C`, and reversed `D--C`. The second endpoint update
must become `[0,2]`, rather than `[2,0]`, so MATLAB's survivor and ordered output
`A--B--C--D` are preserved. The test failed against the old compiled backend and
passed after rebuilding with the fix. No external MATLAB fixture is needed.

### Current results and remaining limits

On MATLAB's exact extension coordinates, processing now returns:

- **191 fibers and 191 exact shared paths**.
- **7,921.061988 pixels** total length, exactly matching MATLAB's path-length
  calculation in the comparison command.
- **3,257 active vertices**, with exact ordered active coordinates, renumbered
  fiber vertex IDs, radii, membership lists, and endpoint lists.

The native-coordinate control experiment processes the exact C++ extension
output in its normal zero-based row/column frame. It also reproduces all 191
paths and the same total length. Thus the remaining discrepancies in this
captured-input experiment are introduced before processing, rather than by
that coordinate-frame conversion.

Starting from the captured FIRE image and using Python preprocessing, the final
output now has **191 fibers, 188 of 191 exact shared paths**, and total length
**7,921.213811 pixels** (about **0.00192%** error). The DSM still has maximum
absolute error `7.15256e-7`; numerical and decision differences before processing
need further investigation. This is not a raw-image curvelet reconstruction
comparison.

Full raw output arrays are not yet identical: C++ still returns 4,122 vertex
records, including unused vertices, while MATLAB returns the 3,257 active
records. The comparisons above explicitly normalize that difference. Final
`V.vall` multiplicity and other unlisted metadata have not been established as
equal. Other images may still expose the separate candidate-order, tie-breaking,
hashing, or precision issues in the inventory.

Validation: the complete curvelet- and GUI-enabled suite finished with
**152 passed, 32 missing-fixture skips, and 0 failures**. The focused merge-order,
vertex-identity, and preprocessing regressions passed (13 cases). The new
merge-order regression was verified to fail on the previous native binary and
pass after rebuilding. Targeted Ruff checks, documentation links, and
`git diff --check` passed. No tolerance, skip condition, or reference output was
changed.

## Smoothing-precision investigation and fix — October 19, 2026

This follow-up addresses issue 3. After the merge-order fix, Python preprocessing
still produced three different final paths, even though processing on the exact
MATLAB DSM reproduced all 191 paths. The investigation and implementation below
close that gap for the captured `real1` FIRE input. Exact arithmetic across all
MATLAB configurations and implementations remains a separate validation target.

### Source of the three divergent paths

MATLAB was run on the capture to save the original distance transform, Gaussian
kernel, and each successive `imfilter` output. The distance transform and the
double-precision kernel matched Python exactly. The differences began inside
filtering, not in distance calculation, nucleation order, or processing.

All three extension trajectories diverged when small single-precision value
differences became ties. For example, at extension point `(223,280)`:

| Boundary candidate `(x,y)` | MATLAB DSM | Previous Python DSM |
| --- | ---: | ---: |
| (226,281) | **4.9578309059143066** | 4.9578304290771484 |
| (226,280) | 4.9578304290771484 | 4.9578304290771484 |

MATLAB chose the higher value. Python saw equal values and chose the earlier
candidate. The other two first divergent choices were `(387,436)` versus
`(386,436)`, and `(507,279)` versus `(506,279)`. Their search radii agreed at
the first divergence points. The three altered extension paths propagated to
final fibers 68, 121, and 164 in this comparison (MATLAB one-based coordinates
and fiber numbering).

Previously, `smooth` promoted the single-precision distance image to double,
then cast it back only after filtering. Merely requesting float32 output from
SciPy's convolution did not reproduce MATLAB: coefficient and accumulation
precision also affect the result.

### Implemented arithmetic

[fire_2d_angle.py](../src/ctfire_py/fire_2d_angle.py) now preserves float32 input
and applies each separable filter using:

1. The existing double-precision Gaussian construction and normalization.
2. A float32 coefficient cast before filtering single-precision images.
3. Zero padding and deterministic increasing-offset tap traversal.
4. Separate float32 multiplication and addition, with rounding at each step;
   no wider accumulator or fused multiply/add substitution.
5. Float32 output passed into the next axis filter.

The existing singleton-depth convolution remains in place. Double inputs retain
the SciPy/double path; integer inputs continue to be promoted to double, preserving
the existing raw-image preprocessing convention rather than implementing generic
integer `imfilter` semantics. The original MATLAB guard `if all(sigma)==0` is now
also reproduced for sigma vectors: any zero component returns the input without
filtering, including mixed zero/nonzero vectors.

No LMP tolerance, candidate comparison, search radius, random perturbation, or
reference output was adjusted to force a match.

### Broader reference fixtures and numerical limits

The original MATLAB `smooth.m` generated 98 portable fixtures: seven shapes,
seven sigma configurations, and both single and double inputs. Shapes include
rectangular 2D images, leading/middle singleton axes, row vectors, and small
volumes. Sigma cases include 0, 0.3, 0.5, 1, 2, positive anisotropic vectors, and
mixed zero/nonzero vectors. Inputs are deterministic nonnegative values.

[matlab_smoothing.json](../tests/test_results/smoothing_test_files/matlab_smoothing.json)
records the MATLAB version, original source hash, generator code, inputs, and
expected outputs. The tests require neither a MATLAB installation nor the
external real-image capture.

Three additional compact distance-map patches preserve the actual candidate
pairs responsible for the `real1` divergences. Their expected candidate scores
come from the captured MATLAB DSM, with the capture checksum retained in the
fixture. The patches include the full filter support for the two queried pixels;
their regressions require exact scores and the original strict preference, not
a tolerance-based tie.

The wider sweep also exposed shape- and kernel-dependent accumulation behavior
inside MATLAB's optimized filtering. A fixed forward or reverse accumulation
order, or a fused-operation model, did not reproduce every tested pass. A
shape-specific reversal heuristic was investigated and rejected rather than
hardcoded into production.

Consequently, the assertions distinguish the supported evidence:

- The single-precision singleton-depth, scalar-0.3 configuration corresponding
  to this capture is tested with exact equality.
- Across all 49 single-precision fixtures, 35 match bitwise and the remaining
  14 differ by at most **two float32 ULPs**. General fixture tests enforce that
  explicit bound; they do not claim universal bitwise agreement.
- Double fixtures allow a small double-roundoff bound (four machine epsilons
  in relative and absolute tolerance), preserving the existing SciPy path.
- Tests also verify dtype preservation, input immutability, and the existing
  promotion behavior for integer images.

These are new, independently generated numerical-reference tests. Existing
parity tolerances were not loosened. The remaining two-ULP differences in other
configurations can still affect LMP decisions and remain a broader parity risk.

### Captured-input result

| Metric | Before this fix | After this fix | MATLAB |
| --- | ---: | ---: | ---: |
| DSM differing pixels | 56,702 | **0** | 0 |
| Maximum DSM absolute error | 7.15256e-7 | **0** | 0 |
| Shared nucleation points | 358 | **358** | 358 |
| Final fibers | 191 | **191** | 191 |
| Exact shared final paths | 188 | **191** | 191 |
| Total length, pixels | 7,921.213811 | **7,921.061988** | 7,921.061988 |

This establishes exact final paths and total length when starting from the
captured MATLAB FIRE image. It does not establish Python curvelet reconstruction
parity from the raw image, equality of unused output-array entries or all
metadata, or exact smoothing for every image/sigma configuration.

Validation: the complete curvelet- and GUI-enabled suite finished with
**254 passed, 32 missing-fixture skips, and 0 failures**. The smoothing module's
102 cases also passed separately: 98 MATLAB references, one integer-promotion
check, and three strict LMP preference regressions. Targeted Ruff checks,
documentation links, and `git diff --check` passed. No existing tolerance, skip
condition, or MATLAB reference output was changed.

## Raw-image reconstruction investigation (trace only)

The next comparison starts from `tests/test_images/real1.tif`, rather than the
captured reconstructed FIRE input. No production implementation was changed
during this investigation.

### Reference and controls

The original MATLAB `CTrec_1.m` operations were rerun with the captured
`matlab_path`, `pct=0.2`, `SS=3`, and original-image mask `raw > 5`.
MATLAB resolves both `fdct_wrapping` and `ifdct_wrapping` to
`CurveLab-2.1.2/fdct_wrapping_matlab`, **not** its separate C++ implementation.
The newly reconstructed, masked image equals `squeeze(runtime_reference.im)`
exactly: maximum error zero, with no foreground-mask differences. The raw TIFF
SHA-256 is `dfd22ea3bb4b733a687a0430811ee1837e1faee0f33350c0b121ed2a3dfaadfb`.
Python uses curvelops 0.23.4 (commit
`d6dc5fde8bcf399e57f81c5beb38449eb8863e45`) and its native CurveLab wrapper.

For this uint8 grayscale image, Python's float32 input conversion preserves
every intensity exactly, and its original-image mask matches MATLAB. Both
implementations use six scales, 16 coarse angles, a finest wavelet, a strict
`abs(coefficient) > cutoff` test, and selected MATLAB scales 3–5. The earlier
threshold-index and scale-count fixes are already present.

### First divergence: the transform backend

The forward transforms produce different coefficient grids. For example,
MATLAB scale 2 begins with wedges of shapes `(18,22)`, `(16,22)`, `(16,22)`,
and `(18,22)`; curvelops reports `(23,19)` for each of those four wedges.
Its wrapper reverses native matrix dimensions when exposing coefficients to
NumPy, but transposing those shapes still does not reconcile their sizes.

The supplied MATLAB and C++ CurveLab sources implement different discrete
window sampling and wrapping geometry. MATLAB constructs low-pass ramps from
integer-derived window lengths and has separate corner/interior wedge sizes.
The C++ implementation samples its low-pass window using continuous scale
ranges and forces wedge dimensions to odd lengths. The curvelops wrapper
exposes the native implementation; it does not implement the MATLAB routines.

| Measurement | MATLAB | Current Python |
| --- | ---: | ---: |
| Total coefficients | 739,001 | 748,041 |
| Threshold cutoff at `pct=0.2` | 5.505283059856382 | 5.579389033730728 |
| Positive pixels after reconstruction and mask | 37,429 | 37,446 |

The masked images have maximum absolute difference **3.07394739**, RMSE
**0.29564023**, and correlation **0.99976752**. Despite the high correlation,
**907 pixels differ in the `image > 0` foreground decision used by FIRE**.

Controls rule out a cutoff-only or axis-layout explanation:

- With all coefficients retained, each backend reconstructs the original raw
  image to maximum absolute error about `1.14e-13`.
- Reconstructing each scale separately, without thresholding, still differs
  between backends. Maximum errors across scales 1–6 are approximately
  `0.6872`, `0.9340`, `0.7542`, `0.6101`, `0.2295`, and `0.1727`.
- Giving MATLAB the Python cutoff still leaves masked reconstruction RMSE
  **0.29228404** relative to Python. Matching the cutoff alone is insufficient.

### Downstream consequence and proposed fix

Feeding the Python reconstructed/masked image into the current extraction code
changes 9,146 DSM pixels (maximum error about 2.95416). It yields 361 nucleation
points, of which 319 match MATLAB's 358 points. Both outputs contain 191 final
fibers, but only **85 final paths match exactly**; Python's total length is
**7,855.884937**, versus MATLAB's **7,921.061988** pixels. The previous 191/191
exact result remains valid when starting from the captured MATLAB FIRE input.
Equal final counts therefore conceal substantial reconstruction-induced path
differences.

This is fixable by implementing the MATLAB wrapping algorithm's forward and
inverse operations, including its window construction, rounding, corner wedges,
indexing, and FFT normalization. The source is available locally; no additional
user files are needed for this investigation. A NumPy/SciPy implementation can
avoid requiring MATLAB at application runtime. Swapping coefficient order,
adjusting only the cutoff, or tuning extraction thresholds cannot reproduce
the measured MATLAB transform.

Before integration, generate durable MATLAB coefficient and reconstruction
fixtures for deterministic inputs, including rectangular and non-power-of-two
shapes. Validate forward coefficients, inverse reconstruction, threshold masks,
and then raw `real1` final paths. FFT rounding can still prevent bitwise
coefficient equality, so numerical agreement and downstream decisions must be
checked separately. General RGB/high-bit-depth input normalization also remains
outside this uint8 grayscale trace.

The diagnostic MATLAB scripts, coefficient arrays, scale controls, and Python
comparisons for this run are in `/tmp/tme-reconstruction-trace`; these temporary
files are not committed fixtures. This was a numerical investigation, not a
full-suite test run or an implementation fix.

## MATLAB wrapping reconstruction fix

The trace above is the pre-fix baseline. Reconstruction now uses
[matlab_wrapping.py](../src/ctfire_py/matlab_wrapping.py), a NumPy implementation
adapted from Laurent Demanet's original MATLAB wrapping routines. It implements
the complex transform with a finest wavelet (`is_real=0`, `finest=2`), matching
`CTrec_1`; real-valued coefficient packing and a finest curvelet are outside
its scope. Other CurveAlign transforms continue to use curvelops.

The port shares wrapping geometry between forward and inverse operations,
including MATLAB half-away-from-zero rounding, quadrant-dependent even-sized
wedge origins, corner windows, periodic wrapping, centered FFT normalization,
and inverse assignment order for repeated corner indices. Reconstruction
retains the previous strict threshold, rank indexing, scale selection, and
original-image mask. Package exports no longer require curvelops to expose
CT-FIRE reconstruction or extraction; the extraction backend is still required.

### Durable numerical references

Ten independent MATLAB cases are stored as pickle-free NPZ files with source
and fixture checksums in
[reconstruction_test_files](../tests/test_results/reconstruction_test_files/README.md).
The generator and conversion script are included. Cases cover square,
rectangular, odd-sized and non-power-of-two images, 8/16/32 coarse angles,
zero input, and an impulse. Each checks forward coefficient values and shapes,
inverse reconstruction from MATLAB-filtered coefficients, and full round trips.
Seven cases additionally exercise the actual CT-FIRE reconstruction function
with curvelops unavailable. A package-export regression verifies that dependency
change. Existing rank-selection tests now mock the new backend; numerical
references independently cover the default scale count and angular layout.

Across these ten cases, maximum forward coefficient error is about `1.71e-13`,
filtered inverse error about `2.49e-14`, and round-trip input error about
`7.17e-14`. Tests use `rtol=2e-13`, `atol=5e-13` for FFT arithmetic, and require
exact coefficient shapes and wedge counts. Existing parity tolerances were
not loosened.

The symmetric impulse exposes a remaining numerical limitation: at the 20%
cutoff, theoretically equal magnitudes straddle the strict threshold differently
after tiny FFT rounding differences. Its final reconstructed-image difference
is about `4.23e-5` despite coefficient agreement at about `1.15e-16`. This case
tests forward and inverse transforms independently; it is explicitly excluded
from the strict-threshold integration parameter list. No skip/xfail or relaxed
integration tolerance conceals that limitation. Matching FFT implementations
or defining a new shared tie policy would be separate work; changing the
MATLAB strict-threshold policy is not part of this fix.

### Actual raw-image CT-FIRE result

The comparison command now exercises `ct_fire` itself, including input
conversion and masking:

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python \
  scripts/compare_matlab_reconstruction.py /path/to/runtime_reference.mat \
  tests/test_images/real1.tif --fraction 0.2 --scales 3 --threshold 5 \
  --output /tmp/reconstruction-parity.json
```

The raw image/capture and parameters must come from the same MATLAB run.
The command records their checksums and compares reconstructed input, DSM,
nucleation, final paths (allowing whole-path reversal), and total path length.
It requires `h5py` and the compiled extraction backend, with no MATLAB runtime.

| Metric | Before backend replacement | After replacement | MATLAB |
| --- | ---: | ---: | ---: |
| Reconstructed masked image maximum absolute error | 3.07394739 | **8.52651e-14** | 0 |
| Reconstructed masked image RMSE | 0.29564023 | **3.71744e-15** | 0 |
| Foreground mask differing pixels | 907 | **0** | 0 |
| DSM differing pixels | 9,146 | **0** | 0 |
| Nucleation points | 361 | **358** | 358 |
| Shared nucleation points | 319 | **358** | 358 |
| Final fibers | 191 | **191** | 191 |
| Exact shared final paths | 85 | **191** | 191 |
| Total length, pixels | 7,855.884937 | **7,921.061988** | 7,921.061988 |

This extends the earlier captured-input result to the actual raw `real1.tif`
pipeline for this configuration. It does not prove every measurement/metadata
field, other captures, RGB/high-bit-depth input conversion, or platform-wide
bitwise equivalence. Broader missing MATLAB fixtures remain outstanding.

Validation: the full curvelet- and GUI-enabled suite completed with **292
passed, 32 missing-fixture skips, and 0 failures** (21 warnings). The focused
reconstruction run completed with **41 passed**: 38 new transform/integration/
export cases and three existing rank-selection cases. The raw-image comparison
command separately produced the results above. Targeted Ruff checks and
`git diff --check` passed. No new skip/xfail or relaxed existing tolerance was
introduced.

## Sources and comparison method

- MATLAB source: `CurveletsSourceCode_CurveAlign5.0_CT-FIRE3.0/CT-FIRE3.0_src`, including its `FIRE` subfolder.
- Runtime capture: `runtime_reference.mat`.
- Supporting final output: `tests/test_images/ctFIREout/ctFIREout_real1.mat`.
- The capture contains `im`, `p`, `ims`, `imt`, `dsm`, ordered `xlink`, extension outputs, inputs and outputs of `fiberproc`, and MATLAB environment information.
- The current C++ source was rebuilt using the repository macOS Makefile into `/tmp/fiber_backend.cpython-313-darwin.so`. Comparisons used one OpenMP thread and Python 3.13. The pre-existing project binary predates the C++ source and was not used for the runtime measurements below.
- HDF5 arrays were transposed into MATLAB logical dimensions. Fiber indices were converted from one-based to zero-based. Spatial axes were mapped explicitly when comparing paths.

This is a CT-FIRE run originating from `real1.tif`, with curvelet reconstruction before FIRE. The captured FIRE input is a `1 x 512 x 512` volume, and its effective `thresh_im2` is **0**, rather than the raw-image threshold of 5 in `ctfP.value`. Relevant parameters include cityblock distance, `sigma_d=0.3`, `s_xlinkbox=8`, `thresh_LMP=0.2`, `thresh_LMPdist=2`, `lam_dirdecay=0.5`, `s_fiberdir=4`, and `thresh_flen=15`.

The current extraction comparison starts from the captured MATLAB FIRE input. It bypasses Python curvelet reconstruction and does not establish end-to-end reconstruction parity.

For isolated extension comparisons, C++ received the saved MATLAB DSM and nucleation points in their original order. For isolated processing, it received MATLAB's extension coordinates and radii, retaining native coordinate values and column order, with fiber indices converted to zero-based. This separates processing logic from differences in extension and coordinate hashing introduced by upstream conventions.

An exact shared path means the same ordered sequence of integer image-plane coordinates, allowing reversal of the entire path. Paths were compared as multisets. This is stricter than raster overlap but does not establish equality of all graph metadata or radius values.

## Runtime evidence

| Experiment | Fiber count | Total path length, pixels | Exact paths shared with corresponding MATLAB stage |
| --- | ---: | ---: | ---: |
| MATLAB extension | 873 | 9,299.59 | Reference |
| C++ extension on exact MATLAB DSM and points | 844 | 8,410.91 | 585 of 873 |
| C++ extension on perturbed MATLAB DSM, same points | 834 | 8,418.96 | 389 of 873 |
| MATLAB final `fiberproc` output | 191 | 7,921.06 | Reference |
| C++ `fiberproc` on MATLAB extension output | 192 | 7,870.38 | 175 of 191 |
| Current extraction stages on captured FIRE input | 182 | 7,511.64 | Not measured |

The current extraction produces 4.71% fewer processed fibers and 5.17% less total length in this run. With identical extension inputs, C++ produces 3.32% fewer extension fibers but 9.56% less extension length. These are distinct measurements, not interchangeable parity scores.

Other measurements:

- Image thresholding matches the capture exactly: **0 differing pixels**.
- Nucleation sets match exactly: **358 of 358 points**, both on the saved MATLAB DSM and in the current extraction preprocessing. This establishes set agreement for this fixture, not every configuration or thread count.
- Current 2D DSM smoothing differs at **70,826 pixels**, with maximum absolute error **0.0459185**.
- Applying the existing Python smoothing to the full singleton-depth volume reduces maximum error to **7.15256e-7**, with **56,702 differing pixels**.
- A diagnostic variant using single-precision arrays at each convolution still differs at **54,990 pixels**, with maximum error **9.53674e-7**. Dimensionality is a major discrepancy; changing precision alone does not establish exact smoothing parity.
- RNG perturbation changes `ceil()` at **335 foreground pixels**, defining foreground here as saved DSM values at least `thresh_LMP`. It changes radii at 191,653 pixels overall, but that larger number includes background pixels and should not be interpreted as affected fiber steps.

## Issue inventory and fixability

“Confirmed” means the code differs or the runtime experiment establishes a discrepancy. It does not mean the contribution of that individual issue to final output has been isolated. “Needs validation” identifies a remaining comparison target rather than a proven cause.

### 1. Nucleation detection mutates the DSM used by extension

**Status: confirmed in source and runtime. Fixable: yes. Priority: high.**

MATLAB `findlocmax` adds random noise to its local `d` and returns only nucleation coordinates. The caller subsequently passes the original `dsm` to `extend_xlink`.

[findlocmax_native.cpp](../src/ctfire_py/CPP/findlocmax_native.cpp) instead writes perturbations into the caller's mutable array. [fire_2d_angle.py](../src/ctfire_py/fire_2d_angle.py) reuses that array for extension and processing. This changes candidate values, ties, and radii even when nucleation coordinates match.

**Fix:** use a private working buffer for nucleation perturbation and preserve the original DSM. Validate both nucleation equality and extension paths. In the isolated runtime experiment, removing noise increases shared paths from 389 to 585, but does not eliminate extension divergence. Final length does not improve monotonically in that experiment, so path improvement should not be presented as a guaranteed aggregate improvement.

### 2. Smoothing omits the singleton depth axis

**Status: confirmed in source and runtime. Fixable: yes. Priority: high.**

MATLAB smooths the three-dimensional `1 x H x W` distance image. Zero-padding along the singleton axis attenuates its values when `sigma_d` is nonzero. Python reduces the distance transform to 2D before smoothing and omits that convolution.

**Fix:** reproduce MATLAB's logical dimensions, axis order, kernels, and boundary conditions. Adding the singleton-axis convolution substantially reduces DSM error in this capture. Thresholding and nucleation happen to remain matched; that does not make the DSM difference harmless for extension.

### 3. Smoothing arithmetic and intermediate precision

**Status: fixed for the captured FIRE configuration; up to two float32 ULPs remain in broader smoothing fixtures.**

MATLAB filters a single-precision distance image through successive `imfilter` calls. Python previously promoted the input to double and cast back after convolution. The smoothing follow-up above preserves single-precision coefficients, multiplication, addition, and intermediate results. Kernel construction and optimized convolution arithmetic can still differ in other configurations.

**Implemented fix and remaining work:** individual filtering passes established the precision mismatch, and the captured DSM is now exact. Simply changing the output dtype was insufficient. Continue validating optimized accumulation order and kernel generation across other inputs and MATLAB versions; small residuals can still change candidate decisions.

### 4. LMP sorting and maximum-value tie-breaking

**Status: confirmed in source. Fixable: yes. Priority: high.**

MATLAB `findLMP` uses `unique(LMP,'rows')`, sorting candidates by `[x,y,z]`. Extension selects the first maximum DSM value after directional filtering. C++ traverses candidates row-first and retains the first qualifying maximum encountered.

**Fix:** explicitly reproduce sorting, deduplication, filtering, and first-maximum selection. A traversal-order experiment that worsens aggregate count does not disprove MATLAB's documented order; other mismatches can interact with it.

### 5. Initial LMP distance pruning uses a different algorithm

**Status: confirmed in source. Fixable: yes. Priority: high.**

MATLAB compares all pairs in the original sorted candidate list and marks later close candidates for removal. A candidate already marked for removal still participates in later comparisons. C++ compares each new candidate only against retained branches.

For a chain A–B–C where adjacent distances are below the threshold but A–C is not, MATLAB can remove B and C, while the greedy C++ rule retains A and C.

**Fix:** match MATLAB's pairwise marking operation before starting branches. Also retain the numeric type of `thresh_LMPdist`: the C++ wrapper currently casts it to integer, which prevents faithful support of fractional settings, although the captured value is 2.

### 6. Boundary neighbors differ near search-box corners

**Status: confirmed structural difference; output contribution needs isolation. Fixable: yes.**

MATLAB `getdB` constructs separate face, edge, and corner neighbor lists. The C++ boundary test compares candidates with any neighboring pixel that also belongs to the boundary. Near a corner, this can include a diagonally adjacent pixel on another face.

Opposing faces being far apart does not exclude neighbors on adjacent faces. The supplied MATLAB helper also contains unusual offset expressions; parity requires following the actual expressions rather than replacing them with presumed intended geometry.

**Fix:** reproduce the helper's candidate and neighbor lists, then compare them on small boxes, corners, and image-edge cases. A blanket interior-neighbor test is not the correct replacement.

### 7. Padding and search-box clipping

**Status: source difference confirmed; independent effect needs validation. Fixable: yes.**

MATLAB adds three zero layers on each side and clips boxes to that padded volume. C++ skips pixels outside the original image. The original analysis calls this “bounds-clamped access,” which does not accurately describe the C++ candidate loops.

**Fix:** reproduce padded-volume semantics or prove an equivalent 2D implementation. A full 3D allocation is not inherently necessary, but clipping, candidate lists, zero neighbors, and coordinate shifts must agree. Test radii that approach and exceed the padding width.

### 8. Direction arithmetic, thresholds, and normalization

**Status: confirmed source difference; individual runtime effect unmeasured. Fixable: yes, with numerical validation.**

MATLAB creates the padded DSM with `zeros(...)`, making its storage double, and assigns single-precision values into it. Coordinates and direction arithmetic are also double. C++ ties DSM storage and direction calculations to float32 and casts several parameters to float.

MATLAB adds `eps` in initial and candidate direction normalization; C++ divides by the norm without that term. C++ processing also uses float32 direction and length calculations.

**Fix:** separate image-value precision from calculation precision and match normalization expressions, threshold comparisons, and operation order. Increasing precision can reduce differences but is not a guarantee of identical paths by itself.

### 9. Radius quantization

**Status: an amplifier of other differences; no minimum-radius bug confirmed. Fixable: by matching inputs and rules.**

Both MATLAB and C++ use `max(2,ceil(...))` for the initial search and plain `ceil(...)` for continuation. The earlier suspicion that continuation needs a minimum of two was incorrect.

**Fix:** preserve these two distinct rules. Compare radius sequences after resolving DSM and candidate-selection differences. Do not add a continuation minimum as a parity correction.

### 10. First-crosslink selection order

**Status: confirmed in source. Fixable: yes.**

MATLAB uses `getbox` and takes the first qualifying crosslink in column-major box order. C++ scans the box row-first and stops at the first encountered crosslink.

**Fix:** match MATLAB's order whenever a box contains multiple crosslinks. Use small fixtures with deliberately competing crosslinks to validate termination choices.

### 11. Vertex identity and duplicate-fiber timing

**Status: fixed in the vertex-identity follow-up; exact captured extension graph verified.**

MATLAB allocates a new vertex for each extension step and removes duplicate fibers using endpoint vertex IDs. Coincident coordinates are handled later by `remove_repeat`. Before the follow-up fix, C++ merged coordinates through a pixel index map and removed duplicates using endpoint pixel pairs.

Separate MATLAB vertex IDs can share coordinates. Merging them early can change which fibers survive and what connectivity later processing sees.

**Implemented fix:** preserve vertex identity until the corresponding MATLAB merge stage, and reproduce duplicate removal in the same order. Symmetric keys alone were insufficient to establish equivalence. See the trace and ordered graph comparisons above.

### 12. Continuity-line sampling

**Status: intended behavior ported; exact equivalence needs validation. Fixable: yes.**

MATLAB `ind_btw_nodes` uses double arithmetic, colon-generated sample locations, and `round` on padded one-based coordinates. C++ uses float arithmetic, `floor(length)` samples, and `floor(value+0.5)` on unpadded coordinates, skipping out-of-range samples.

**Fix:** compare sampled integer pixel sequences for diagonal lines, near-half-pixel values, endpoints, and edges. The continuity check's presence does not prove exact sampling equality.

### 13. Endpoint-direction index in `fiberproc`

**Status: confirmed indexing mismatch. Fixable: yes. Priority: high.**

For a final endpoint, MATLAB's local `getvect` selects one-based position `max(1,len-sp)`. Its zero-based equivalent is `max(0,len-sp-1)`. [fiberproc_native.cpp](../src/ctfire_py/CPP/fiberproc_native.cpp) uses `max(0,len-sp)`.

**Fix:** correct the index and reproduce the normalization expression. This changes direction estimates used by both intersection linking and gap linking, so validate both stages.

### 14. Gap-link candidate membership and order

**Status: confirmed source difference. Fixable: yes.**

MATLAB `make_vertexmatrix` stores one vertex ID per rounded pixel; `findclose` traverses that matrix spatially and filters by distance. C++ scans all vertices by ID without the same per-pixel overwrite behavior.

**Fix:** reproduce endpoint-matrix membership, overwrite behavior, rounding, distance filtering, and candidate order. Coincident endpoints and equal scores are particularly useful cases.

### 15. Intersection-link ties and connectivity-list semantics

**Status: merge endpoint/membership ordering fixed and validated on this capture; multiway tie behavior still needs broader validation.**

MATLAB `min2` finds the first minimum column-first. C++ scans the pairwise matrix row-first. MATLAB merge helpers sort endpoint lists; C++ previously removed entries and appended surviving fiber IDs without sorting. The merge-order follow-up above fixes that mismatch and sorted membership updates. MATLAB `setdiff` also deduplicates entries within each vertex's contribution to `F.f`, while retaining repeated contributions across vertices.

**Fix:** match tie order, sorting, and the precise locations where duplicates are removed or retained. Connectivity multiplicity can affect short-fiber removal. Audit MATLAB's reused `A` matrix against C++'s newly allocated matrix before declaring multiway linking faithful.

### 16. Coordinate hashing and trim behavior

**Status: needs focused validation. Fixable: yes.**

MATLAB `remove_repeat` hashes coordinates using `ceil(X/max(X(:))*1000)`. This is not generally invariant to exchanging axes or shifting all coordinates from one-based to zero-based. C++'s early pixel merging and later hashing therefore need comparison under the actual coordinate conventions.

C++ processing retains the vertex array through intermediate trimming, whereas MATLAB compacts coordinates and renumbers vertices. Unused vertices may affect a hash normalized by the maximum coordinate. Returned C++ vertex counts should not be treated as active-network counts without checking which vertices fibers reference.

**Fix:** compare hashing, coincident-vertex replacement, compaction, and renumbering after each stage. Preserve MATLAB's timing and coordinate interpretation.

### 17. Dangler behavior and ignored configuration

**Status: confirmed. Fixable: yes; not a demonstrated source of current path drift.**

MATLAB `fire_2D_ang1` calls `check_danglers`, then overwrites all its outputs with the original extension outputs. Thus downstream extraction does not depend on its returned result, regardless of its internal logic bug.

Python forces `faithful_danglers=True`, overriding the user's parameter, and its preceding log can announce corrected behavior that does not run.

**Fix:** explicitly document and implement compatibility behavior. If corrected dangler removal remains an option, expose it deliberately and report the mode accurately. Do not enable it while claiming faithful MATLAB output.

### 18. Curvelet reconstruction threshold indexing

**Status: confirmed in source; not measured by this capture comparison. Fixable: yes.**

MATLAB `CTrec_1` chooses `cutoff=cfs(nb)`, with `nb=round(pct*length(cfs))`. [ct_reconstruction.py](../src/ctfire_py/ct_reconstruction.py) uses `all_coeffs[nb]`; the equivalent zero-based index is `nb-1` for MATLAB-valid values.

**Fix:** correct the index and define behavior for parameter extremes separately from MATLAB's valid domain. Compare reconstructed images before extraction. Also verify transform parameters: Python explicitly computes scale count using `floor`, whereas the MATLAB source describes its wrapping default using `ceil`; this is a candidate discrepancy for non-power-of-two image sizes, not a measured result here.

### 19. Reference environment and function shadowing

**Status: confirmed reproducibility concern. Fixable: yes.**

The source bundle contains CT-FIRE and CurveAlign copies of FIRE functions, plus two `calc_fiberlen.m` files with different output guards. MATLAB path order determines which implementation executes. A file edited in one folder may not be the active implementation.

**Fix:** retain the captured environment metadata, record resolved paths for core functions, and pin the intended implementation. Rebuild native code before collecting Python baselines. Store effective FIRE parameters alongside the original CT-FIRE settings.

### 20. Previously fixed indexing and allocation bugs

**Status: fixes present in source. Fixable: already addressed, retain regression coverage.**

The row-major `findlocmax` indexing, extension width/height constructor order, and invalid vector-reference lifetime in `remove_repeat` have source fixes. These are ordinary correctness bugs rather than unavoidable parity limits.

Bounds tests are useful but do not prove rectangular MATLAB parity. Square dimensions do not make a transposed non-symmetric image identical; claims that fixes preserve every metric require measurements, even when two transpositions happen to cancel in part of the pipeline.

## Weaknesses in the original parity analysis

| Original claim or approach | Correction supported by this review |
| --- | --- |
| Exact fibers are “not feasible” for principled reasons. | The identified problems are engineering mismatches. Exact matching is unproven, but no fundamental impossibility was established. |
| Preprocessing is identical. | Thresholding matches this capture, but DSM smoothing differs in dimensions and arithmetic. The capture does not separately save the unsmoothed distance transform, so this review does not establish exact `bw_dist` equality. |
| Extension starts from identical DSM values. | The Python buffer is perturbed by nucleation detection, and preprocessing differs. Identical-input experiments must explicitly use the saved MATLAB DSM. |
| `fiberproc` is correct and only inherits extension differences. | On MATLAB's own extension output, C++ produces 192 fibers versus 191, with 175 exact shared paths and 0.64% length drift. An endpoint indexing mismatch is confirmed. |
| Boundary-only neighbors are equivalent for all practical radii. | Adjacent faces meet near corners, and MATLAB has distinct edge neighbor rules. Continuation can also use radii below two. |
| Count regression disproves matching MATLAB traversal order. | MATLAB sorting is explicit. Ordering must be evaluated together with the pairwise pruning and tie rules, using path comparisons. |
| Symmetric pixel-pair deduplication is MATLAB-faithful. | MATLAB uses vertex IDs and delays merging coincident coordinates. Identity and timing matter. |
| Three changes would be sufficient for exact matching. | Ordering, pruning, crosslink selection, vertex construction, processing indices, gap search, and reconstruction indexing also require attention. |
| Direction drift explains the remaining gap. | It is plausible, but the review finds discrete mismatches; no direction-only experiment quantified its contribution. |
| Continuation uses `max(2,ceil(...))`. | That minimum applies only to the initial search. MATLAB continuation uses plain `ceil(...)`. |
| The length shortfall is caused by short-fiber handling. | That attribution is not isolated. Large extension-length differences already exist before `fiberproc`. |
| Length drift is at most 5%. | The same document later mentions about 9%. The new run measures 5.17% for the current extraction and 9.56% for isolated extension. Report stage and configuration explicitly. |
| Historical counts describe the current `real1` baseline. | The old table lists 481 MATLAB fibers; this reconstructed-input capture has 191. Do not compare these configurations without establishing matching inputs, parameters, and filtering stages. |
| Two MATLAB references are stale because Python RNG was corrected. | Changing Python RNG does not invalidate a MATLAB reference. Regenerate references when MATLAB code, parameters, input, or provenance require it. |
| Passing tests demonstrates high parity. | Several tests allow approximately 15% count drift, 10% length drift, or histogram correlation above 0.5. Some C++ comparisons skip complex fiber structures. These are approximate checks. |
| EMD 0.16–0.17 establishes high orientation fidelity. | Its units, angular wrap treatment, normalization, and calculation are not documented sufficiently to interpret it as a parity percentage. These values were not recomputed in this review. |
| The remaining drift is smaller than biological variability. | No supporting variability measurement is provided. Biological variation does not establish software correctness on identical inputs. |

The documented test baseline of 94 passed and 2 xfailed was not reproduced. An earlier targeted run of `tests/test_fire_2d_angle.py` and `tests/test_cpp_functions.py` produced **26 passed, 22 skipped** with plugin autoload disabled and a headless plotting backend. That run used the older installed native binary; it is test-coverage context, not validation of the freshly built runtime comparisons. No claim is made about the full current suite.

[TOLERANCE_RATIONALE.md](../tests/test_results/cpp_test_files/TOLERANCE_RATIONALE.md) also needs revision: it still describes a custom RNG, suggests sorting vertices within fibers, and mixes approximate count allowances with exact-connectivity expectations. Sorting a fiber's internal vertices destroys its path order; canonicalize whole-path reversal and graph IDs instead. Its claim that matching seeds alone implies matching fiber counts omits the other algorithm differences. The stated `assert_allclose` bound is additive (`atol + rtol * abs(reference)`), not two separate conditions that must both hold.

## Recommended implementation and validation order

1. Preserve a verified baseline: input, effective parameters, MATLAB paths/version, source revision, native build, and reference checksum.
2. Fix DSM mutation and reproduce singleton-depth smoothing. Compare threshold masks, DSM errors, radius changes, and ordered nucleation points separately.
3. Match LMP ordering, pairwise pruning, face/edge neighbors, padding, crosslink selection, and continuity samples on small diagnostic fixtures and the saved real-image inputs.
4. Match extension arithmetic and vertex identity. Compare complete path multisets, endpoint connections, and radii before processing.
5. Correct endpoint direction indexing and reproduce processing candidate order, tie rules, coordinate hashing, and trim behavior. Feed identical graphs into both implementations and compare after each processing stage.
6. Fix and validate curvelet reconstruction separately, then repeat end-to-end extraction using matching reconstructed inputs.
7. Expand to synthetic, rectangular, edge-touching, tied-candidate, fractional-parameter, and multi-thread cases. Tighten tolerances after the corresponding discrepancies are resolved.

For this fixture, the runtime capture is sufficient to start fixes. Additional MATLAB captures after individual `fiberproc` operations or for selected extension steps would make remaining first-divergence debugging easier; they are useful follow-ups rather than a prerequisite for the confirmed fixes.

## Limits of this review

This review measures one captured image/configuration. It does not promise that each fix improves aggregate count or length monotonically, that all listed issues occur in this run, or that matching this fixture establishes universal parity. Curvelet reconstruction, interpolated fibers, beam processing, angles, density, and other derived outputs have not all been validated numerically here.

The practical target should be exact geometry and connectivity where feasible, accompanied by explicitly measured numerical tolerances for derived quantities. Track count, total length, path equality, topology, radii, and angular agreement as separate metrics.
