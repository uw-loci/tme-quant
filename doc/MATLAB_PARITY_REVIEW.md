# CT-FIRE MATLAB parity review

Author: Andrew Kim
Date: October 12, 2026

I worked with Codex to review the differences in our output and that of the original implementation of CT-FIRE. I analyzed both the MATLAB source code as well as the several variables from a runtime test on `real1.tif`, and compared them to the old parity analysis file to see what we had missed before. This document is a list of findings and recommended approaches to get our implementation as close to the original as possible.

## Conclusion

Substantially closer parity is achievable. The remaining differences include reproducible algorithm and indexing mismatches, rather than only floating-point drift. None of the identified issues establishes a fundamental barrier to reproducing MATLAB's fiber geometry and connectivity on a fixed reference environment.

Exact matching remains an objective, not a demonstrated outcome. Individual fixes can interact, and their effect must be measured using matching intermediate inputs. Universal byte-for-byte agreement across platforms is a harder target than reproducing fiber paths, connectivity, and measurements for a defined MATLAB version.

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

**Status: source difference confirmed; residual numerical cause needs validation. Fixable: likely, with numerical comparison.**

MATLAB filters a single-precision distance image through successive `imfilter` calls. Python's `smooth` promotes the input to double and casts back after convolution. Kernel construction and convolution arithmetic can also differ.

**Fix:** compare individual filtering passes, kernel values, accumulation order, and output casts. Preserve exact values around integer radius boundaries and decision thresholds. A float32 diagnostic did not produce exact equality, so simply changing the array dtype is insufficient evidence of a solution.

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

**Status: confirmed structural difference. Fixable: yes, potentially requiring graph-construction changes.**

MATLAB allocates a new vertex for each extension step and removes duplicate fibers using endpoint vertex IDs. Coincident coordinates are handled later by `remove_repeat`. C++ merges coordinates through a pixel index map and removes duplicates using endpoint pixel pairs.

Separate MATLAB vertex IDs can share coordinates. Merging them early can change which fibers survive and what connectivity later processing sees.

**Fix:** preserve vertex identity until the corresponding MATLAB merge stage, and reproduce duplicate removal in the same order. Symmetric keys alone are insufficient to establish equivalence.

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

**Status: confirmed ordering differences; full effect needs validation. Fixable: yes.**

MATLAB `min2` finds the first minimum column-first. C++ scans the pairwise matrix row-first. MATLAB merge helpers sort endpoint lists; C++ removes entries and appends surviving fiber IDs without the same explicit sorting. MATLAB `setdiff` also deduplicates entries within each vertex's contribution to `F.f`, while retaining repeated contributions across vertices.

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
