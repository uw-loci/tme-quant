# MATLAB parity harness for `BDcreation_reg2` (SHG <-> H&E registration)

Developer-side validation that the Python port (`pycurvelets.SHG_HE_registration`
with the default `registration_method="matlab"`) reproduces MATLAB's
`BDcreation_reg2.m` bit-for-bit. **Nothing in the runtime pipeline needs
MATLAB**; MATLAB was only used once, offline, to produce the dumps in this
folder. The dumps are committed so the parity tests can be re-run by anyone.

Run the parity suites locally (they are skipped by default / in CI):

```bash
TMEQ_RUN_MATLAB_PARITY=1 pytest -q -p no:napari \
    tests/test_shg_he_registration_matlab_parity.py \
    tests/test_shg_he_registration_gt.py
```

## Files

### MATLAB scripts (run once, offline, R2025b)

| File | Purpose |
| --- | --- |
| `dump_bdc_reg2.m` | Re-implements the preprocessing + `imregtform` calls of `BDcreation_reg2.m` and dumps, per case, everything Python needs to compare against: exact `double` inputs to the registration, both stage transforms, optimizer settings, and optionally the `DisplayOptimization` trace and every preprocessing intermediate. Also re-runs `test2` to prove MATLAB is deterministic (`test2_rerun`). |
| `probe_es_steps.m` | Runs `imregtform` for 1..k iterations from identity and prints the resulting `tform.A`. The parameter deltas expose `radius * scales * N(0,1)`, which is how the RNG seed (12345), transform centre and per-pyramid-level radius/epsilon refiner of MATLAB's (1+1)-ES were pinned down without access to the mex source. |
| `probe_interp2d.m` | Samples `images.internal.interp2d` (what `imwarp` uses) on a dense grid straddling the image border to capture the exact "inside" rule and fill behaviour; result in `dumps/interp2d_probe.mat`. |

### Python analysis

| File | Purpose |
| --- | --- |
| `analyze_dumps.py` | Offline comparison tool. For each case: Dice of Python's collagen mask vs MATLAB's `HEmoving`; the ITK-v3 engine run on MATLAB's exact inputs compared to `tform_*.txt` (parameter space and, where a trace exists, iteration-by-iteration metric values, reporting the first divergent iteration); same-basin check on Python's own mask; ground-truth affine recovery for patient_02. `--preproc` compares every preprocessing intermediate step by step. Writes `dumps/analysis_summary.json`, `dumps/preprocessing_parity.json`, `dumps/gt_affine_*.json`. |

### `dumps/<case>/` (one folder per case: test1-3 = patient_001 at ppm 1.5/2.0/3.0, test4-7 = patient_02 rois)

| File | Purpose |
| --- | --- |
| `images.mat` | `HEmoving` and `fixedSHG` as exact MATLAB doubles - the two images handed to `imregtform`. Input to the engine-parity test. |
| `HEmoving.tif`, `fixedSHG.tif` | 8-bit previews of the same (lossy; kept for eyeballing only). |
| `tform_similarity.txt`, `tform_affine.txt` | `tformSimilarity.T` and `tform.T` (3x3, MATLAB 1-based, row-vector convention) - stage 1 and stage 2 results. Golden for the engine-parity test. |
| `meta.mat` | Optimizer settings (`InitialRadius`, `Epsilon`, `GrowthFactor`, `MaximumIterations`, `PyramidLevels`), `pixpermic`, image shapes. |
| `optimization_trace.txt` (test1 only) | `DisplayOptimization` output: per-iteration Mattes MI for both stages. The Python engine matches it iteration for iteration. |
| `es_probe.txt` (test1 only) | Output of `probe_es_steps.m`. |
| `intermediates.mat` (gitignored, ~8 MB) | Every preprocessing intermediate (`imresize` RGB, `imadjust` output, HSV, `graythresh` values, nuclei / collagen masks, filtered gray). Regenerate with `dump_bdc_reg2` if you need `analyze_dumps.py --preproc`. |

### `dumps/` top level

| File | Purpose |
| --- | --- |
| `gt_affine_test{4,5,6,7}.json` | Ground truth for the synthetic patient_02 cases. `forward_2x3_input_grid_0based` is the affine (SIFT + RANSAC, inlier RMS ~0.6 px) between the input HE and `patient_02_HE_original-<roi>.tif` (the truly aligned HE). Also stores its working-grid version, MATLAB's error vs GT (angle, scale, corner displacement) and the Python engine's error (identical), plus Mattes MI at identity / MATLAB / GT transforms. Same ROI => same GT, so test8/9 (roi4) reuse test6's. |
| `interp2d_probe.mat` | Golden for the `imwarp` edge-rule test. |
| `analysis_summary.json` | Last `analyze_dumps.py` run: per-case Dice, parameter deltas, first divergent iteration, determinism check. |
| `preprocessing_parity.json` | Last `analyze_dumps.py --preproc` run: max abs diff per preprocessing stage (all 0 or ~1e-16). |
| `*.log` (gitignored) | MATLAB batch run logs. |

## What "parity" means here

Layered so a regression points at one stage:

1. Primitive ports (`graythresh`, even-kernel `imfilter` centre, `imwarp`
   inside rule, full-precision `rgb2gray`) vs values captured from MATLAB.
2. Preprocessing: Python `fixedSHG` / `HEmoving` == `images.mat` doubles.
3. Engine: ITK-v3 (1+1)-ES on `images.mat` reproduces `tform_affine.txt`
   (|dA| < 1e-9) and the per-iteration trace.
4. End to end: `shg_he_registration(...)` reproduces the golden TIFF
   pixel-for-pixel (MAE 0, 100 % exact) on all seven reg2 cases.
