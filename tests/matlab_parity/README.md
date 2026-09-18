# MATLAB parity harness (git / clone only)

**Not in the wheel or the sdist.** Installed releases contain only what is
needed to run or rebuild the package. This directory is a maintainer
check: clone the repo and set `TMEQ_RUN_MATLAB_PARITY=1`.

Validates that `pycurvelets.SHG_HE_registration` reproduces MATLAB
`BDcreation_reg2.m` (tests 1-7, `pipeline="reg2"`) and `BDcreation_reg.m`
(tests 8-9, `pipeline="reg1"`). Runtime registration does not need MATLAB
or these dumps.

```bash
TMEQ_RUN_MATLAB_PARITY=1 pytest -q -p no:napari \
    tests/test_shg_he_registration_matlab_parity.py \
    tests/test_shg_he_registration_gt.py
```

## Tracked files (needed to re-run or regenerate)

| File | Purpose |
| --- | --- |
| `dump_bdc_reg2.m` | Offline MATLAB dump of reg2 cases (tests 1-7). |
| `dump_bdc_reg1.m` | Same for reg1 (tests 8-9). Seed 28 matches the goldens (`dumps/test8_km3`, `dumps/test9_km3`). |
| `dump_srgb2lab_components.m` | Rebuilds `src/pycurvelets/data/matlab_srgb2lab_components.npz` if the ICC tables change. |
| `probe_interp2d.m` | Regenerates `dumps/interp2d_probe.mat` (`imwarp` edge rule). |
| `analyze_dumps.py` | Optional offline comparison; writes local `analysis_summary.json` (gitignored) and can refresh `gt_affine_*.json`. |

## Tracked dumps (pytest inputs)

Per case (`test1`–`test7`, `test8_km3`, `test9_km3`): `images.mat` (exact
`HEmoving` / `fixedSHG` doubles) and `tform_*.txt`. Also
`interp2d_probe.mat` and `gt_affine_test{4–7}.json`.

Regenerable sidecars (`*.tif` previews, `meta.mat`, traces, `test2_rerun/`,
k-means-optima folders, `intermediates.mat`) are gitignored.

`BDcreation_reg.m` never seeds `kmeans`. The Python default `kmeans_seed=28`
is the optimum that produced the committed goldens.

## Tumor annotation (`BDcreationHE2` / `BDcreationHE`)

Validates that `pycurvelets.tumor_annotation_from_he` reproduces MATLAB
`BDcreationHE2.m` (default HSV path, tests 1-3) and `BDcreationHE.m`
(RGB k-means at `rng(28,'twister')`). Inputs are the already-registered
HE TIFFs under `tests/test_for_shg_he_registration_BDcreation/HE/HE_registered_test{1,2,3}/`.

The committed `SHG/CA_Boundary/BDcreationHE_testNresults_*.tif` goldens are
a fresh HE2 run (the previous copies were `BDcreationHE.m` output; the
filename still says HE). Compare HE-path masks to `dumps/he_testN/images.mat`.

```bash
TMEQ_RUN_MATLAB_PARITY=1 pytest -q -p no:napari \
    tests/test_he_bdc_annotation.py \
    tests/test_tumor_annotation_from_he.py \
    tests/test_tumor_annotation_matlab_parity.py
```

### Regenerating annotation dumps

From this directory, with MATLAB on `PATH` (or the app bundle `matlab`):

```bash
matlab -batch "dump_annotation_primitives; dump_bdc_he2; dump_bdc_he"
```

| File | Purpose |
| --- | --- |
| `dump_annotation_primitives.m` | `histeq` / `fspecial('disk')` / `strel` / `padarray` / `im2bw` probes. |
| `dump_bdc_he2.m` | Instrumented `BDcreationHE2` on registered HE tests 1-3. |
| `dump_bdc_he.m` | Instrumented `BDcreationHE`; pins `rng(28,'twister')` before `kmeans`. |
| `analyze_he_dumps.py` | Offline first-divergent-step report (`--preproc` for intermediates). |

Per case `dumps/he2_<case>/` and `dumps/he_<case>/`: track `images.mat`;
`intermediates.mat` and `meta.mat` are gitignored. Annotation primitives
are `dumps/annotation_primitives.mat`.
