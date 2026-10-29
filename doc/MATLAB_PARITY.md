# MATLAB parity harness (git / clone only)

The scripts and dumps live in `archive/matlab_parity/`. They are a maintainer
check, not part of the pytest tree and not in the wheel or the sdist.
Clone the repo and set `TMEQ_RUN_MATLAB_PARITY=1`. File names below are
relative to `archive/matlab_parity/`.

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
| `dump_bdc_reg2.m` | Offline MATLAB dump of reg2 cases (tests 1-7). Optional `callOriginal` also runs CurveAlign `BDcreation_reg2.m`. |
| `dump_bdc_reg1.m` | Same for reg1 (tests 8-9). Default is `rng(28,'twister')` immediately before `kmeans`, matching the goldens (`dumps/test8_km3`, `dumps/test9_km3`) and Python `kmeans_seed=28`. Optional `callOriginal` also runs CurveAlign `BDcreation_reg.m` (that function still does not seed `kmeans`). |
| `call_original_bdc.m` | Used when `callOriginal` is true. Copies inputs to a temp HE/SHG pair and writes `registered_original.tif`. |
| `resolve_curvealign_dir.m` | Resolves the CurveAlign directory argument or `CURVEALIGN_MATLAB`. |
| `dump_srgb2lab_components.m` | Rebuilds `src/pycurvelets/data/matlab_srgb2lab_components.npz` if the ICC tables change. |
| `probe_interp2d.m` | Regenerates `dumps/interp2d_probe.mat` (`imwarp` edge rule). |
| `analyze_dumps.py` | Optional offline comparison; writes local `analysis_summary.json` (gitignored) and can refresh `gt_affine_*.json`. |

## Tracked dumps (pytest inputs)

Per case (`test1`–`test7`, `test8_km3`, `test9_km3`): `images.mat` (exact
`HEmoving` / `fixedSHG` doubles) and `tform_*.txt`. Also
`interp2d_probe.mat` and `gt_affine_test{4–7}.json`.

Regenerable sidecars (`*.tif` previews, `meta.mat`, traces, `test2_rerun/`,
k-means-optima folders, `intermediates.mat`) are gitignored.

`BDcreation_reg.m` in CurveAlign never seeds `kmeans` (the call is line 69).
This repo does not change that file. The reproducible MATLAB reference is
`dump_bdc_reg1`, which calls `rng(28,'twister')` immediately before its own
`kmeans`. A raw call of unmodified `BDcreation_reg.m` is not the same
reference as the committed goldens. The Python default `kmeans_seed=28`
matches that seeded dump.

## Calling the original CurveAlign functions

`callOriginal` is off by default. The instrumented dump still runs
(intermediates, transforms, and for reg1 the seeded `registered.tif`).
When `callOriginal` is true, the original function is an additional step.
It writes `dumps/<case>/registered_original.tif` from a temporary copy of
the inputs, so the fixture tree is not modified. Pass the directory that
contains the `.m` file, or set `CURVEALIGN_MATLAB`.

```matlab
dump_bdc_reg2('', false, true, '/path/to/CurveAlign_CT-FIRE')
dump_bdc_reg1('', false, 28, '', true, '/path/to/CurveAlign_CT-FIRE')
```

For reg2, `registered_original.tif` is the deterministic MATLAB reference
(`BDcreation_reg2` has no `kmeans`). For reg1, that file follows unseeded
`kmeans` and is not the seeded golden; `registered.tif` from the same dump
is the reproducible reference. The script prints MAE of
`registered_original.tif` against the committed golden when that golden is
in the fixture tree.

Fixtures are read from `tests/test_for_shg_he_registration_BDcreation/`.
