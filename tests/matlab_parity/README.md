# MATLAB parity harness (git-dev only)

**Not in the wheel or the sdist.** `MANIFEST.in` prunes this directory from
PyPI source releases; installed wheels contain only `src/`. Clone the git
repo and set `TMEQ_RUN_MATLAB_PARITY=1` to run these checks.

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
