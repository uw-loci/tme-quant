# BDcreation registration comparison figures (tests 1-9)

Registration only (`BDcreation_reg` / `BDcreation_reg2`). The scripts call
`pycurvelets.shg_he_registration` and write PNGs under
`tests/artifacts/bdc_regression_viz/current/`. That directory is gitignored.
File names are listed in [ARTIFACTS.md](ARTIFACTS.md).

The figures record the Python port of `BDcreation_reg2.m` and
`BDcreation_reg.m`. They are produced with the ITK v3 (1+1)-ES port of
MATLAB `imregtform`. Tests 1-7 use `pipeline="reg2"` (HSV collagen mask).
Tests 8-9 use `pipeline="reg1"`.

## Choosing reg2 or reg1

`pipeline` selects which MATLAB script the port runs. Use `reg2` unless the
comparison target is historical `BDcreation_reg` output. The two pipelines
build different moving images at different resolutions, so one is not a
correction for a poor result of the other.

**reg2 (default).** `BDcreation_reg2.m`, the current CurveAlign script, and
the golden for tests 1-7.

- SHG is capped at 2 px/µm. The warped H&E is then resized back to the original SHG grid.
- The moving image is a binary HSV collagen mask (`ecm_method="hsv"`). `ecm_method="rgb"` uses the fixed RGB cuts from `BDcreation_reg` on that same grid and does not run k-means.
- Deterministic. No k-means.
- test5 (patient_02 roi4, ppm 1.5) stays about 69 px from synthetic ground truth in both MATLAB and Python. That is MATLAB's local optimum on this case. Parity with `BDcreation_reg2` is the goal there, so reg2 stays the pipeline for that case.

**reg1.** `BDcreation_reg.m`. Use it to reproduce that older script (tests 8-9).

- SHG stays at native resolution. There is no 2 px/µm cap. The output is warped onto the SHG grid.
- The moving image is decorrstretch, fixed RGB cuts, then CIELAB a*b* k-means (k=3, 3 replicates).
- MATLAB leaves `kmeans` unseeded, so a raw rerun of `BDcreation_reg.m` is not deterministic. Python `kmeans_seed` (default 28) replays `rng(28,'twister')` and matches the tests 8-9 goldens. Those goldens sit about 8 px from the synthetic ground-truth H&E, which is MATLAB's accuracy on those cases.

Regenerate (tests 4-9 use the patient_02 fixtures in the clone):

```bash
python archive/visualization/generate_registration_comparison.py            # tests 1-3
python archive/visualization/generate_registration_comparison_patient02.py  # tests 4-9
```

The generators are development scripts in `archive/visualization/`, not
pytest tests. Output is gitignored. Pass an output directory as the first
argument to write somewhere else. Patient_02 also accepts `--cases id1,id2`.

## Files

| File | What it is |
| --- | --- |
| `archive/visualization/generate_registration_comparison.py` | patient_001 (tests 1-3, ppm 1.5 / 2.0 / 3.0) vs the `BDcreation_reg2` goldens. Calls `shg_he_registration` with the default `pipeline="reg2"` and `ecm_method="hsv"`. CLI: `[out_dir]`. |
| `archive/visualization/generate_registration_comparison_patient02.py` | patient_02 (tests 4-9). Tests 4-7 pass `pipeline="reg2"`; tests 8-9 pass `pipeline="reg1"`. Adds ground-truth scoring (see below). CLI: `[out_dir] [--cases id,...]`. |
| `comparison_test{1,2,3}_ppm*_matlab_hsv.png` | patient_001 figures. 3x4 panel: inputs (HE, SHG), MATLAB golden, Python output; split / checkerboard / 50-50 blend of MATLAB vs Python; Python-HE vs SHG checkerboard; \|diff\| x3, per-pixel mean \|diff\| heat-map, signed-diff histogram, and a stats box (MAE, RMSE, PSNR, SSIM, exact %, within-5/10/20, SHG MI vs identity, SHG NCC). |
| `comparison_test{4..7}_reg2_*_matlab_hsv.png` | patient_02 vs reg2 goldens, same layout, plus a **ground truth vs Python** checkerboard and GT metrics in the stats box. |
| `comparison_test{8,9}_reg1_*_matlab_hsv.png` | patient_02 vs `BDcreation_reg` (reg1) goldens. Python `pipeline="reg1"` (decorrstretch + LAB k-means, seed 28) reproduces them pixel-for-pixel. |

## Results with the current build

MAE / SSIM are Python vs the MATLAB golden (uint8). `GT disp` is the mean
corner displacement (px, working grid) of the recovered transform vs the
ground-truth affine for the synthetic patient_02 cases (`identity` = error of
doing nothing); lower is better.

| Case | Input | ppm | Golden | MAE | SSIM | Exact | GT disp (identity) | MATLAB GT disp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test1 | patient_001 | 1.5 | reg2 | 0.00 | 1.0000 | 100 % | n/a (real data) | - |
| test2 | patient_001 | 2.0 | reg2 | 0.00 | 1.0000 | 100 % | n/a | - |
| test3 | patient_001 | 3.0 | reg2 | 0.00 | 1.0000 | 100 % | n/a | - |
| test4 | patient_02 roi2 | 2.6 | reg2 | 0.00 | 1.0000 | 100 % | 5.5 px (37.2) | 5.5 px |
| test5 | patient_02 roi4 | 1.5 | reg2 | 0.00 | 1.0000 | 100 % | 68.7 px (82.6) | 68.7 px |
| test6 | patient_02 roi4 | 2.6 | reg2 | 0.00 | 1.0000 | 100 % | 9.4 px (63.6) | 9.4 px |
| test7 | patient_02 roi5 | 2.6 | reg2 | 0.00 | 1.0000 | 100 % | 7.5 px (48.7) | 7.5 px |
| test8 | patient_02 roi4 | 3.0 | reg1 | 0.00 | 1.0000 | 100 % | 8.1 px (82.6) | 8.1 px |
| test9 | patient_02 roi4 | 2.6 | reg1 | 0.00 | 1.0000 | 100 % | 7.9 px (82.6) | 7.9 px |

Reading the table:

* Tests 1-7: the Python default (`pipeline="reg2"`) reproduces MATLAB
  `BDcreation_reg2` exactly, including MATLAB's own failure on test5 (roi4 at
  ppm 1.5 stays ~69 px from truth in both implementations). Parity is the
  goal here, not accuracy.
* Tests 8-9: `pipeline="reg1"` reproduces MATLAB `BDcreation_reg` exactly
  (MAE 0). MATLAB's own `kmeans` is unseeded; the Python default seed (28)
  lands in the same rare optimum the committed goldens used. Those goldens
  sit ~8 px from the synthetic ground-truth HE (SIFT residual), which is
  MATLAB's accuracy, not a porting error.
