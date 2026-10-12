# BDcreation registration comparison figures (tests 1-9)

Registration only (`BDcreation_reg` / `BDcreation_reg2`). The PNGs live in
`tests/artifacts/bdc_regression_viz/current/` and are indexed in
[ARTIFACTS.md](ARTIFACTS.md).

The figures are a visual record kept in git, not in the wheel or sdist
(see `MANIFEST.in`). They record the Python port of `BDcreation_reg2.m`
(`pycurvelets.SHG_HE_registration`). `current/` holds one figure per test case,
produced with the ITK v3 (1+1)-ES port of MATLAB `imregtform` and the
BDcreation_reg2 HSV collagen mask.

Regenerate (tests 4-9 use the patient_02 fixtures in the clone):

```bash
python archive/visualization/generate_registration_comparison.py            # tests 1-3
python archive/visualization/generate_registration_comparison_patient02.py  # tests 4-9
```

Only `tests/artifacts/bdc_regression_viz/current/` is tracked. Other output
directories are gitignored. The generators are development scripts in
`archive/visualization/`, not pytest tests.

## Files

| File | What it is |
| --- | --- |
| `archive/visualization/generate_registration_comparison.py` | patient_001 (tests 1-3, ppm 1.5 / 2.0 / 3.0) vs the `BDcreation_reg2` goldens. Calls `shg_he_registration` with the default `pipeline="reg2"` and `ecm_method="hsv"`. CLI: `[out_dir]`. |
| `archive/visualization/generate_registration_comparison_patient02.py` | patient_02 (tests 4-9). Tests 4-7 pass `pipeline="reg2"`; tests 8-9 pass `pipeline="reg1"`. Adds ground-truth scoring (see below). CLI: `[out_dir] [--cases id,...]`. |
| `tests/artifacts/bdc_regression_viz/current/comparison_test{1,2,3}_ppm*_matlab_hsv.png` | patient_001 figures. 3x4 panel: inputs (HE, SHG), MATLAB golden, Python output; split / checkerboard / 50-50 blend of MATLAB vs Python; Python-HE vs SHG checkerboard; |diff| x3, per-pixel mean |diff| heat-map, signed-diff histogram, and a stats box (MAE, RMSE, PSNR, SSIM, exact %, within-5/10/20, SHG MI vs identity, SHG NCC). |
| `tests/artifacts/bdc_regression_viz/current/comparison_test{4..7}_reg2_*_matlab_hsv.png` | patient_02 vs reg2 goldens, same layout, plus a **ground truth vs Python** checkerboard and GT metrics in the stats box. |
| `tests/artifacts/bdc_regression_viz/current/comparison_test{8,9}_reg1_*_matlab_hsv.png` | patient_02 vs `BDcreation_reg` (reg1) goldens. Python `pipeline="reg1"` (decorrstretch + LAB k-means, seed 28) reproduces them pixel-for-pixel. |

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
