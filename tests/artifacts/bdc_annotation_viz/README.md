# BDcreation annotation comparison figures (tests 1-3)

This directory is **annotation only** (`BDcreationHE2` / `BDcreationHE`).
Registration figures live in `../bdc_registration_viz/`.

Visual record kept in git, not in the wheel or sdist (see `MANIFEST.in`).
Compares `pycurvelets.tumor_annotation_from_he` to MATLAB masks on the
already-registered HE fixtures (`HE_registered_test{1,2,3}`).

Annotation figures are worth generating even when the port is pixel-exact:
unlike registration (MAE/SSIM on RGB), the deliverable is a **binary tumor
mask**. The panels show region overlap (TP/FP/FN), boundary alignment, and
Hausdorff / boundary-F1, which a single IoU number hides.

Regenerate:

```bash
python tests/artifacts/bdc_annotation_viz/generate_comparison.py
python tests/artifacts/bdc_annotation_viz/generate_comparison.py --method hsv
python tests/artifacts/bdc_annotation_viz/generate_comparison.py --method rgb_kmeans
```

Only `current/` is tracked; other output directories are gitignored.

## Files

| File | What it is |
| --- | --- |
| `generate_comparison.py` | patient_001 tests 1-3. CLI: `[out_dir] [--method hsv\|rgb_kmeans]`. |
| `current/comparison_he2_test{1,2,3}_ppm*_hsv.png` | HE2 / HSV vs committed CA_Boundary goldens (fresh `BDcreationHE2.m`). |
| `current/comparison_he_test{1,2,3}_ppm*_rgb_kmeans.png` | HE / RGB k-means vs `dumps/he_testN/images.mat` (`rng(28,'twister')`). |

3x4 panel: HE input, MATLAB mask, Python mask, dual contours on HE;
MATLAB/Python overlays; TP/FP/FN; XOR; boundary pixels; disagreement
distance to the MATLAB boundary; TP/FP/FN counts; stats box (IoU, Dice,
pixel accuracy, Hausdorff, boundary F1, n_diff).

## Results with the current build

Filled by `generate_comparison.py`. IoU / Dice / exact are Python vs MATLAB
on the binary mask. `n_diff=0` is pixel identity.

| Case | Method | ppm | IoU | Dice | Exact | n_diff | Hausdorff (px) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test1 | HE2 HSV | 1.5 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test2 | HE2 HSV | 2.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test3 | HE2 HSV | 3.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test1 | HE RGB k-means | 1.5 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test2 | HE RGB k-means | 2.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test3 | HE RGB k-means | 3.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
