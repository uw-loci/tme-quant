# BDcreation annotation comparison figures (tests 1-3)

Annotation only (`BDcreationHE2` / `BDcreationHE`). The script calls
`pycurvelets.tumor_annotation_from_he` and writes PNGs under
`tests/artifacts/bdc_annotation_viz/current/`. That directory is gitignored.
File names are listed in [ARTIFACTS.md](ARTIFACTS.md). Registration figures
are described in [BDC_REGISTRATION_COMPARISONS.md](BDC_REGISTRATION_COMPARISONS.md).

The figures compare the annotation API to MATLAB masks on the
already-registered HE fixtures (`HE_registered_test{1,2,3}`).

Annotation figures are worth generating even when the port is pixel-exact:
unlike registration (MAE/SSIM on RGB), the deliverable is a **binary tumor
mask**. The panels show region overlap (TP/FP/FN), boundary alignment, and
Hausdorff / boundary-F1, which a single IoU number hides.

Regenerate:

```bash
python archive/visualization/generate_annotation_comparison.py
python archive/visualization/generate_annotation_comparison.py --method hsv
python archive/visualization/generate_annotation_comparison.py --method rgb_kmeans
```

The generator is a development script in `archive/visualization/`, not a
pytest test. Output is gitignored. Pass an output directory as the first
argument to write somewhere else.

## Files

| File | What it is |
| --- | --- |
| `archive/visualization/generate_annotation_comparison.py` | patient_001 tests 1-3. CLI: `[out_dir] [--method hsv\|rgb_kmeans]`. |
| `comparison_he2_test{1,2,3}_ppm*_hsv.png` | HE2 / HSV (`annotation_method="hsv"`) vs committed CA_Boundary goldens (fresh `BDcreationHE2.m`). |
| `comparison_he_test{1,2,3}_ppm*_rgb_kmeans.png` | HE / RGB k-means (`annotation_method="rgb_kmeans"`) vs `archive/matlab_parity/dumps/he_testN/images.mat` (`rng(28,'twister')`). |

3x4 panel: HE input, MATLAB mask, Python mask, dual contours on HE;
MATLAB/Python overlays; TP/FP/FN; XOR; boundary pixels; disagreement
distance to the MATLAB boundary; TP/FP/FN counts; stats box (IoU, Dice,
pixel accuracy, Hausdorff, boundary F1, n_diff).

## Results with the current build

Filled by `archive/visualization/generate_annotation_comparison.py`. IoU / Dice / exact are Python vs MATLAB
on the binary mask. `n_diff=0` is pixel identity.

| Case | Method | ppm | IoU | Dice | Exact | n_diff | Hausdorff (px) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test1 | HE2 HSV | 1.5 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test2 | HE2 HSV | 2.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test3 | HE2 HSV | 3.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test1 | HE RGB k-means | 1.5 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test2 | HE RGB k-means | 2.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |
| test3 | HE RGB k-means | 3.0 | 1.0000 | 1.0000 | 100 % | 0 | 0 |

## Future annotation work

- Compare both methods against independently reviewed tumor masks from
  diverse, held-out images, using overlap and boundary metrics. MATLAB
  parity alone does not measure annotation accuracy.
- Investigate empty or unexpectedly small HSV masks, including the effects
  of staining, thresholds, morphology, and image resolution.
- Evaluate each annotation algorithm in both RGB and HSV color spaces with
  comparable preprocessing. Account for the circular nature of hue and
  its instability at low saturation.
- Expose additional settings through the user interface or API to handle
  different images, such as HSV hue ranges, RGB clustering settings, and
  morphology thresholds and scales. Preserve the existing MATLAB-compatible
  settings as a reproducible option and validate configurable alternatives.
- Explore tumor-boundary annotation based on segmented cells.
- Add a convenient way to generate and compare boundary overlays for new
  images, recording the method and pixels per micron used.
