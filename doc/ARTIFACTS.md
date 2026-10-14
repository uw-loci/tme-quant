# Comparison figures

These PNGs are a local visual check of the MATLAB ports. They are not an
input to `shg_he_registration` or `tumor_annotation_from_he`, and they are
not stored in git. Numeric checks live in `tests/test_shg_he_registration.py`
and `tests/test_tumor_annotation_from_he.py`.

The scripts that draw them are development tools in `archive/visualization/`.
They write under `tests/artifacts/` (gitignored, and pruned from the sdist).
How to read a panel, and the measured MAE / IoU tables, are in
[BDC_REGISTRATION_COMPARISONS.md](BDC_REGISTRATION_COMPARISONS.md) and
[BDC_ANNOTATION_COMPARISONS.md](BDC_ANNOTATION_COMPARISONS.md).

```bash
python archive/visualization/generate_registration_comparison.py
python archive/visualization/generate_registration_comparison_patient02.py
python archive/visualization/generate_annotation_comparison.py
```

## Registration (`tests/artifacts/bdc_regression_viz/current/`)

Written by `generate_registration_comparison.py` (patient_001, `pipeline="reg2"`,
`ecm_method="hsv"`) and `generate_registration_comparison_patient02.py`
(patient_02; tests 4-7 use `pipeline="reg2"`, tests 8-9 use `pipeline="reg1"`).

| Figure | Case |
| --- | --- |
| `comparison_test1_ppm1.5_matlab_hsv.png` | patient_001, ppm 1.5, `BDcreation_reg2` |
| `comparison_test2_ppm2.0_matlab_hsv.png` | patient_001, ppm 2.0, `BDcreation_reg2` |
| `comparison_test3_ppm3.0_matlab_hsv.png` | patient_001, ppm 3.0, `BDcreation_reg2` |
| `comparison_test4_reg2_roi2_ppm2.6_matlab_hsv.png` | patient_02 roi2, ppm 2.6, reg2 |
| `comparison_test5_reg2_roi4_ppm1.5_matlab_hsv.png` | patient_02 roi4, ppm 1.5, reg2 |
| `comparison_test6_reg2_roi4_ppm2.6_matlab_hsv.png` | patient_02 roi4, ppm 2.6, reg2 |
| `comparison_test7_reg2_roi5_ppm2.6_matlab_hsv.png` | patient_02 roi5, ppm 2.6, reg2 |
| `comparison_test8_reg1_roi4_ppm3.0_matlab_hsv.png` | patient_02 roi4, ppm 3.0, reg1 |
| `comparison_test9_reg1_roi4_ppm2.6_matlab_hsv.png` | patient_02 roi4, ppm 2.6, reg1 |

## Annotation (`tests/artifacts/bdc_annotation_viz/current/`)

Written by `generate_annotation_comparison.py` via
`tumor_annotation_from_he` (`annotation_method="hsv"` or `"rgb_kmeans"`).

| Figure | Case |
| --- | --- |
| `comparison_he2_test1_ppm1.5_hsv.png` | HE2 / HSV, ppm 1.5 |
| `comparison_he2_test2_ppm2.0_hsv.png` | HE2 / HSV, ppm 2.0 |
| `comparison_he2_test3_ppm3.0_hsv.png` | HE2 / HSV, ppm 3.0 |
| `comparison_he_test1_ppm1.5_rgb_kmeans.png` | HE / RGB k-means, ppm 1.5 |
| `comparison_he_test2_ppm2.0_rgb_kmeans.png` | HE / RGB k-means, ppm 2.0 |
| `comparison_he_test3_ppm3.0_rgb_kmeans.png` | HE / RGB k-means, ppm 3.0 |
