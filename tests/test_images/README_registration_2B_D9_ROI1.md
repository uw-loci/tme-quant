# Registration parity artifacts — `2B_D9_ROI1`

## MATLAB reference (golden)

- **`2B_D9_ROI1_registered_matlab.tif`** — Output of MATLAB `curvelets/.../CurveAlign_CT-FIRE/BDcreation_reg2.m`, committed for **pytest** comparison.
- **Inputs** (repo root, not under `tme-quant/`):  
  `utils/TestimagesCA6.0_20240722/HE/2B_D9_ROI1.tif`  
  `utils/TestimagesCA6.0_20240722/SHG/2B_D9_ROI1.tif`  
  Parameters: `pixelpermicron=2.0` (same as the MATLAB run used to build the golden).

## Python regression test

`tests/test_shg_he_registration.py::test_shg_he_registration_matches_matlab_golden_2b_d9_roi1` loads the golden TIFF, runs `pycurvelets.SHG_HE_registration.shg_he_registration` with the same inputs, and asserts **MAE / RMSE / per-channel NCC** within documented bounds.

- **Skips** if the golden file or `utils/...` inputs are missing (e.g. CI without full repo).
- **Requires SimpleITK** (`@pytest.mark.skipif(not has_simpleitk(), ...)`).

Bounds are **empirical** (Python uses SimpleITK Mattes MI + similarity/affine; MATLAB uses `imregconfig('multimodal')` / `imregtform`). They are **regression guards**, not pixel-equality checks.

## Optional: saved Python run

- **`2B_D9_ROI1_registered_python.tif`** — Optional manual export for visual diff; not required for pytest.

## Reproduce Python output

```python
from pycurvelets.SHG_HE_registration import SHGHERegistrationParameters, shg_he_registration

p = SHGHERegistrationParameters(
    HEfilepath=".../utils/TestimagesCA6.0_20240722/HE",
    HEfilename="2B_D9_ROI1.tif",
    pixelpermicron=2.0,
    SHGfilepath=".../utils/TestimagesCA6.0_20240722/SHG",
)
registered = shg_he_registration(p, save_output=False)
```

## Example metrics (update after pipeline changes)

Re-measure when `BDcreation_reg2.m` or Python registration changes, then adjust thresholds in `test_shg_he_registration.py`.

| Metric        | Example (same ROI) |
|---------------|---------------------|
| MAE           | ~0.033              |
| RMSE          | ~0.09               |
| NCC (R, G, B) | ~0.34, 0.36, 0.30   |
