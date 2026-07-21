# Preparing MATLAB Parity Tests for New Images

This guide describes how to add a new test image, generate the corresponding MATLAB reference files, register the test cases, and run the parity verification suite.

---

## Workflow Overview

To add MATLAB parity tests for a new image, you will follow these four steps:

```
[Add Image] ──> [Run MATLAB & Save .mat] ──> [Register in JSON] ──> [Run Pytest]
```

---

## Step 1: Add the Raw Test Image

Place your raw input image in the standard test images directory:
*   **Path**: `tests/test_images/` (e.g., [tests/test_images/new_sample.tif](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_images/))

---

## Step 2: Generate MATLAB reference files

You need to save the outputs from your MATLAB execution of ctFIRE.

### A. Curvelet Reconstruction Reference (for `ct_fire` verification)
This verifies that the Python curvelet reconstruction step matches the MATLAB output prior to fiber extraction.

1.  In MATLAB ctFIRE, run curvelet reconstruction.
2.  Save the `CTRimage` variable immediately after reconstruction (before `fire_2D_ang1` is called) in **v7.3 HDF5 format**:
    ```matlab
    recon_img = CTRimage;
    save('recon_img_new_sample_SS3_TH02.mat', 'recon_img', '-v7.3');
    ```
3.  **Naming Convention**: `recon_img_{image_base}_SS{num_scales}_TH{int(coefficient_percentile*10):02d}.mat`
4.  **Save to**: [tests/test_results/ct_fire_test_files/](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/ct_fire_test_files/)

### B. Core FIRE 2D Reference (for `fire_2d_angle` verification)
This verifies the fiber extraction algorithm matches the MATLAB implementation.

1.  Save the structure returned by `fire_2D_ang1(p, im3, 0)`:
    ```matlab
    % Assuming the output structure is named 'data'
    save('test_fire_2d_new_sample_default.mat', 'data', '-v7.3');
    ```
2.  **Naming Convention**: `test_fire_2d_{image_base}_{parameter_suffix}.mat`
3.  **Save to**: [tests/test_results/fire_2d_test_files/](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/fire_2d_test_files/)

---

## Step 3: Register the Test Cases

Add your new test case details to the corresponding JSON configuration files so the test suite automatically discovers them.

### A. For FIRE 2D Extraction Parity
Add a case entry to the array in [tests/test_results/fire_2d_test_files/test_cases_fire_2d.json](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/fire_2d_test_files/test_cases_fire_2d.json):

```json
{
  "name": "new_sample_default",
  "image": "new_sample.tif",
  "description": "New sample image with default parameters",
  "params": {
    "sigma_im": 0,
    "sigma_d": 0.3,
    "dtype": "cityblock",
    "thresh_im2": 5,
    "thresh_Dxlink": 1.5,
    "s_xlinkbox": 8,
    "thresh_LMP": 0.2,
    "thresh_LMPdist": 2,
    "thresh_ext": 0.342,
    "lam_dirdecay": 0.5,
    "s_minstep": 2,
    "s_maxstep": 6,
    "thresh_dang_aextend": 0.9848,
    "thresh_dang_L": 15,
    "thresh_short_L": 15,
    "s_fiberdir": 4,
    "thresh_linkd": 15,
    "thresh_linka": -0.866,
    "thresh_flen": 15,
    "thresh_numv": 3,
    "scale": [1.0, 1.0, 1.0],
    "s_boundthick": 10,
    "blist": 1,
    "s_maxspace": 5,
    "lambda": 0.01,
    "ang_interval": 3
  },
  "expected_outputs": {
    "min_fiber_count": 10,
    "max_fiber_count": 500,
    "min_avg_length": 15,
    "max_avg_length": 100
  },
  "matlab_reference_mat": "test_fire_2d_new_sample_default.mat"
}
```

### B. For End-to-End CT-FIRE Parity
Add a case entry to the array in [tests/test_results/ct_fire_test_files/test_cases_ct_fire.json](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/ct_fire_test_files/test_cases_ct_fire.json):

```json
{
  "name": "new_sample_default",
  "image": "new_sample.tif",
  "description": "New sample image with default CT-FIRE parameters",
  "ctfire_params": {
    "coefficient_percentile": 0.2,
    "num_scales": 3,
    "LL1": 30,
    "value": {
      "sigma_im": 0,
      "sigma_d": 0.3,
      "dtype": "cityblock",
      "thresh_im2": 5,
      ...
    }
  },
  "matlab_reference_mat": "test_ct_fire_new_sample_default.mat"
}
```

---

## Step 4: Generate CSV References (Optional)

If you are adding alignment tests (`process_image` or `new_curv`), you must convert the MATLAB `.mat` struct file into a `.csv` reference.

1.  Use the helper conversion script [tests/test_results/mat_to_csv.py](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/mat_to_csv.py):
    ```bash
    # Standard format conversion
    python tests/test_results/mat_to_csv.py path/to/reference.mat tests/test_results/process_image_test_files/test_process_image_new_sample_default.csv
    ```
2.  For `new_curv` metrics (which extracts coordinates and orientation angles), include the `--new-curv` flag:
    ```bash
    python tests/test_results/mat_to_csv.py path/to/reference.mat tests/test_results/new_curv_test_files/test_new_curv_new_sample.csv --new-curv
    ```

---

## Step 5: Run the Parity Verification

To verify that the Python/C++ pipeline produces parity with the MATLAB outputs:

### A. Set up the correct Environment
Parity tests require the compiled C++ `fiber_backend` and the `curvelops` library. 
On Windows, you must run the tests from your **MSYS2 UCRT64 environment** where the compiled `.pyd` is located (see [doc/DEVELOPMENT.md](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/doc/DEVELOPMENT.md#L23) for details).

### B. Execute Pytest
Run the full test suite from your target environment with `TMEQ_RUN_CURVELETS=1` exported to ensure curvelet-dependent tests are not skipped:

```bash
# Export the flag to activate curvelet comparisons
export TMEQ_RUN_CURVELETS=1

# Run the 2D FIRE tests
pytest tests/test_fire_2d_angle.py -v

# Run the end-to-end CT-FIRE tests
pytest tests/test_ct_fire.py -v
```

If a test case fails, consult [doc/MATLAB_PARITY_ANALYSIS.md](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/doc/MATLAB_PARITY_ANALYSIS.md) and [TOLERANCE_RATIONALE.md](file:///h:/GitHub.06.2022/tmequant_ctfire/tme-quant/tests/test_results/cpp_test_files/TOLERANCE_RATIONALE.md) to evaluate if the discrepancy is within expected floating-point or boundary limits.
