# Testing Guide

This project has a native C++ extension, `fiber_backend`, and a curvelet dependency, `curvelops`. Both are ABI-specific, which means they were compiled against a specific Python runtime and platform binary interface. In practice, the test environment must match the Python build used to compile them. Using the wrong interpreter can make the backend tests silently skip instead of exercising the actual implementation.

`ABI` stands for Application Binary Interface. It is the low-level contract for compiled code: the calling convention, function layout, memory layout, runtime libraries, and platform/toolchain expectations. A binary built for one ABI is not guaranteed to work with a different Python build or a different compiler/runtime environment.

## Quick summary

- Windows parity testing: use the repo root `.venv` built from the MSYS2 UCRT64 Python 3.14 environment.
- Linux / macOS: use the project venv or working Python environment, then run the repo’s normal test commands.
- Curvelet-dependent tests require `TMEQ_RUN_CURVELETS=1`.
- If all C++ tests skip, the usual cause is a mismatch between the Python interpreter and the compiled backend.

---

## Windows / MSYS2 (recommended for parity runs)

This is the required environment for the MATLAB parity checks and the native extension tests.

### Required setup

Use the MSYS2 UCRT64 login shell and activate the repo `.venv`:

```bash
C:\msys64\usr\bin\bash.exe -lc "export MSYSTEM=UCRT64 && source /etc/profile && \
  cd /h/GitHub.06.2022/tmequant_ctfire/tme-quant && source .venv/bin/activate && \
  export TMEQ_RUN_CURVELETS=1 && export MPLCONFIGDIR=/c/msys64/tmp/mpl && \
  python -V"
```

The repo’s current setup expects the `.venv` created from the MSYS2 UCRT64 Python 3.14 runtime, matching the compiled file:

- `.venv/`
- `fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd`

### Run the full suite

```bash
C:\msys64\usr\bin\bash.exe -lc "export MSYSTEM=UCRT64 && source /etc/profile && \
  cd /h/GitHub.06.2022/tmequant_ctfire/tme-quant && source .venv/bin/activate && \
  export TMEQ_RUN_CURVELETS=1 && export MPLCONFIGDIR=/c/msys64/tmp/mpl && \
  python -m pytest tests/ -v"
```

### Run just the MATLAB parity targets

```bash
C:\msys64\usr\bin\bash.exe -lc "export MSYSTEM=UCRT64 && source /etc/profile && \
  cd /h/GitHub.06.2022/tmequant_ctfire/tme-quant && source .venv/bin/activate && \
  export TMEQ_RUN_CURVELETS=1 && export MPLCONFIGDIR=/c/msys64/tmp/mpl && \
  python -m pytest tests/test_fire_2d_angle.py -q"
```

```bash
C:\msys64\usr\bin\bash.exe -lc "export MSYSTEM=UCRT64 && source /etc/profile && \
  cd /h/GitHub.06.2022/tmequant_ctfire/tme-quant && source .venv/bin/activate && \
  export TMEQ_RUN_CURVELETS=1 && export MPLCONFIGDIR=/c/msys64/tmp/mpl && \
  python -m pytest tests/test_ct_fire.py -q"
```

### Why this matters

Running pytest under an unrelated interpreter (for example, the Windows Store Python or a throwaway virtual environment) can cause the C++ backend not to import. In that case, the tests may skip instead of running, which hides real regressions.

---

## Linux / macOS

The project’s general test entry points are:

```bash
make test
```

Headless mode:

```bash
QT_QPA_PLATFORM=offscreen make test
```

### macOS checklist

Use this checklist when running on macOS:

```bash
# 1) Install the toolchain if needed
xcode-select --install

# 2) Create or activate the project environment
python -m venv .venv
source .venv/bin/activate

# 3) Install project dependencies
pip install -U pip
pip install -e .

# 4) Enable curvelet parity tests when curvelops is available
export TMEQ_RUN_CURVELETS=1

# 5) Run the parity tests
python -m pytest tests/test_fire_2d_angle.py -q
python -m pytest tests/test_ct_fire.py -q

# 6) Or run the full suite
python -m pytest tests/ -v
```

If you are testing the ctFIRE parity path locally instead of using the repo default shell workflow, use the same pattern:

```bash
source .venv/bin/activate
export TMEQ_RUN_CURVELETS=1
python -m pytest tests/test_fire_2d_angle.py -q
python -m pytest tests/test_ct_fire.py -q
```

Make sure the Python environment matches the compiled backend/curvelet installation or the C++ tests will not run as intended.

### Why the suite can skip tests

The full suite in this workspace was run with skip reporting enabled (`pytest -rs`), and the skipped cases were not failures. They were intentionally skipped because the project’s optional data or dependencies are not present in the current environment.

The exact reasons were:

- `tests/napari_curvealign_test.py` skipped because `napari` is not installed.
- `tests/test_cpp_functions.py` skipped several MATLAB-reference checks because the input image `tests/test_images/2B_D9_ROI1.tif` is missing.
- the same file skipped additional MATLAB reference checks because reference `.mat` files under `tests/test_results/cpp_test_files/` are missing.
- `tests/test_pipeline_validation.py` skipped because the MATLAB reference files it expects under `tests/test_results/cpp_test_files/` are missing.

In total, the project’s Windows/MSYS2 full suite currently reports:

- 144 passed
- 13 skipped
- 0 failed

This means the suite is healthy in the supported environment, and the skips are due to optional or data-dependent tests rather than code failures.

---

## Troubleshooting

### C++ backend tests skip unexpectedly

Cause: wrong interpreter or missing compiled extension.

Fix:

- make sure you are using the same Python build that compiled `fiber_backend`
- activate the repo `.venv`
- on Windows, use the MSYS2 UCRT64 shell

### Curvelet tests are skipped

Cause: `curvelops` not installed or `TMEQ_RUN_CURVELETS` unset.

Fix:

```bash
export TMEQ_RUN_CURVELETS=1
```

### Matplotlib warning under MSYS2

The repo recommends:

```bash
export MPLCONFIGDIR=/c/msys64/tmp/mpl
```

This avoids a home-directory cache issue in the MSYS2 login shell.

---

## Verified project status

As of the current workspace, the parity suite was verified in the supported MSYS2 UCRT64 environment:

- `tests/test_fire_2d_angle.py`: 25 passed
- `tests/test_ct_fire.py`: 57 passed

This is the current evidence for the repo’s ctFIRE MATLAB parity checks.
