# Getting Started — TMEQuant

Step-by-step guide for setting up a conda environment and running the two
primary example scripts on Windows Subsystem for Linux (WSL).

---

## Project layout

```
tme-quant/src/tme_quant/       ← project root (pyproject.toml lives here)
├── pyproject.toml
├── docs/
│   └── getting_started.md     ← this file
└── src/
    ├── tme_quant/             ← library source
    └── examples/
        ├── example_hierarchy_object_analysis.py
        ├── examples_tmequant_complete_cl_usage_ctfire_curvealign.py
        └── smoke_test_ctfire_curvealign.py
```

All commands below assume the project root as the working directory:

```bash
cd /mnt/h/GitHub.06.2022/tme-quant/src/tme_quant
```

---

## Prerequisites

- Miniconda or Anaconda installed in WSL Ubuntu
- `gcc` / `g++` available in WSL (pre-installed on Ubuntu; install with
  `sudo apt install build-essential` if missing)
- The project cloned / present at the path above

---

## Step 1 — Create the conda environment

```bash
conda create -n tmequant python=3.10 -y
conda activate tmequant
```

Python 3.10 is the tested minimum. 3.11 and 3.12 also work.

---

## Step 2 — Install curvelops (optional but recommended)

[curvelops](https://github.com/PyLops/curvelops) provides the production-quality
fast discrete curvelet transform (FDCT) used by CurveAlign and CT-FIRE.
It requires `fftw` and a C compiler.

```bash
conda install -c conda-forge pyfftw -y
pip install "curvelops>=0.23.0"
```

**If curvelops fails to build, skip this step and continue.**
The library automatically falls back to a NumPy FFT approximation and emits
a `UserWarning` during curvelet transforms. Results are still produced but
3-D volumetric accuracy is reduced.

---

## Step 3 — Install PyTorch and StarDist

StarDist (used for cell segmentation in the CT-FIRE / CurveAlign example)
depends on `csbdeep`, which in turn depends on TensorFlow. Install PyTorch
first, then StarDist.

### CPU-only (sufficient for the examples)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "stardist>=0.8.3" "cellpose>=2.0.0"
```

### GPU (CUDA) — faster StarDist inference

Replace the CPU torch line with your CUDA version. Example for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "stardist>=0.8.3" "cellpose>=2.0.0"
```

> **TensorFlow `libdevice` warning** — you may see messages like
> `libdevice not found at ./libdevice.10.bc` at runtime. This is a
> TensorFlow / CUDA toolkit path issue and does **not** affect correctness
> when running on CPU. To suppress it:
>
> ```bash
> export CUDA_VISIBLE_DEVICES=""   # forces CPU; add to ~/.bashrc to make permanent
> ```

---

## Step 4 — Install tme-quant in editable mode

From the project root, install the package with the `[examples]` extras group.
This covers every dependency required by both example scripts:

```bash
pip install -e ".[examples]"
```

The editable install (`-e`) makes `tme_quant` importable directly from `src/`
without any `PYTHONPATH` manipulation. The `[examples]` group installs:

| Group | Packages |
|---|---|
| core | numpy, scipy, pandas, scikit-image, shapely, matplotlib, opencv-python, tifffile, imageio, openpyxl |
| `[fiber]` | curvelops, pyimagej, scyjava |
| `[dl]` | torch, stardist, cellpose |
| `[network-lite]` | networkx |

Verify the install:

```bash
pip show tme-quant
pip list | grep -E "numpy|scipy|shapely|stardist|curvelops|networkx"
```

---

## Step 5 — Run the examples

After the editable install, `PYTHONPATH` is no longer needed.

### Example 1 — Hierarchy demo

No image files required — all objects are constructed programmatically.

```bash
python src/examples/example_hierarchy_object_analysis.py
```

Expected output: 11 demo sections print to stdout, ending with
`Demo complete — all hierarchy patterns executed successfully.`

### Example 2 — CT-FIRE / CurveAlign workflows

Workflows 1 and 2 require real image files. Place them in a `data/` folder
at the project root (or adjust the paths in `__main__`):

```
data/
├── patient_001_HE.tif
└── patient_001_SHG.tif
```

Workflow 3 (3-D volumetric) generates a synthetic volume internally and
requires no data files.

```bash
python src/examples/examples_tmequant_complete_cl_usage_ctfire_curvealign.py
```

### Smoke test (no image files needed)

Verifies all import chains, param construction, FiberAnalyzer 2-D/3-D, and
TACS classification without real images:

```bash
python src/examples/smoke_test_ctfire_curvealign.py
```

Expected output: `PASSED : 41/41 — All checks passed.`

---

## Step 6 — Run the tests

### Standard suite (no curvelops required)

```bash
# From src/tme_quant/  (project root containing pyproject.toml)
pytest tests/ -v
```

Expected: all tests pass; curvelops integration tests are skipped automatically
when curvelops is not installed.

### Full suite with curvelops

`extract_curvelet_fiber_candidates` and its real-dataset tests require
curvelops.  If curvelops is installed in a separate environment (e.g. WSL
miniconda — see Step 2), run pytest with that interpreter:

```bash
# Example: WSL miniconda
wsl bash -c "cd /mnt/h/GitHub.06.2022/tme-quant/src/tme_quant && \
    ~/miniconda3/bin/python -m pytest tests/ -v"
```

All tests should pass with 0 skipped when curvelops is available.

### MATLAB-reference parity checks

A subset of `test_curvelet_fiber_candidates.py` compares output against
MATLAB-generated reference CSVs.  These are skipped by default and enabled
with an environment variable:

```bash
TMEQ_VALIDATE_MATLAB=1 pytest tests/test_curvelet_fiber_candidates.py -v
# or from WSL:
wsl bash -c "cd /mnt/.../tme_quant && \
    TMEQ_VALIDATE_MATLAB=1 ~/miniconda3/bin/python -m pytest \
    tests/test_curvelet_fiber_candidates.py -v"
```

Reference CSVs live in
`H:/GitHub.06.2022/tme-quant/tests/test_results/new_curv_test_files/`.

---

## Optional dependency groups

Install additional groups as needed with `pip install -e ".[group]"`:

| Group | Command | Use case |
|---|---|---|
| Core only | `pip install -e "."` | Hierarchy example, library development |
| `[fiber]` | `pip install -e ".[fiber]"` | Production CurveAlign/CT-FIRE curvelet backend + Fiji bridge |
| `[dl]` | `pip install -e ".[dl]"` | StarDist / Cellpose cell segmentation |
| `[examples]` | `pip install -e ".[examples]"` | **Both example scripts (recommended starting point)** |
| `[network]` | `pip install -e ".[network]"` | Full network visualization (adds plotly + python-louvain) |
| `[full]` | `pip install -e ".[full]"` | Everything except the napari Qt stack |
| `[napari]` | `pip install -e ".[napari]"` | napari GUI plugin |
| `[dev]` | `pip install -e ".[dev]"` | Development tools (pytest, black, ruff, mypy) |

Multiple groups can be combined:

```bash
pip install -e ".[examples,napari,dev]"
```

---

## Non-PyPI research packages

`voxelmorph` and `comir` (used by deep learning registration methods) are
not on PyPI. Install them manually if needed:

```bash
pip install voxelmorph
pip install git+https://github.com/CVLAB-Unibo/comir
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tme_quant'`**

The editable install is not active in this environment. Re-run from the
project root:

```bash
pip install -e ".[examples]"
```

Or fall back to the explicit path approach:

```bash
PYTHONPATH=src python src/examples/example_hierarchy_object_analysis.py
```

**`libdevice not found at ./libdevice.10.bc`** (TensorFlow warning)

Not an error — StarDist continues on CPU. Add `export CUDA_VISIBLE_DEVICES=""`
to `~/.bashrc` to suppress permanently.

**`curvelops` build failure**

Skip Step 2. The NumPy FFT fallback activates automatically. You will see:

```
UserWarning: curvelops not installed — NumPy FFT fallback in use.
             Out-of-plane fibers will NOT be detected in 3-D mode.
```

Install curvelops later by repeating Step 2 once build tools are available.

**`WARNING: Using NumPy FFT fallback`** during CurveAlign / CT-FIRE

Install curvelops (Step 2) to eliminate this and enable the full 3-D
volumetric curvelet transform.

**`python-louvain` install fails**

The `[network]` group (not needed for the examples) includes this package.
If it fails, try pinning the version:

```bash
pip install python-louvain==0.16
```