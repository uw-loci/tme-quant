# Running the CurveAlign pipeline

`examples/curvealign_pipeline.py` runs a full CurveAlign analysis on one image from the command line, with no GUI. It is also the reference for the Fiji plugin ([issue #66](https://github.com/uw-loci/tme-quant/issues/66)): the plugin should mirror its `run_curvealign()` function.

## 1. Set up the environment

Curvelet mode needs `curvelops`, which you build locally because CurveLab cannot be redistributed. See [INSTALL.md](INSTALL.md) for how to get CurveLab.

**Linux, WSL and macOS:** install once, then put `uv run` before each `python` command below.

```bash
bash bin/install.sh
```

On WSL, use a clone inside the Linux file system, not under `/mnt/`. See [DEVELOPMENT.md](DEVELOPMENT.md#command-line-pipeline-examplescurvealign_pipelinepy).

**Windows:** in an MSYS2 UCRT64 shell, activate the environment where you built `curvelops`, then point Python at the source folder:

```bash
source <path to your curvelops environment>/bin/activate
export PYTHONPATH=src
```

**`No module named 'tkinter'`:** `process_image` imports Tk. The Python that `bin/install.sh` sets up on WSL includes it, but other Python installs may not. Install it for the Python you're using, for example `sudo apt install python3-tk` on Ubuntu or `brew install python-tk` with Homebrew.

## 2. Run it

Run these from the repo root. Only the image path is required.

Each run overwrites earlier results for the same image in the output folder, so use `--output-dir` to keep results you want to compare.

```bash
# Default settings; results go to tests/test_images/CA_Out/
python examples/curvealign_pipeline.py tests/test_images/real1.tif

# Your own settings, with boundary analysis
python examples/curvealign_pipeline.py tests/test_images/real1.tif \
    --output-dir ../ca_results/real1 \
    --keep 0.05 \
    --boundary-mask tests/test_images/CA_Boundary/mask_real1.tiff \
    --distance-threshold 150
```

## 3. Set the parameters

| Option | Default | What it does |
|---|---|---|
| `input` (required) | none | Image file; its name also becomes the output file prefix |
| `--output-dir` | `<image folder>/CA_Out` | Where results are written; created if missing |
| `--keep` | `0.1` | Fraction of curvelet coefficients to keep |
| `--boundary-mask` | none | TIFF mask; turns on boundary analysis |
| `--distance-threshold` | `100` | How far from the boundary to evaluate fibers, in pixels |
| `--fiber-mode` | `0` | `0` = curvelet; `1`, `2`, `3` = read existing CT-FIRE results |
| `--fire-dir` | none | Folder containing `ct_fire_out/ct_fire_out<image name>.mat`; required for modes 1, 2 and 3 |

Other settings, such as the curvelet group radius and heatmap filters, have no command-line option. To change them, edit `AdvancedAnalysisOptions` at Step 5 of `run_curvealign()`.

FIRE modes have not had a full test run yet.

## 4. Read the results

The terminal shows the fiber count, angle and alignment statistics, and the files written. Each file name starts with the image name:

| File ending | Contents | Written |
|---|---|---|
| `_overlay.tiff` | Fibers drawn on the image | Always |
| `_procmap.tiff` | Alignment heatmap | Always |
| `_FiberFeatures.xlsx` | Per-fiber table | Always |
| `_hist.csv`, `_values.csv` | Angle histogram and angle values | Always |
| `_reconstructed.tiff` | Image rebuilt from the kept curvelets | Curvelet mode only |
| `_BoundaryMeasurements.xlsx` | Boundary measurements | With a boundary mask only |

The `total_length`, `end_to_end_length`, `curvature` and `width` columns are currently always empty (NaN). CSV boundaries are not implemented, so only TIFF masks are offered.

## 5. Call it from Python

This is the same pattern the Fiji plugin follows, except that Fiji supplies the image array and the parameter values.

```python
import sys

sys.path.insert(0, "examples")
from curvealign_pipeline import load_grayscale, run_curvealign

if __name__ == "__main__":  # required when using a boundary mask
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    img = load_grayscale("tests/test_images/real1.tif")
    out = run_curvealign(img, "real1", "../ca_results/real1", keep=0.05)
    print(out["features"].head())    # per-fiber DataFrame, or None if no fibers were found
    print(out["files"]["overlay"])   # path to the overlay TIFF
```

The `reconfigure` line is needed on Windows. `process_image` prints emoji, which crash Python with `UnicodeEncodeError` when its output is piped or the shell uses a legacy encoding, as in an MSYS2 terminal. The command-line script already sets this for you, but code that calls `run_curvealign()` directly must set it itself.

## Check your setup

```bash
TMEQ_RUN_CURVELETS=1 python -m pytest tests/test_curvealign_pipeline.py -v -rs
```

All 10 tests should pass. Without `TMEQ_RUN_CURVELETS=1`, the 2 tests that need curvelets are skipped.
