# macOS Intel (x86_64) Installation Troubleshooting

## Problem

Running `bash bin/install.sh` fails at the verification step with:

```
✗ Validation failed:
  - pycurvelets: HAS_CURVELETS is False (curvelet backend not functional; check FFTW/FDCT)
  - curvelops: No module named 'curvelops'
```

## Root Cause

The install script runs:

```bash
uv sync --extra curvelops --extra segmentation
```

The `segmentation` extra depends on `cellpose` and `cellcast`, which transitively depend on `torch`. PyTorch 2.12.0 does not publish a wheel for macOS Intel (`macosx_*_x86_64`), only for:

- `manylinux_2_28_aarch64`
- `manylinux_2_28_x86_64`
- `macosx_14_0_arm64`
- `win_amd64`

Because `uv sync` resolves all extras together, the missing torch wheel causes the entire sync to fail. As a result, `curvelops` is never installed even though it has no dependency on torch.

## Fix

Install the `curvelops` extra without `segmentation`:

```bash
export FFTW="/path/to/utils/fftw-2.1.5"
export FDCT="/path/to/utils/CurveLab-2.1.3"
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"

uv sync --extra curvelops
uv pip install -e .
```

This installs curvelops and the napari plugin without pulling in torch. The segmentation features (cellpose, cellcast) will not be available on macOS Intel unless torch is pinned to an older version that still supported x86_64 (2.5.x or earlier).

After syncing, pin the Qt6 runtime to match the PyQt6 bindings:

```bash
uv pip install "pyqt6-qt6==6.11.0"
```

### Why

`uv sync` resolves `pyqt6-qt6==6.11.1`, but the `pyqt6==6.11.0` Python bindings were compiled against Qt 6.11.0. The cocoa platform plugin from 6.11.1 is rejected by Qt's plugin loader due to this version mismatch, producing:

```
Could not find the Qt platform plugin "cocoa" in ""
This application failed to start because no Qt platform plugin could be initialized.
```

Pinning `pyqt6-qt6` to 6.11.0 aligns the runtime libraries with the bindings and resolves the error. Note that `uv sync` may revert this pin, so re-run the command if napari fails to start after a sync.

## Running Napari

```bash
conda deactivate   # if conda is active; its env vars interfere with Qt
uv run napari
```

Then go to **Plugins → napari-curvealign**.

## Verification

```bash
uv run python -c "
import pycurvelets as pc
print(f'HAS_CURVELETS = {pc.HAS_CURVELETS}')
import curvelops
print(f'curvelops {curvelops.__version__}')
import napari_curvealign
print('napari_curvealign OK')
"
```

Expected output:

```
HAS_CURVELETS = True
curvelops 0.23.4
napari_curvealign OK
```
