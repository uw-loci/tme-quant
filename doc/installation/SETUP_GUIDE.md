# tme-quant Setup Guide

This guide covers installation of tme-quant with the **curvelet backend** (curvelops) and Napari plugin. For one-click installation, use [QUICK_START.md](QUICK_START.md).

## Quick Start (Recommended)

Use the automated install script:

```bash
cd tme-quant
bash bin/install.sh
```

This script will:
1. Check for uv (exits if not found)
2. Download and build FFTW 2.1.5 in `../utils`
3. Detect CurveLab in `../utils`
4. Sync the uv environment with curvelops
5. Install tme-quant and run verification

**Prerequisites:** Clone this repo, install [uv](https://docs.astral.sh/uv/), and download CurveLab to `../utils`. See [QUICK_START.md](QUICK_START.md).

## Manual Installation

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install FFTW (for curvelet backend)

**Best practice:** Place FFTW in `../utils` (parallel to tme-quant) so it matches the install script layout:

```bash
# Create utils directory next to tme-quant
cd /path/to/parent/of/tme-quant
mkdir -p utils
cd utils

# Download FFTW
curl -L -O http://www.fftw.org/fftw-2.1.5.tar.gz
tar xzf fftw-2.1.5.tar.gz
cd fftw-2.1.5

# Configure and build (PIC required for shared libs used by curvelops)
./configure --prefix="$(pwd)" --disable-fortran CFLAGS="-fPIC"
make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || 4)
make install

export FFTW="$(pwd)"
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"
```

**Note:** Set `FFTW` in your shell config (e.g. `~/.bashrc`) if you build CurveLab in a later session.

### Step 3: Install CurveLab

CurveLab requires a separate download due to licensing:
1. Visit: https://curvelet.org/download.php
2. Agree to the license and download CurveLab
3. Extract to `../utils` (e.g. `../utils/CurveLab-2.1.3`)

```bash
# Set the CurveLab path (adjust to your version)
export FDCT="/path/to/utils/CurveLab-2.1.x"

# Build CurveLab components (if needed)
cd "$FDCT/fdct_wrapping_cpp/src"
make FFTW_DIR="$FFTW"

cd "$FDCT/fdct/src"
make FFTW_DIR="$FFTW"

cd "$FDCT/fdct3d/src"
make FFTW_DIR="$FFTW"
```

### Step 4: Install tme-quant with uv

```bash
cd tme-quant

# Ensure FFTW and FDCT are set (from steps above)
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"

# With curvelet backend
uv sync --extra curvelops
uv pip install -e .
```

### Step 5: Verify installation

```bash
uv run python -c "
import pycurvelets as pc
print('Installed ✅')
print('HAS_CURVELETS =', pc.HAS_CURVELETS)
try:
    import curvelops
    print('curvelops', curvelops.__version__)
except ImportError:
    print('curvelops: not installed')
"
```

## Package Installation Options

### Basic (no curvelet backend)

```bash
uv sync
uv pip install -e .
```

Core API only; no curvelops. Install curvelops manually if you need it later.

### With curvelet backend

```bash
uv sync --extra curvelops
uv pip install -e .
```

Requires FFTW and CurveLab as above.

### Using pip instead of uv

If you prefer pip, create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[curvelops]"
```

Requires FFTW and CurveLab with `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) set.

## Troubleshooting

### uv not found

- Install uv: https://docs.astral.sh/uv/
- Or run: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### FFTW build errors

- **macOS:** Ensure Xcode Command Line Tools: `xcode-select --install`
- **Linux:** `sudo apt-get install build-essential gcc g++ make curl`
- Always use `CFLAGS="-fPIC"` for shared library builds

### CurveLab not found

- Download from https://curvelet.org/
- Place in `../utils/` (e.g. `../utils/CurveLab-2.1.3`) for compatibility with install script
- Set `FDCT` to the CurveLab directory

### curvelops build errors

- Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to your install roots
- Verify FFTW 2D/3D libraries were built

### Plugin not showing in napari

- Ensure the package is installed: `uv pip install -e .` (the install script does this)
- Restart napari after installing
- Check the plugin is registered: `uv run python -c "from importlib.metadata import entry_points; print([ep.name for ep in entry_points(group='napari.manifest') if 'curvealign' in ep.name])"`

## Activation

After installation:

```bash
uv run napari
```

The CurveAlign widget appears in: **Plugins → napari-curvealign** (or **CurveAlign for Napari**) → **CurveAlignPy**

## Making the napari plugin available (for developers)

The plugin is registered via two pieces:

1. **`src/napari_curvealign/napari.yaml`** – Declares the plugin (commands, widgets, display names).
2. **`pyproject.toml`** – Entry point that tells napari where to find the manifest:
   ```toml
   [project.entry-points."napari.manifest"]
   napari-curvealign = "napari_curvealign:napari.yaml"
   ```

To make it available in your environment:

```bash
# Install the package (editable, so changes take effect immediately)
uv pip install -e .
# or: uv sync  (installs the project from pyproject.toml)

# Run napari
uv run napari
```

Then open **Plugins** in the napari menu and select the CurveAlign widget.

## Running for testing

- **Interactive:** `uv run napari` → Plugins → napari-curvealign
- **Headless tests:** `QT_QPA_PLATFORM=offscreen make test` (or `uv run pytest`)
