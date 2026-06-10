# Setting up the CT-FIRE FIRE-only pipeline

This guide walks through setting up the **FIRE-only** CT-FIRE pipeline
(`use_ct_reconstruction=False`) on Windows, Linux, and macOS (Apple Silicon).

**What you get:** individual fiber extraction with full TACS classification and
the interactive matplotlib viewer — **without** installing curvelops or any
curvelet transform library.

---

## Pre-built extensions at a glance

Pre-built C++ extension binaries are included in `src/ctfire_py/` for all three
platforms. You **must** use the matching Python version or the binary will not load.

| Platform | Binary file | Python version required |
|----------|------------|------------------------|
| Windows (MSYS2 UCRT64 x64) | `fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd` | **3.14** |
| Linux (x86_64) | `fiber_backend.cpython-310-x86_64-linux-gnu.so` | **3.10** |
| macOS (Apple Silicon / M-chip) | `fiber_backend.cpython-312-darwin.so` | **3.12** |

If your Python version does not match, you will get
`ImportError: No module named 'fiber_backend'` or an ABI mismatch error.
In that case, rebuild the extension from source — see `src/ctfire_py/CPP/`.

---

## Directory layout and why we don't use `pip install`

Understanding the two-level `src/` structure explains why a normal
`pip install -e .` cannot set up this environment on its own.

```
tme-quant/                          <- git root (clone/copy goes here)
└── src/                            <- repo-level src: holds all packages as siblings
    ├── ctfire_py/                  <- ctfire_py package (includes fiber_backend.*)
    ├── pycurvelets/                <- pycurvelets utilities (required by ctfire_py)
    └── tme_quant/                  <- project root (pyproject.toml lives here)
        └── src/
            └── tme_quant/         <- actual tme_quant Python package
```

There are two distinct problems with using `pip install -e .` here:

**Problem 1 — uv ignores the activated venv.**
When `uv pip install -e .` is run from inside `src/tme_quant/` (which contains
`pyproject.toml`), uv detects the project and installs into the project's own
configured virtual environment (`.venv`) regardless of which venv is currently
activated. A separately created environment is silently bypassed.

**Problem 2 — ctfire_py and pycurvelets are outside the package discovery path.**
`pyproject.toml` uses `where = ["src"]`, which resolves to `src/tme_quant/src/`
(the project's own source layout). `ctfire_py` and `pycurvelets` sit one level
above at `src/` (repo-level) — outside that path. Even if the install went into
the right environment, those two packages would still not be importable.

**Solution — two `.pth` files written directly into site-packages.**
Python reads every `.pth` file in `site-packages/` at startup and appends the
listed paths to `sys.path`. Writing them manually sidesteps both problems above:

| .pth file | Path added | Provides |
|-----------|-----------|----------|
| `aa_tme_quant_pkg.pth` | `src/tme_quant/src/` | `tme_quant` package |
| `tme_quant_src.pth` | `src/` (repo-level) | `ctfire_py`, `pycurvelets` |

The `aa_` prefix on the first file ensures Python processes it alphabetically
before the second file. This matters because `src/` also contains the
`tme_quant/` project directory, which Python would otherwise pick up as a
namespace package and shadow the real `tme_quant` package.

The `.pth` file commands are **identical on all platforms** — only the Python
and package installation steps differ by OS.

---

## Platform setup

Follow the section for your OS, then continue at
[Create the environment and add path files](#create-the-environment-and-add-path-files).

### Windows (MSYS2 UCRT64)

**Python required: 3.14**

#### Step W1 — Install MSYS2

1. Download the installer from **https://www.msys2.org**.
2. Run it; accept the default path `C:\msys64`.
3. Open the **MSYS2 UCRT64** shell (Start menu → "MSYS2 UCRT64" or `C:\msys64\ucrt64.exe`).

   > Always use the **UCRT64** shell, not MSYS, MinGW32, or MinGW64.
   > The `fiber_backend` extension is linked against the UCRT64 runtime.

4. Update the package database:
   ```bash
   pacman -Syu
   # The shell may close; re-open UCRT64 and run again:
   pacman -Su
   ```

#### Step W2 — Install Python 3.14 and C-extension dependencies

```bash
pacman -S --needed \
  mingw-w64-ucrt-x86_64-python \
  mingw-w64-ucrt-x86_64-python-pip \
  mingw-w64-ucrt-x86_64-python-numpy \
  mingw-w64-ucrt-x86_64-python-scipy \
  mingw-w64-ucrt-x86_64-python-scikit-image \
  mingw-w64-ucrt-x86_64-python-opencv \
  mingw-w64-ucrt-x86_64-python-pandas \
  mingw-w64-ucrt-x86_64-python-matplotlib \
  mingw-w64-ucrt-x86_64-python-shapely \
  mingw-w64-ucrt-x86_64-python-tifffile \
  mingw-w64-ucrt-x86_64-python-openpyxl \
  mingw-w64-ucrt-x86_64-python-pillow \
  mingw-w64-ucrt-x86_64-python-imageio

python --version   # must print Python 3.14.x
```

#### Step W3 — Add MSYS2 UCRT64 bin to the Windows System PATH

`fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd` is a GCC-compiled C extension
that loads GCC runtime DLLs at import time. These DLLs live in
`C:\msys64\ucrt64\bin`. If that directory is not on PATH, Python raises
`ImportError: DLL load failed` when ctfire_py is imported.

**Add `C:\msys64\ucrt64\bin` to the Windows System PATH (one time only):**

1. Open **Settings → System → About → Advanced system settings**.
2. Click **Environment Variables**.
3. Under **System variables**, select **Path** → **Edit**.
4. Click **New** and paste: `C:\msys64\ucrt64\bin`
5. Click OK on all dialogs, then **reopen** the UCRT64 shell.

```bash
echo $PATH | tr ':' '\n' | grep ucrt64
# should show /c/msys64/ucrt64/bin
```

#### Step W4 — Get the tme-quant repository

```bash
git clone <repo-url> /h/my-repos/tme-quant
cd /h/my-repos/tme-quant
git checkout prototype/hierarchy-model-for-CApy

REPO=/h/my-repos/tme-quant   # adjust to your actual path
```

**Or copy from another machine.** The pre-built `fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd`
is already in the repo, so no compilation is needed.

#### Step W5 — Create the venv

Windows uses `--system-site-packages` because the C-extension packages (numpy, scipy,
opencv, etc.) were installed into the MSYS2 system Python via `pacman`, not into pip.

```bash
cd "$REPO/src/tme_quant"
python -m venv --system-site-packages venv_fire_only
source venv_fire_only/bin/activate
```

Then continue at [Create the environment and add path files](#create-the-environment-and-add-path-files).

---

### Linux (x86_64)

**Python required: 3.10**

The pre-built `fiber_backend.cpython-310-x86_64-linux-gnu.so` requires Python 3.10
exactly. Conda (Miniforge/Miniconda) is the recommended way to get it cleanly.

#### Step L1 — Install Conda (if not already present)

```bash
# Miniforge (recommended — conda-forge by default):
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Or use an existing Conda installation (Miniconda, Anaconda).

> **Alternative — system Python 3.10:**
> If your distro ships Python 3.10 (`apt install python3.10 python3.10-venv`),
> you can skip Conda and use a plain venv; install packages with
> `pip install numpy scipy scikit-image opencv-python pandas matplotlib shapely tifffile openpyxl pillow imageio`.
> In that case, replace `conda activate fire_only_env` with
> `source venv_fire_only/bin/activate` throughout.

#### Step L2 — Create the Conda environment

```bash
conda create -n fire_only_env python=3.10 \
    numpy scipy scikit-image pandas matplotlib \
    shapely tifffile openpyxl pillow imageio
conda install -n fire_only_env -c conda-forge opencv
conda activate fire_only_env

python --version   # must print Python 3.10.x
```

> No `libgomp1` step is needed — `numpy` and `scipy` from conda-forge pull in
> the OpenMP runtime automatically. If you see an `libgomp` error, run
> `conda install -c conda-forge libgomp`.

#### Step L3 — Get the tme-quant repository

```bash
git clone <repo-url> ~/repos/tme-quant
cd ~/repos/tme-quant
git checkout prototype/hierarchy-model-for-CApy

REPO=~/repos/tme-quant   # adjust to your actual path
```

Then continue at [Create the environment and add path files](#create-the-environment-and-add-path-files).

---

### macOS (Apple Silicon / M-chip)

**Python required: 3.12**

The pre-built `fiber_backend.cpython-312-darwin.so` was compiled with OpenMP support
via `libomp` from Conda. **Conda is strongly recommended** — it provides both
Python 3.12 and `libomp` together, avoiding dylib loading errors.

#### Step M1 — Install Xcode command-line tools (if not already present)

```bash
xcode-select --install
```

#### Step M2 — Install Conda (if not already present)

```bash
# Miniforge for Apple Silicon (ARM64):
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o Miniforge3.sh
bash Miniforge3.sh
```

#### Step M3 — Create the Conda environment

```bash
conda create -n fire_only_env python=3.12 \
    numpy scipy scikit-image pandas matplotlib \
    shapely tifffile openpyxl pillow imageio libomp
conda install -n fire_only_env -c conda-forge opencv
conda activate fire_only_env

python --version   # must print Python 3.12.x
```

> `libomp` is required explicitly because `fiber_backend.cpython-312-darwin.so`
> links against `libomp.dylib`. Without it Python raises
> `ImportError: Library not loaded: @rpath/libomp.dylib` at import time.

#### Step M4 — Get the tme-quant repository

```bash
git clone <repo-url> ~/repos/tme-quant
cd ~/repos/tme-quant
git checkout prototype/hierarchy-model-for-CApy

REPO=~/repos/tme-quant   # adjust to your actual path
```

Then continue at [Create the environment and add path files](#create-the-environment-and-add-path-files).

---

## Create the environment and add path files

These steps are **identical on all platforms.** Run them in the activated environment
created in the platform section above (Windows venv, or Linux/macOS Conda env).

```bash
cd "$REPO/src/tme_quant"

# pth 1 — real tme_quant package at src/tme_quant/src/
# aa_ prefix: loaded first (alphabetical), so Python finds the real package
# before it can pick up the src/tme_quant/ project directory as a namespace package.
python -c "
import sysconfig, pathlib, os
site = sysconfig.get_paths()['purelib']
pkg_src = str(pathlib.Path(os.getcwd()) / 'src')
(pathlib.Path(site) / 'aa_tme_quant_pkg.pth').write_text(pkg_src + '\n')
print('aa_tme_quant_pkg.pth ->', pkg_src)
"

# pth 2 — repo-level src/ for ctfire_py and pycurvelets
python -c "
import sysconfig, pathlib, os
site = sysconfig.get_paths()['purelib']
repo_src = str(pathlib.Path(os.getcwd()).parent.resolve())
(pathlib.Path(site) / 'tme_quant_src.pth').write_text(repo_src + '\n')
print('tme_quant_src.pth ->', repo_src)
"
```

---

## Verify the installation

```bash
python -c "
import tme_quant, ctfire_py, pycurvelets
from ctfire_py.fire_2d_angle import fire_2d_angle
from tme_quant.tme_analysis.pipelines import curvealign_ctfire_mode_pipeline
print('tme_quant:', tme_quant.__file__)
print('ctfire_py:', ctfire_py.__file__)
print('All imports OK — fire-only pipeline is ready.')
"
```

Expected output (paths will reflect your repo location):
```
tme_quant: /your/path/tme-quant/src/tme_quant/src/tme_quant/__init__.py
ctfire_py: /your/path/tme-quant/src/ctfire_py/__init__.py
All imports OK — fire-only pipeline is ready.
```

If `tme_quant.__file__` is `None` (namespace package), the `aa_tme_quant_pkg.pth`
file was not created or was processed after `tme_quant_src.pth`. Re-run the pth
block above and verify both `.pth` files exist in the environment's `site-packages/`.

---

## Run the fire-only example

Place the test images at:
```
$REPO/tests/test_images/real1.tif
$REPO/tests/test_images/CA_Boundary/mask_real1.tiff
```

Then run:
```bash
cd "$REPO/src/tme_quant"
python examples/example_curvealign_ctfire_pipeline.py --scenario fire-only
```

Expected output:
- Fiber count printed to stdout
- TACS breakdown (TACS-1/2/3 percentages)
- Overlay PNG + heatmap PNG + xlsx saved to `examples/output/`
- Interactive matplotlib TACS viewer window (requires a display; see Troubleshooting for headless)

---

## Activating the environment in future sessions

**Windows:**
```bash
cd "$REPO/src/tme_quant"
source venv_fire_only/bin/activate
```

**Linux / macOS:**
```bash
conda activate fire_only_env
cd "$REPO/src/tme_quant"
```

---

## Alternative: flat-layout pip install

If you prefer a single `pip install` command over writing `.pth` files manually,
you can copy `ctfire_py` and `pycurvelets` into the project's own `src/` directory
and use the provided `pyproject_flat.toml` instead.

### Which option to choose

| | `.pth` file approach | Flat-layout approach |
|---|---|---|
| **Recommended for** | Developers; keeps ctfire_py as a clearly separate package | End-user deployment; simplest install command |
| **ctfire_py location** | `src/ctfire_py/` (repo-level, separate) | `src/tme_quant/src/ctfire_py/` (copied in) |
| **Install command** | No pip install needed | `python -m pip install -e .` |
| **Syncing ctfire_py updates** | Replace folder, nothing else changes | Must re-copy folder each time |
| **Risk of tme_quant tree pollution** | None | Low, but copies live inside the project src |

> **Developer note:** do not push a repo where `ctfire_py` or `pycurvelets` have
> been copied into `src/tme_quant/src/`. That layout is for local deployment only.
> The canonical development layout keeps them as separate sibling packages.

### Steps

**1. Copy the two packages into the project src:**

```bash
cd "$REPO"
cp -r src/ctfire_py  src/tme_quant/src/ctfire_py
cp -r src/pycurvelets src/tme_quant/src/pycurvelets
```

Verify:
```bash
ls src/tme_quant/src/
# should show: tme_quant/  ctfire_py/  pycurvelets/
```

**2. Swap in the flat-layout pyproject.toml:**

```bash
cd "$REPO/src/tme_quant"
cp pyproject.toml pyproject.toml.bak     # keep the original safe
cp pyproject_flat.toml pyproject.toml
```

**3. Create the environment and install:**

```bash
# Activate your environment first (conda activate or source venv/bin/activate)

# Use standard pip — NOT uv pip install.
# uv pip install always targets the project's own .venv, ignoring your active env.
python -m pip install -e .
```

> **If you prefer uv:** run `uv pip install -e .` instead. It will install into
> `.venv` (not your named environment). Activate `.venv` with
> `source .venv/bin/activate` (Linux/macOS) or `source .venv/Scripts/activate` (Windows)
> in subsequent sessions.

Verify with the same command from the Verify section above.

---

## Troubleshooting

| Error | Platform | Cause | Fix |
|-------|----------|-------|-----|
| `ImportError: DLL load failed while importing fiber_backend` | Windows | MSYS2 DLLs not on Windows PATH | Add `C:\msys64\ucrt64\bin` to System PATH (Step W3) and reopen shell |
| `ImportError: Module use of python314.dll conflicts` | Windows | Python version mismatch | Must use Python 3.14 (cp314); check `python --version` |
| `ImportError: Library not loaded: @rpath/libomp.dylib` | macOS | libomp not installed in Conda env | `conda install -c conda-forge libomp` |
| `ImportError: No module named 'fiber_backend'` | Linux/macOS | Wrong Python version activated | Linux needs 3.10, macOS needs 3.12; check `python --version` |
| `ModuleNotFoundError: No module named 'ctfire_py'` | All | `tme_quant_src.pth` not created | Re-run the pth 2 block; verify `site-packages/tme_quant_src.pth` exists |
| `ModuleNotFoundError: No module named 'pycurvelets'` | All | Same as above | Re-run the pth 2 block |
| `tme_quant.__file__ is None` (namespace package) | All | `aa_tme_quant_pkg.pth` missing or wrong | Re-run the pth 1 block; verify `site-packages/aa_tme_quant_pkg.pth` exists |
| `ModuleNotFoundError: No module named 'cv2'` | Windows | opencv not installed | `pacman -S mingw-w64-ucrt-x86_64-python-opencv` |
| `ModuleNotFoundError: No module named 'cv2'` | Linux/macOS | opencv not in Conda env | `conda install -c conda-forge opencv` |
| `RuntimeWarning: FIRE ran out of memory (std::bad_alloc)` | All | Too many FIRE seeds | Raise `thresh_LMPdist` (e.g. 12) in `fire_only_params`; see comments in `scenario_fire_only()` |
| `SKIPPED — image not found` | All | Test images missing | Copy `real1.tif` and `mask_real1.tiff` to the paths in Run section |
| `_tkinter.TclError: no display name` | All | Headless / SSH session | Add `matplotlib.use("Agg")` before `import matplotlib.pyplot` in the example; output will be file-only |
| `ModuleNotFoundError: ctfire_py` (flat layout) | All | Folders not copied or used `uv pip` | Verify `src/tme_quant/src/ctfire_py/` exists; use `python -m pip install -e .` (not `uv pip`) |
| `ModuleNotFoundError: tme_quant.tme_analysis` (flat layout) | All | Used `uv pip install` which installed to `.venv` | Activate `.venv` instead, or use `python -m pip install -e .` |

---

## What is NOT installed

The following are intentionally absent from `fire_only_env` / `venv_fire_only`:

| Package | Why omitted |
|---------|-------------|
| `curvelops` | Only needed for `use_ct_reconstruction=True` (CT-FIRE mode) |
| `pyimagej` / `scyjava` | Fiji/ImageJ bridge — not used by FIRE pipeline |
| `torch` / `stardist` / `cellpose` | Deep learning cell segmentation — not used |
| `napari` / `Qt` | GUI plugin — not needed for script-based use |

To add curvelops later and run the full CT-FIRE mode, see
`CLAUDE.md § "Running napari with curvelops"` and `.venv-curvelops`.

---

## Syncing ctfire_py with upstream changes

`src/ctfire_py/` is tracked in this repo and synced from the `32-convert-ctfire`
branch as needed (see `doc/DEVELOPMENT.md § Synchronizing ctfire_py`).
After a sync commit, the `.pth` files written above persist and do not need
to be recreated. If you used the flat-layout alternative, re-copy `src/ctfire_py/`
into `src/tme_quant/src/ctfire_py/` and re-run `python -m pip install -e .`.
