# Setting up the CT-FIRE FIRE-only pipeline on Windows (MSYS2 UCRT64)

This guide walks through setting up the **FIRE-only** CT-FIRE pipeline
(`use_ct_reconstruction=False`) on a fresh Windows x64 machine.

**What you get:** individual fiber extraction with full TACS classification and
the interactive matplotlib viewer — **without** installing curvelops or any
curvelet transform library.

**Target platform:** Windows 10/11 x64, Python 3.14 via MSYS2 UCRT64.
The pre-built `fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd` extension
included in `src/ctfire_py/` is compiled for exactly this combination.

---

## Directory layout and why we don't use `pip install`

Understanding the two-level `src/` structure explains why a normal
`pip install -e .` cannot set up this environment on its own.

```
tme-quant/                          ← git root (clone/copy goes here)
└── src/                            ← repo-level src: holds all packages as siblings
    ├── ctfire_py/                  ← ctfire_py package (includes fiber_backend.pyd)
    ├── pycurvelets/                ← pycurvelets utilities (required by ctfire_py)
    └── tme_quant/                  ← project root (pyproject.toml lives here)
        └── src/
            └── tme_quant/         ← actual tme_quant Python package
```

There are two distinct problems with using `pip install -e .` here:

**Problem 1 — uv ignores the activated venv.**
When `uv pip install -e .` is run from inside `src/tme_quant/` (which contains
`pyproject.toml`), uv detects the project and installs into the project's own
configured virtual environment (`.venv`) regardless of which venv is currently
activated. A separately created `venv_fire_only` is silently bypassed.

**Problem 2 — ctfire_py and pycurvelets are outside the package discovery path.**
`pyproject.toml` uses `where = ["src"]`, which resolves to `src/tme_quant/src/`
(the project's own source layout). `ctfire_py` and `pycurvelets` sit one level
above at `src/` (repo-level) — outside that path. Even if the install went into
the right venv, those two packages would still not be importable.

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

---

## Prerequisites at a glance

| Item | Version | Notes |
|------|---------|-------|
| Windows | 10 or 11 (x64) | ARM64 not supported by the pre-built extension |
| MSYS2 UCRT64 | Latest | Provides Python 3.14 + GCC runtime DLLs |
| Python | 3.14 (UCRT64) | **Must match** the `cp314` tag in `fiber_backend.pyd` |
| uv | Latest | pip-installable; used for fast package installs |
| tme-quant repo | `prototype/hierarchy-model-for-CApy` | Pre-built `fiber_backend.pyd` is included |

---

## Step 1 — Install MSYS2

1. Download the installer from **https://www.msys2.org**.
2. Run the installer. Accept the default path `C:\msys64`.
3. After installation, open the **MSYS2 UCRT64** shell:
   - Start menu → "MSYS2 UCRT64", **or**
   - Run `C:\msys64\ucrt64.exe`

> ⚠️ Always use the **UCRT64** shell, not MSYS, MinGW32, or MinGW64.
> The `fiber_backend` extension is linked against the UCRT64 runtime.

4. Update the package database:
   ```bash
   pacman -Syu
   # The shell may close; re-open UCRT64 and run again:
   pacman -Su
   ```

---

## Step 2 — Install Python 3.14 and C-extension dependencies

Using `pacman` for C-extension packages avoids build failures under GCC 15
(the same approach used for `.venv-curvelops` on this development machine).

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
```

Verify the Python version is 3.14:
```bash
python --version   # should print Python 3.14.x
```

---

## Step 3 — Add MSYS2 UCRT64 bin to the Windows System PATH

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

Verify the DLL directory is visible:
```bash
echo $PATH | tr ':' '\n' | grep ucrt64
# should show /c/msys64/ucrt64/bin
```

---

## Step 4 — Install uv

```bash
pip install uv
uv --version   # verify
```

---

## Step 5 — Get the tme-quant repository

**Option A — git clone** (if the repo is hosted on GitHub/GitLab):
```bash
git clone <repo-url> /h/my-repos/tme-quant
cd /h/my-repos/tme-quant
git checkout prototype/hierarchy-model-for-CApy
```

**Option B — copy from another machine:**
Copy the entire `tme-quant/` folder. The pre-built
`src/ctfire_py/fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd` is already
in the repository, so no compilation is needed.

From this point, all commands assume the repo root is at some path — adjust
to match your actual location. The variable `REPO` is used below to keep
the commands copy-pasteable:

```bash
REPO=/h/GitHub.06.2022/tme-quant   # change this to your actual path
```

---

## Step 6 — Create `venv_fire_only` and add path files

See "Directory layout and why we don't use `pip install`" above for the full
explanation. The short version: `uv pip install -e .` in this project always
installs into `.venv` (not into a separately created venv), and `ctfire_py`/
`pycurvelets` are outside the pyproject.toml package discovery path regardless.
Two `.pth` files written directly into site-packages solve both problems.

```bash
cd "$REPO/src/tme_quant"

# Create virtual environment that reuses pacman-installed C-extension packages
# (numpy, scipy, opencv, etc.) to avoid recompilation under GCC 15.
python -m venv --system-site-packages venv_fire_only
source venv_fire_only/bin/activate

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

## Alternative: flat-layout pip install

If you prefer a single `pip install` command over writing `.pth` files manually,
you can copy `ctfire_py` and `pycurvelets` into the project's own `src/` directory
and use the provided `pyproject_flat.toml` instead.

### Which option to choose

| | `.pth` file approach (Step 6) | Flat-layout approach (this section) |
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

**3. Create the venv and install:**

```bash
python -m venv --system-site-packages venv_fire_only
source venv_fire_only/bin/activate

# Use standard pip — NOT uv pip install.
# uv pip install always targets the project's own .venv, ignoring venv_fire_only.
python -m pip install -e .
```

> **If you prefer uv:** run `uv pip install -e .` instead. It will install into
> `.venv` (not `venv_fire_only`). Activate `.venv` with
> `source .venv/bin/activate` in subsequent sessions.

Verify with the same command from Step 7:
```bash
python -c "
import tme_quant, ctfire_py, pycurvelets
print('tme_quant:', tme_quant.__file__)
print('ctfire_py:', ctfire_py.__file__)
print('All imports OK.')
"
```

---

## Step 7 — Verify the installation

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
tme_quant: H:/your/path/tme-quant/src/tme_quant/src/tme_quant/__init__.py
ctfire_py: H:/your/path/tme-quant/src/ctfire_py/__init__.py
All imports OK — fire-only pipeline is ready.
```

If `tme_quant.__file__` is `None` (namespace package), the `aa_tme_quant_pkg.pth`
file was not created or was processed after `tme_quant_src.pth`. Re-run Step 6 and
verify both `.pth` files exist in the venv's `site-packages/`.

---

## Step 8 — Run the fire-only example

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
- Interactive matplotlib TACS viewer window

---

## Activating the environment in future sessions

```bash
cd "$REPO/src/tme_quant"
source venv_fire_only/bin/activate
python examples/example_curvealign_ctfire_pipeline.py --scenario fire-only
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: DLL load failed while importing fiber_backend` | MSYS2 DLLs not on Windows PATH | Add `C:\msys64\ucrt64\bin` to System PATH (Step 3) and reopen shell |
| `ImportError: Module use of python314.dll conflicts` | Python version mismatch | Must use Python 3.14 (cp314); check `python --version` |
| `ModuleNotFoundError: No module named 'ctfire_py'` | `tme_quant_src.pth` not created | Re-run the pth 2 block in Step 6; verify `site-packages/tme_quant_src.pth` exists |
| `ModuleNotFoundError: No module named 'pycurvelets'` | Same as above | Re-run the pth 2 block in Step 6 |
| `tme_quant.__file__ is None` (namespace package) | `aa_tme_quant_pkg.pth` missing or wrong | Re-run the pth 1 block in Step 6; verify `site-packages/aa_tme_quant_pkg.pth` exists |
| `ModuleNotFoundError: No module named 'cv2'` | opencv not installed | `pacman -S mingw-w64-ucrt-x86_64-python-opencv` |
| `RuntimeWarning: FIRE ran out of memory (std::bad_alloc)` | Too many FIRE seeds | Raise `thresh_LMPdist` (e.g. 12) in `fire_only_params`; see comments in `scenario_fire_only()` |
| `SKIPPED — image not found` | Test images missing | Copy `real1.tif` and `mask_real1.tiff` to the paths in Step 8 |
| `_tkinter.TclError: no display name` | Headless / SSH session | Add `matplotlib.use("Agg")` before `import matplotlib.pyplot` in the example; output will be file-only |
| `ModuleNotFoundError: ctfire_py` (flat layout) | Folders not copied or used `uv pip` | Verify `src/tme_quant/src/ctfire_py/` exists; use `python -m pip install -e .` (not `uv pip`) |
| `ModuleNotFoundError: tme_quant.tme_analysis` (flat layout) | Used `uv pip install` which installed to `.venv` | Run `source .venv/bin/activate` then retry, or use `python -m pip install -e .` into `venv_fire_only` |

---

## What is NOT installed

The following are intentionally absent from `venv_fire_only`:

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
After a sync commit, the `.pth` files written in Step 6 persist and do not need
to be recreated. If you used the flat-layout alternative, re-copy `src/ctfire_py/`
into `src/tme_quant/src/ctfire_py/` and re-run `python -m pip install -e .`.
