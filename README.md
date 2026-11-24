### tme-quant

Python translation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) with the goal of unifying cell and collagen analysis.

Interactive front ends live in their own repositories and call this library:
- Fiji plugin: [uw-loci/fiji-curvealign](https://github.com/uw-loci/fiji-curvealign)
- napari plugin: [uw-loci/napari-curvealign](https://github.com/uw-loci/napari-curvealign)

### About this repo
The repository holds independently built Python projects, each with its own
`pyproject.toml`, `src/` and `tests/`. There is no project at the root.

| Project | Distribution | Import name | What it is |
| --- | --- | --- | --- |
| `pycurvelets-core/` | `pycurvelets-core` | `pycurvelets` | The analysis library: curvelet fiber extraction (`new_curv`, `get_ct`, `process_image`), the SHG–HE registration and tumor annotation ports, and the segmentation and preprocessing helpers. Has no curvelet transform of its own. |
| `pycurvelets-fdct/` | `pycurvelets-fdct` | `pycurvelets_fdct` | FDCT (CurveLab) backend through `curvelops`. Needs a local CurveLab + FFTW build. The MATLAB-referenced tests use this backend. |
| `pycurvelets-udct/` | `pycurvelets-udct` | `pycurvelets_udct` | UDCT (uniform discrete curvelet transform) backend through [`curvelets`](https://github.com/cako/curvelets). Pure Python, no native build. |

Core finds transforms through the `pycurvelets.backends` entry-point group.
With several backends installed it uses `fdct` unless you pick one with
`CurveletControlParameters(backend=...)`, `FiberAnalysisParameters(backend=...)`
or the `TMEQ_CURVELET_BACKEND` environment variable. New backends implement
the `pycurvelets.backends.CurveletBackend` protocol and must pass
`pycurvelets.backends.conformance.check_backend`.

`.github/workflows/ci.yml` tests core alone, core on FDCT, and builds every project.

### Wheel vs sdist
Each project builds its own wheel and sdist (`make wheel` / `make sdist`
for all of them, or `python -m build <project>`).

| View | How to get it | What it contains |
| --- | --- | --- |
| **Wheel** | `pip install pycurvelets-core` or `make wheel` | Runtime only: `pycurvelets-core/src/` (`pycurvelets`, `data/*.npz`). |
| **sdist** | `pip install pycurvelets-core --no-binary pycurvelets-core` or `make sdist` | Source to rebuild the wheel, plus the lightweight pytest suite. |

Registration tests 1-9, annotation tests 1-3, MATLAB dumps, and ground-truth
TIFFs live in git (clone / CI) only. They are not release artifacts.
Comparison PNGs are not stored; regenerate them with
`archive/visualization/` (see [doc/ARTIFACTS.md](doc/ARTIFACTS.md)).
`pycurvelets-core/tests/test_packaging.py` asserts this split. Bit-exact MATLAB parity needs
a clone plus `TMEQ_RUN_MATLAB_PARITY=1`; see [doc/MATLAB_PARITY.md](doc/MATLAB_PARITY.md).

### Licensing and prerequisites
This project depends on code that cannot be redistributed here:
- CurveLab (FDCT/FDCT3D) and FFTW 2.x are separately licensed. You must accept their licenses and build them locally if you want the optional curvelet backend (`curvelops`).

Base requirements:
- macOS, Linux, or Windows (see notes below)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick start (core only, no curvelet backend)

Run these from the repository root; `uv run` then uses the root `.venv`.
(Inside a project folder, `uv run` would build a separate per-project env.)

```bash
uv venv --python 3.11
uv pip install -e ./pycurvelets-core pytest
uv run pytest -q pycurvelets-core
```

Registration, annotation, segmentation and preprocessing work; curvelet
fiber analysis raises `BackendUnavailableError` naming the package to install.

For curvelet analysis without building CurveLab, add the UDCT backend:

```bash
uv pip install -e ./pycurvelets-udct
uv run pytest -q pycurvelets-udct
```

`bash bin/install.sh` (or `make setup`) does that install for you: core, with
the `segmentation` and `fiji` extras, and UDCT. CurveLab is optional.

UDCT results are close to FDCT's but not identical (different wedge layout),
so the MATLAB-referenced pipeline tests stay pinned to FDCT.

### Curvelet backend: FDCT (CurveLab via curvelops)

FDCT needs FFTW 2.1.5 and CurveLab, then `pycurvelets-fdct`, which builds curvelops. Extract CurveLab into this repo's `utils/` directory (see [doc/INSTALL.md](doc/INSTALL.md)) and run the same script. It detects that tree, builds FFTW and CurveLab, and installs `pycurvelets-fdct` along with core and UDCT.

**Prerequisites:** Check out this repo, install [uv](https://docs.astral.sh/uv/), and, for FDCT, extract CurveLab into `utils/CurveLab-<version>/`.

macOS/Linux:
```bash
bash bin/install.sh
# or: make setup
```

It creates `.venv` at the repository root and prints the curvelet backends it finds.

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows is not verified end to end: the `curvelops` Python extension requires MSVC (`cl.exe`), while FFTW and CurveLab are built with GNU tools. MSVC-compatible FFTW and CurveLab libraries are also needed. Use WSL2 for the supported install path.

### Development

See [doc/DEVELOPMENT.md](doc/DEVELOPMENT.md) for setup and troubleshooting. Running tests:

- Default test suite (every project):
```bash
make test
```

- Enable curvelet-dependent tests (needs `pycurvelets-fdct` installed):
```bash
export TMEQ_RUN_CURVELETS=1
uv run pytest -q pycurvelets-core/tests/test_get_ct.py pycurvelets-core/tests/test_new_curv.py pycurvelets-core/tests/test_process_image.py -rs
uv run pytest -q pycurvelets-fdct
```

- Enable strict MATLAB-reference parity assertions:
```bash
export TMEQ_VALIDATE_MATLAB=1
```

- SHG–HE registration (issue 33). CI always runs the cheap gates in
  `pycurvelets-core/tests/test_shg_he_registration.py` and `pycurvelets-core/tests/test_he_bdc_reg1.py`
  (fixtures are in the clone, not the wheel or sdist). The bit-exact dump
  suite (tests 1-9) is git-only; enable it with:
```bash
export TMEQ_RUN_MATLAB_PARITY=1
uv run pytest -q \
    pycurvelets-core/tests/test_shg_he_registration.py \
    pycurvelets-core/tests/test_he_bdc_reg1.py \
    pycurvelets-core/tests/test_shg_he_registration_matlab_parity.py \
    pycurvelets-core/tests/test_shg_he_registration_gt.py
```

- Tumor annotation (issue 34). CI always runs `pycurvelets-core/tests/test_he_bdc_annotation.py`
  and `pycurvelets-core/tests/test_tumor_annotation_from_he.py`. The bit-exact HE2/HE dump
  suite is git-only; enable it with the same `TMEQ_RUN_MATLAB_PARITY=1` flag:
```bash
export TMEQ_RUN_MATLAB_PARITY=1
uv run pytest -q \
    pycurvelets-core/tests/test_he_bdc_annotation.py \
    pycurvelets-core/tests/test_tumor_annotation_from_he.py \
    pycurvelets-core/tests/test_tumor_annotation_matlab_parity.py
```

Notes:
- MATLAB parity tests are opt-in because they validate exact numerical agreement with historical MATLAB reference artifacts.

Testing policy:
- Tests must not write files to the repository root. Use a system
  temporary directory instead (e.g., `tempfile.TemporaryDirectory`).
- If a deterministic artifact is needed across runs, commit it under
  `pycurvelets-core/tests/test_resources/` and read from there during tests.

### Continuous integration
- CI has five jobs in `.github/workflows/ci.yml`:
  - `lint`: `ruff check` on every project, using the root `ruff.toml`. Same command as `make check`.
  - `test-core`: `pycurvelets-core` alone on Python 3.11–3.13, with no curvelet backend installed.
  - `test-udct`: `pycurvelets-core` + `pycurvelets-udct` on Python 3.11–3.13; needs no secrets, so it also runs on forked PRs.
  - `test-fdct`: secure lane that fetches/builds CurveLab + FFTW, installs `pycurvelets-core` and `pycurvelets-fdct`, and runs both test suites with curvelets enabled. Skipped on forked PRs (no secrets).
  - `build`: builds the sdist and wheel of every project, runs `twine check`, and installs the core wheel alone to prove it imports without a backend.

### Working with secrets in GitHub Actions
This project uses GitHub Actions secrets for tasks that require authentication or access to private resources. Secrets can be configured in the [Settings tab of the repostiory](https://github.com/uw-loci/tme-quant/settings/secrets/actions). For more information about secrets, see:
- [Secrets as a concept](https://docs.github.com/en/actions/concepts/security/secrets)
- [Secrets in actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions).

Never hardcode secrets or access tokens in workflow files; always use the `secrets:` context.

### Troubleshooting
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
- `BackendUnavailableError`: no curvelet backend is installed, or the one you asked for is missing; the message lists the installed backends and what to install. `python -c "from pycurvelets.backends import available_backends; print(available_backends())"` shows what is registered.
- Installing the repository root (`pip install git+https://github.com/uw-loci/tme-quant.git`) no longer works; install a project subdirectory instead, e.g. `pip install "pycurvelets-core @ git+https://github.com/uw-loci/tme-quant.git#subdirectory=pycurvelets-core"`.
