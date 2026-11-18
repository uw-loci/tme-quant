### tme-quant

Python translation of [CurveAlign](https://loci.wisc.edu/software/curvealign/) with the goal of unifying cell and collagen analysis, provided as a modern Python `src/` package.

Interactive front ends live in their own repositories and call this library:
- Fiji plugin: [uw-loci/fiji-curvealign](https://github.com/uw-loci/fiji-curvealign)
- napari plugin: [uw-loci/napari-curvealign](https://github.com/uw-loci/napari-curvealign)

### About this repo
- `pycurvelets-core/src/pycurvelets`: Python implementation using the curvelet transform, including the SHG–HE registration port (`SHG_HE_registration`), tumor annotation, and the segmentation, preprocessing and TACS helpers the plugins drive
- `pycurvelets-core/tests/`: pytest suite (data-driven tests)
- `.github/workflows/ci.yml`: GitHub Actions workflow (runs core tests only)

### Wheel vs sdist
Two installable views. Both are produced from a clone (`make wheel` /
`make sdist`, or `python -m build`).

| View | How to get it | What it contains |
| --- | --- | --- |
| **Wheel** | `pip install tme-quant` or `make wheel` | Runtime only: `src/` (`pycurvelets`, `data/*.npz`). |
| **sdist** | `pip install tme-quant --no-binary tme-quant` or `make sdist` | Source to rebuild the wheel, plus docs and the lightweight pytest suite. |

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

### Quick start (without curvelets)

```bash
uv sync
uv run pytest -q
```

### Optional: curvelet backend (curvelops)

To enable curvelet-powered features and tests you must build FFTW 2.1.5 and CurveLab, then install with curvelops. Use the automated script:

**Prerequisites:** Check out this repo, install [uv](https://docs.astral.sh/uv/), and extract CurveLab into this repo's `utils/` directory (see [doc/INSTALL.md](doc/INSTALL.md)).

macOS/Linux:
```bash
bash bin/install.sh
# or: make setup
```

Windows options:
- Recommended: use WSL2 (Ubuntu). Follow the macOS/Linux steps inside WSL.
- Native Windows is not verified end to end: the `curvelops` Python extension requires MSVC (`cl.exe`), while FFTW and CurveLab are built with GNU tools. MSVC-compatible FFTW and CurveLab libraries are also needed. Use WSL2 for the supported install path.

### Development

See [doc/DEVELOPMENT.md](doc/DEVELOPMENT.md) for setup and troubleshooting. Running tests:

- Default test suite:
```bash
make test
```

- Enable curvelet-dependent tests:
```bash
export TMEQ_RUN_CURVELETS=1
uv run pytest -q pycurvelets-core/tests/test_get_ct.py pycurvelets-core/tests/test_new_curv.py pycurvelets-core/tests/test_process_image.py -rs
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
- CI has two lanes in `.github/workflows/ci.yml`:
  - `test-basic`: default Python matrix without CurveLab secret requirements.
  - `test-curvelab`: secure lane that fetches/builds CurveLab + FFTW and runs curvelet-enabled tests.

### Working with secrets in GitHub Actions
This project uses GitHub Actions secrets for tasks that require authentication or access to private resources. Secrets can be configured in the [Settings tab of the repostiory](https://github.com/uw-loci/tme-quant/settings/secrets/actions). For more information about secrets, see:
- [Secrets as a concept](https://docs.github.com/en/actions/concepts/security/secrets)
- [Secrets in actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions).

Never hardcode secrets or access tokens in workflow files; always use the `secrets:` context.

### Troubleshooting
- curvelops build errors: ensure `FFTW` and `FDCT` point to your install roots and the 2D/3D libraries were built.
