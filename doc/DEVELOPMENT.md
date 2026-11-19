# Development

Use the same install path as users: `bash bin/install.sh` (or `make setup`).

## Plugins

The interactive front ends are developed in their own repositories and depend
on this library:
- Fiji: [uw-loci/fiji-curvealign](https://github.com/uw-loci/fiji-curvealign)
- napari: [uw-loci/napari-curvealign](https://github.com/uw-loci/napari-curvealign)

If you only need the segmentation features (Cellpose/StarDist) without curvelets,
from the repository root:

```bash
uv venv --python 3.11
uv pip install -e "./pycurvelets-core[segmentation]" pytest
```

## Projects and curvelet backends

Each folder at the root with a `pyproject.toml` is a separately built project;
there is no root project. Run `uv` and `make` from the repository root so they
use the shared `.venv` (inside a project folder, `uv run` builds a per-project env).

`pycurvelets-core` never imports a transform library. `pycurvelets.backends`
loads transforms from the `pycurvelets.backends` entry-point group, which
backend projects register in their `pyproject.toml`:

```toml
[project.entry-points."pycurvelets.backends"]
fdct = "pycurvelets_fdct.backend:FDCTBackend"
```

A backend implements `pycurvelets.backends.CurveletBackend` (`forward`,
`scale_geometry`, `inverse`) and its tests call
`pycurvelets.backends.conformance.check_backend`, which also pins the angle
and pixel-position conventions `new_curv` relies on.
`pycurvelets-core/tests/test_backend_boundaries.py` fails if core imports a
provider directly. Backend projects pin `pycurvelets-core` exactly, because
the protocol is versioned with core.

Choosing a backend: `CurveletControlParameters(backend=...)` /
`FiberAnalysisParameters(backend=...)`, else `TMEQ_CURVELET_BACKEND`, else the
only one installed, else `fdct`. The choice is logged at INFO level.

## Running tests

```bash
make test
```

Curvelet tests run when `TMEQ_RUN_CURVELETS=1` and `pycurvelets-fdct` is installed; otherwise they are skipped.

### SHG–HE registration (BDcreation_reg / BDcreation_reg2)

| Suite | When it runs | Needs |
| --- | --- | --- |
| `pycurvelets-core/tests/test_he_bdc_reg1.py` | CI | nothing extra |
| `pycurvelets-core/tests/test_shg_he_registration.py` | CI | patient fixtures in the clone (not in wheel/sdist) |
| `pycurvelets-core/tests/test_shg_he_registration_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `archive/matlab_parity/dumps` (git only) |
| `pycurvelets-core/tests/test_shg_he_registration_gt.py` | `TMEQ_RUN_MATLAB_PARITY=1` | dumps + patient_02 GT HE TIFFs (git only) |

See [MATLAB_PARITY.md](MATLAB_PARITY.md), [BDC_REGISTRATION_COMPARISONS.md](BDC_REGISTRATION_COMPARISONS.md), [BDC_ANNOTATION_COMPARISONS.md](BDC_ANNOTATION_COMPARISONS.md), and [ARTIFACTS.md](ARTIFACTS.md).

### Tumor annotation (BDcreationHE / BDcreationHE2)

| Suite | When it runs | Needs |
| --- | --- | --- |
| `pycurvelets-core/tests/test_he_bdc_annotation.py` | CI | nothing extra |
| `pycurvelets-core/tests/test_tumor_annotation_from_he.py` | CI | patient_001 HE fixtures + CA_Boundary goldens (clone) |
| `pycurvelets-core/tests/test_tumor_annotation_matlab_parity.py` | `TMEQ_RUN_MATLAB_PARITY=1` | `archive/matlab_parity/dumps/he2_*` and `he_*` (git only) |

Default `annotation_method="hsv"` is `BDcreationHE2.m`. `annotation_method="rgb_kmeans"` is `BDcreationHE.m` (unseeded MATLAB `kmeans`; Python replays `rng(28,'twister')`). MATLAB Image Processing Toolbox ports live in `_matlab_util.py`. Their names are the MATLAB operation (`rgb2gray`, `imresize`, `kmeans`); each docstring says the body was migrated from MATLAB, because those numerics are not what NumPy or scikit-image return for the same name. Pipeline steps stay in `_he_bdc_common.py`, `_he_bdc_reg1.py`, and `_he_bdc_annotation.py`. The ITK `imregtform` engine stays in `_itk_v3_matlab_engine.py`. Comparison figures are generated locally; see [ARTIFACTS.md](ARTIFACTS.md).

## Wheel vs sdist

| Command | Output |
| --- | --- |
| `make wheel` | `dist/*.whl`, one per project — each project's `src/` only (users) |
| `make sdist` | `dist/*.tar.gz`, one per project — source + lightweight tests (no tests 1-9 / dumps) |

`pycurvelets-core/tests/test_packaging.py` checks the contract.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, extract as `utils/CurveLab-<version>/` inside this repository |
| `curvelops` reports MSVC is required | Native Windows needs `cl.exe` and MSVC-compatible FFTW/CurveLab libraries; use WSL2 for the verified install path |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| `BackendUnavailableError` | No curvelet backend installed, or not the one requested; install `pycurvelets-fdct` (see `bin/install.sh`) |