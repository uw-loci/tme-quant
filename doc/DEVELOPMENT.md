# Development

Use the same install path as users: `bash bin/install.sh` (or `make setup`).

## Plugins

The interactive front ends are developed in their own repositories and depend
on this library:
- Fiji: [uw-loci/fiji-curvealign](https://github.com/uw-loci/fiji-curvealign)
- napari: [uw-loci/napari-curvealign](https://github.com/uw-loci/napari-curvealign)

If you only need the segmentation features (Cellpose/StarDist) without curvelets:

```bash
uv sync --extra segmentation
```

## Running tests

```bash
make test
```

Curvelet tests run automatically when curvelops is installed; otherwise they are skipped.

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
| `make wheel` | `dist/*.whl` — `src/` only (users) |
| `make sdist` | `dist/*.tar.gz` — source + docs + lightweight tests (no tests 1-9 / dumps) |

`pycurvelets-core/tests/test_packaging.py` checks the contract.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, extract as `utils/CurveLab-<version>/` inside this repository |
| `curvelops` reports MSVC is required | Native Windows needs `cl.exe` and MSVC-compatible FFTW/CurveLab libraries; use WSL2 for the verified install path |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`