# pycurvelets-core

Curvelet fiber quantification, SHG–HE registration, and tumor annotation for [CurveAlign](https://loci.wisc.edu/software/curvealign/). Import this project as `pycurvelets`.

This project does not include a curvelet transform. Install one backend next to it:

- [pycurvelets-fdct](../pycurvelets-fdct/README.md) — CurveLab, the transform the MATLAB references use. Needs a local CurveLab and FFTW build.
- [pycurvelets-udct](../pycurvelets-udct/README.md) — uniform discrete curvelet transform. Pure Python.

Without a backend, registration, annotation, segmentation and preprocessing still run. `new_curv`, `get_ct` and `process_image` raise `BackendUnavailableError` and name the package to install.

```bash
pip install -e ./pycurvelets-core
```

From another repository:

```bash
pip install "pycurvelets-core @ git+https://github.com/uw-loci/tme-quant.git#subdirectory=pycurvelets-core"
```

Optional extras: `segmentation` (Cellpose and StarDist) and `fiji` (Appose, used by the Fiji plugin).

Boundary-relative fiber angles in the analysis pipeline come from `pycurvelets.get_tif_boundary.get_relative_angle`. The napari plugin's TACS histogram lives in [napari-curvealign](https://github.com/uw-loci/napari-curvealign), not in this package.

Repository install and development notes: [../doc/INSTALL.md](../doc/INSTALL.md), [../doc/DEVELOPMENT.md](../doc/DEVELOPMENT.md).
