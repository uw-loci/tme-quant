# pycurvelets-fdct

FDCT backend for [pycurvelets-core](../pycurvelets-core/README.md). It wraps CurveLab's 2D wrapping transform through `curvelops` and registers itself as `fdct` in the `pycurvelets.backends` entry-point group.

This is the transform MATLAB CurveAlign uses. The core tests that compare against those references pin `backend="fdct"`.

Building it needs FFTW 2.1.5 and CurveLab on the machine. `FFTW` and `FDCT` must point at those installs. From a tme-quant checkout, `bash bin/install.sh` does that and installs this project. See [../doc/INSTALL.md](../doc/INSTALL.md).

```bash
pip install -e ./pycurvelets-core -e ./pycurvelets-fdct
```

`fdct` is the default when more than one backend is installed. Pick it explicitly with `backend="fdct"` or `TMEQ_CURVELET_BACKEND=fdct`.
