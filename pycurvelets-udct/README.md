# pycurvelets-udct

UDCT backend for [pycurvelets-core](../pycurvelets-core/README.md). It wraps `curvelets.numpy.UDCT` from the [curvelets](https://github.com/cako/curvelets) package and registers itself as `udct` in the `pycurvelets.backends` entry-point group.

Pure Python. No CurveLab or FFTW build.

```bash
pip install -e ./pycurvelets-core -e ./pycurvelets-udct
```

From another repository, install both subdirectories of [tme-quant](https://github.com/uw-loci/tme-quant) at the same commit. This project pins `pycurvelets-core` exactly, because the backend protocol is versioned with core.

Select it with `backend="udct"` or `TMEQ_CURVELET_BACKEND=udct`. When `fdct` is also installed, `fdct` stays the default.

`curvelets` 1.2 imports `typing_extensions` on every Python version but only declares that dependency for Python 3.10 and older, so this project depends on `typing-extensions` directly.
