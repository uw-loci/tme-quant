"""FDCT (CurveLab wrapping) curvelet backend for pycurvelets.

The backend class is :class:`pycurvelets_fdct.backend.FDCTBackend`. It is
registered as ``fdct`` in the ``pycurvelets.backends`` entry-point group, so
``pycurvelets`` loads it by name. Importing this package does not import
``curvelops``; importing ``pycurvelets_fdct.backend`` does.
"""
