"""Checks every curvelet backend must pass.

Backend packages call :func:`check_backend` from their own test suites, so the
contract lives in one place::

    from pycurvelets.backends.conformance import check_backend
    check_backend(MyBackend())

Each failed check raises ``AssertionError`` naming the backend, the shape and
the scale involved.
"""

from __future__ import annotations

import numpy as np

from .protocol import CurveletBackend, ScaleGeometry

# Relative reconstruction error allowed for inverse(forward(img)).
ROUNDTRIP_RTOL = 1e-6


def _test_image(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    stripes = np.sin(2 * np.pi * (rows * 0.6 + cols * 0.8) / 9.0)
    return (100 + 50 * stripes + 10 * rng.standard_normal(shape)).astype(np.float64)


def _check_scale(backend, coefficients, shape, scale) -> None:
    where = f"{backend.name} backend, shape {shape}, scale {scale}"
    geometry = backend.scale_geometry(shape, scale)
    assert isinstance(geometry, ScaleGeometry), f"{where}: scale_geometry must return ScaleGeometry"
    assert np.isfinite(geometry.angle_step_deg) and geometry.angle_step_deg > 0, (
        f"{where}: angle_step_deg must be positive, got {geometry.angle_step_deg}"
    )
    assert 0 < len(geometry.wedges) <= len(coefficients[scale]), (
        f"{where}: {len(geometry.wedges)} wedges described, "
        f"forward returned {len(coefficients[scale])}"
    )
    for k, wedge in enumerate(geometry.wedges):
        n_coeffs = np.asarray(coefficients[scale][k]).size
        assert np.isfinite(wedge.angle_deg), f"{where}, wedge {k}: angle is not finite"
        for label, values in (("rows", wedge.rows), ("cols", wedge.cols)):
            values = np.asarray(values)
            assert values.ndim == 1 and values.size == n_coeffs, (
                f"{where}, wedge {k}: {label} has shape {values.shape}, "
                f"coefficient array has {n_coeffs} entries"
            )
            assert np.isfinite(values).all(), f"{where}, wedge {k}: {label} not finite"


def check_backend(backend: CurveletBackend, shape: tuple[int, int] = (256, 256), seed: int = 0) -> None:
    """Assert that ``backend`` honours the ``CurveletBackend`` contract.

    Parameters
    ----------
    backend : CurveletBackend
        Instance to check.
    shape : tuple of int
        Image shape to test with.
    seed : int
        Seed of the synthetic test image.

    Raises
    ------
    AssertionError
        On the first violated requirement.
    """
    assert isinstance(backend, CurveletBackend), (
        f"{type(backend).__name__} does not implement CurveletBackend"
    )
    assert isinstance(backend.name, str) and backend.name, "backend.name must be a non-empty string"

    img = _test_image(shape, seed)
    coefficients = backend.forward(img)
    assert isinstance(coefficients, list) and len(coefficients) >= 3, (
        f"{backend.name}: forward must return at least 3 scales, got {type(coefficients).__name__} "
        f"of length {len(coefficients) if hasattr(coefficients, '__len__') else '?'}"
    )
    for s, wedges in enumerate(coefficients):
        assert isinstance(wedges, list) and wedges, f"{backend.name}: scale {s} must be a non-empty list"
        for k, c in enumerate(wedges):
            assert isinstance(c, np.ndarray), f"{backend.name}: coefficients[{s}][{k}] is not an ndarray"

    # new_curv analyses scale index len - 1 - scale for scale = 1 (default) and 2.
    for analysed in (len(coefficients) - 2, len(coefficients) - 3):
        _check_scale(backend, coefficients, shape, analysed)

    reconstructed = np.real(backend.inverse(coefficients, shape))
    assert reconstructed.shape == shape, f"{backend.name}: inverse returned shape {reconstructed.shape}"
    error = np.max(np.abs(reconstructed - img)) / np.max(np.abs(img))
    assert error < ROUNDTRIP_RTOL, (
        f"{backend.name}: inverse(forward(img)) differs from img by {error:.2e} (relative)"
    )
