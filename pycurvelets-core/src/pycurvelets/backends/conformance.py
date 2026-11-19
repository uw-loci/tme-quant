"""Checks every curvelet backend must pass.

Backend packages call :func:`check_backend` from their own test suites, so the
contract lives in one place::

    from pycurvelets.backends.conformance import check_backend
    check_backend(MyBackend())

Each failed check raises ``AssertionError`` naming the backend, the shape and
the scale involved.

Besides array shapes and the round trip, two checks pin down conventions that
new_curv relies on but cannot verify itself:

* orientation: for a straight line at angle theta (degrees counterclockwise
  from the column axis, rows pointing down, so 0 is horizontal and 90 is
  vertical), the strongest wedge must report ``angle_deg`` close to theta
  modulo 180;
* position: for a small oriented patch, the strongest coefficient must sit
  at the patch's (1-based) pixel position, give or take one grid step.
"""

from __future__ import annotations

import numpy as np

from .protocol import CurveletBackend, ScaleGeometry

# Relative reconstruction error allowed for inverse(forward(img)).
ROUNDTRIP_RTOL = 1e-6
# Line orientations used by the orientation check; avoids 0/45/90/135, where
# two mirrored wedges can tie.
ORIENTATION_TEST_ANGLES_DEG = (20.0, 70.0, 110.0, 160.0)
# Allowed orientation error, in units of angle_step_deg. More than one step
# because FDCT labels its slope-uniform wedges with evenly spaced nominal
# angles; at 512x512 that puts the strongest wedge 1.06 steps off.
ORIENTATION_TOLERANCE_STEPS = 1.5
# Extra slack (pixels) on top of one coefficient-grid step in the position check.
POSITION_SLACK_PX = 2.0


def _test_image(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    stripes = np.sin(2 * np.pi * (rows * 0.6 + cols * 0.8) / 9.0)
    return (100 + 50 * stripes + 10 * rng.standard_normal(shape)).astype(np.float64)


def _line_image(shape: tuple[int, int], theta_deg: float, width: float = 2.0) -> np.ndarray:
    """Gaussian-profile line through the image centre at ``theta_deg``."""
    rows, cols = np.indices(shape, dtype=float)
    r0, c0 = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    t = np.radians(theta_deg)
    dist = np.abs((cols - c0) * -np.sin(t) - (rows - r0) * np.cos(t))
    return np.exp(-(dist**2) / (2 * width**2))


def _patch_image(
    shape: tuple[int, int], center: tuple[int, int], theta_deg: float, period: float = 4.0, sigma: float = 4.0
) -> np.ndarray:
    """Gaussian-windowed grating at ``center`` (0-based), stripes along ``theta_deg``."""
    rows, cols = np.indices(shape, dtype=float)
    t = np.radians(theta_deg)
    across = (cols - center[1]) * -np.sin(t) - (rows - center[0]) * np.cos(t)
    envelope = np.exp(-((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2 * sigma**2))
    return envelope * np.cos(2 * np.pi * across / period)


def _angle_error_mod180(a_deg: float, b_deg: float) -> float:
    d = (a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def _strongest_wedge(coefficients, scale, n_wedges, reduce):
    scores = [reduce(np.abs(np.asarray(coefficients[scale][k]))) for k in range(n_wedges)]
    return int(np.argmax(scores))


def _check_orientation(backend, shape, scale) -> None:
    geometry = backend.scale_geometry(shape, scale)
    allowed = ORIENTATION_TOLERANCE_STEPS * geometry.angle_step_deg
    for theta in ORIENTATION_TEST_ANGLES_DEG:
        coefficients = backend.forward(_line_image(shape, theta))
        k = _strongest_wedge(coefficients, scale, len(geometry.wedges), lambda a: np.sum(a**2))
        got = geometry.wedges[k].angle_deg
        error = _angle_error_mod180(got, theta)
        assert error <= allowed, (
            f"{backend.name} backend, shape {shape}, scale {scale}: line at {theta} deg "
            f"is strongest in wedge {k} with angle_deg={got:.2f} "
            f"(off by {error:.2f} deg, allowed {allowed:.2f})"
        )


def _grid_step(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values))
    return float(np.max(np.diff(unique))) if unique.size > 1 else 0.0


def _check_position(backend, shape, scale) -> None:
    geometry = backend.scale_geometry(shape, scale)
    # Asymmetric target, so transposed or flipped positions fail.
    center = (int(0.3 * shape[0]), int(0.7 * shape[1]))
    coefficients = backend.forward(_patch_image(shape, center, theta_deg=30.0))
    k = _strongest_wedge(coefficients, scale, len(geometry.wedges), np.max)
    flat_index = int(np.argmax(np.abs(np.asarray(coefficients[scale][k])).ravel()))
    wedge = geometry.wedges[k]
    got = (float(np.asarray(wedge.rows)[flat_index]), float(np.asarray(wedge.cols)[flat_index]))
    expected = (center[0] + 1, center[1] + 1)
    allowed = (
        _grid_step(wedge.rows) + POSITION_SLACK_PX,
        _grid_step(wedge.cols) + POSITION_SLACK_PX,
    )
    assert abs(got[0] - expected[0]) <= allowed[0] and abs(got[1] - expected[1]) <= allowed[1], (
        f"{backend.name} backend, shape {shape}, scale {scale}: patch at 1-based (row, col) "
        f"{expected} peaks in wedge {k} at {got}; allowed offset {allowed}"
    )


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
    _check_orientation(backend, shape, len(coefficients) - 2)
    _check_position(backend, shape, len(coefficients) - 2)

    reconstructed = np.real(backend.inverse(coefficients, shape))
    assert reconstructed.shape == shape, f"{backend.name}: inverse returned shape {reconstructed.shape}"
    error = np.max(np.abs(reconstructed - img)) / np.max(np.abs(img))
    assert error < ROUNDTRIP_RTOL, (
        f"{backend.name}: inverse(forward(img)) differs from img by {error:.2e} (relative)"
    )
