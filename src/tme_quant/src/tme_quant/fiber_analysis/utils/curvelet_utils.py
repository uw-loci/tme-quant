# -*- coding: utf-8 -*-
"""
Curvelet transform utilities for CurveAlign and CT-FIRE fiber analysis.

Architecture
============
Three backends are supported, selected in priority order:

  1. **curvelops** (recommended for production)
     Python wrapper around a C++ curvelet implementation via the PyLops
     framework.  Supports genuine 2-D and 3-D curvelet transforms.
     Install: ``pip install curvelops``
     Source:  https://github.com/PyLops/curvelops

  2. **MATLAB Engine** (legacy, for direct CT-FIRE / CurveAlign parity)
     Calls the original MATLAB CT-FIRE curvelet functions via
     ``matlab.engine``.  Requires a licensed MATLAB installation and the
     MATLAB Engine for Python package.

  3. **NumPy / SciPy fallback** (always available, approximate)
     FFT-based directional filter bank that approximates curvelet angular
     energy responses.  Suitable for development, unit tests, and quick
     prototyping; NOT a substitute for genuine curvelet coefficients on
     production data.

3-D semantics
=============
Functions with the ``_3d`` suffix operate on **volumetric** data
``(Z, H, W)`` — true 3-D analysis that treats the volume as a single
entity, not slice-by-slice processing of independent 2-D planes.

  * ``curvelet_transform_3d`` with the curvelops backend applies a genuine
    3-D FDCT (Fast Discrete Curvelet Transform) treating (Z, H, W) jointly.
  * The NumPy fallback **does not** implement true 3-D curvelets; it falls
    back to slice-by-slice processing and warns accordingly.  Any downstream
    code that needs accurate 3-D volumetric results should install curvelops.

References
----------
- Bredfeldt et al. (2014) Computational segmentation of collagen fibers from
  second-harmonic generation images of breast cancer. J Biomed Opt 19(1):016007.
- Peng et al. (2017) CurveAlign 4.0 — quantitative analysis of fiber
  organization in the ECM. https://loci.wisc.edu/software/curvealign
- Ravasi & Ulrich (2021) PyLops — a linear-operator Python library for scalable
  geophysical inversions. SoftwareX 13:100604.
- curvelops: https://github.com/PyLops/curvelops
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# Backend availability probes  (evaluated lazily at first call)
# ─────────────────────────────────────────────────────────────────────────────

def _has_curvelops() -> bool:
    """Return True if the curvelops package is importable."""
    try:
        import curvelops  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _has_matlab_engine() -> bool:
    """Return True if the MATLAB Engine for Python is importable."""
    try:
        import matlab.engine  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — 2-D
# ─────────────────────────────────────────────────────────────────────────────

def curvelet_transform_2d(
    image: np.ndarray,
    n_levels: int = 4,
    n_angles: int = 8,
    use_matlab: bool = False,
    use_curvelops: bool = True,
) -> np.ndarray:
    """
    Compute a 2-D curvelet transform of *image*.

    Returns an array of shape ``(H, W, n_angles)`` where ``[..., k]`` holds
    the energy at angular bin ``k * 180 / n_angles`` degrees.

    Backend priority
    ----------------
    1. curvelops  (if ``use_curvelops=True`` and package is installed)
    2. MATLAB Engine  (if ``use_matlab=True`` and engine is available)
    3. NumPy / SciPy approximation  (always available, approximate)

    Parameters
    ----------
    image : ndarray, shape (H, W)
        2-D grayscale image (any numeric dtype).
    n_levels : int
        Number of curvelet decomposition levels.
    n_angles : int
        Number of angular bins.  Should be a power of 2, ≥ 4.
    use_matlab : bool
        Try the MATLAB Engine backend before the NumPy fallback.
    use_curvelops : bool
        Try the curvelops backend first (default True).

    Returns
    -------
    ndarray, shape (H, W, n_angles), dtype float32
    """
    if image.ndim != 2:
        raise ValueError(f"curvelet_transform_2d expects a 2-D image, got shape {image.shape}")

    if use_curvelops and _has_curvelops():
        result = _curvelops_curvelet_2d(image, n_levels, n_angles)
        if result is not None:
            return result

    if use_matlab and _has_matlab_engine():
        result = _matlab_curvelet_2d(image, n_levels, n_angles)
        if result is not None:
            return result

    # NumPy fallback
    return _numpy_curvelet_2d(image, n_levels, n_angles)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — 3-D  (true volumetric transform)
# ─────────────────────────────────────────────────────────────────────────────

def curvelet_transform_3d(
    image: np.ndarray,
    n_levels: int = 3,
    n_angles: int = 8,
    use_matlab: bool = False,
    use_curvelops: bool = True,
) -> np.ndarray:
    """
    Compute a **true volumetric** 3-D curvelet transform of *image*.

    This function treats ``image`` as a single 3-D volume (Z, H, W), not as
    a stack of independent 2-D slices.  The curvelet transform is applied
    jointly across all three spatial dimensions, capturing fiber structures
    that traverse multiple z-planes.

    .. important::
        True 3-D curvelet analysis requires the **curvelops** backend.
        If curvelops is not installed, this function falls back to
        slice-by-slice 2-D processing and emits a ``UserWarning``.
        Slice-by-slice results **cannot** detect out-of-plane fibers and
        are NOT equivalent to volumetric analysis.

    Returns an array of shape ``(Z, H, W, n_angles)`` where ``[z, ..., k]``
    holds the energy at angular bin ``k`` for slice ``z`` (or the marginalised
    3-D curvelet energy when using curvelops).

    Parameters
    ----------
    image : ndarray, shape (Z, H, W)
        3-D grayscale volume.
    n_levels : int
        Number of curvelet decomposition levels.
    n_angles : int
        Number of angular bins per scale.
    use_matlab : bool
        Try the MATLAB Engine backend (3-D CT-FIRE curvelet).
    use_curvelops : bool
        Try the curvelops 3-D FDCT backend (recommended, default True).

    Returns
    -------
    ndarray, shape (Z, H, W, n_angles), dtype float32

    Raises
    ------
    ValueError
        If *image* is not 3-D.
    """
    if image.ndim != 3:
        raise ValueError(f"curvelet_transform_3d expects a 3-D volume, got shape {image.shape}")

    if use_curvelops and _has_curvelops():
        result = _curvelops_curvelet_3d(image, n_levels, n_angles)
        if result is not None:
            return result

    if use_matlab and _has_matlab_engine():
        result = _matlab_curvelet_3d(image, n_levels, n_angles)
        if result is not None:
            return result

    warnings.warn(
        "True 3-D curvelet transform requires the 'curvelops' package "
        "(pip install curvelops).  Falling back to slice-by-slice 2-D "
        "processing — out-of-plane fibers will NOT be detected.  "
        "Install curvelops for genuine volumetric analysis.",
        UserWarning,
        stacklevel=2,
    )
    slices = [_numpy_curvelet_2d(image[z], n_levels, n_angles) for z in range(image.shape[0])]
    return np.stack(slices, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# curvelops backend  (production quality, genuine FDCT)
# ─────────────────────────────────────────────────────────────────────────────

def _curvelops_curvelet_2d(
    image: np.ndarray,
    n_levels: int,
    n_angles: int,
) -> Optional[np.ndarray]:
    """
    2-D curvelet transform via curvelops (FDCT wrapping layer).

    Uses ``curvelops.FDCT2D`` to compute genuine curvelet coefficients,
    then projects them into the ``(H, W, n_angles)`` energy format expected
    by CurveAlign and CT-FIRE processing code.

    Returns None on any error so the caller can fall back gracefully.
    """
    try:
        import curvelops  # type: ignore

        img = image.astype(np.float64)
        h, w = img.shape

        # FDCT2D: nscales controls decomposition levels; angles_coarse
        # is the number of directions at the coarsest detail scale.
        # curvelops doubles angles at each finer scale (parabolic scaling).
        fdct = curvelops.FDCT2D(
            n=(h, w),
            nscales=n_levels,
            angles_coarse=n_angles,
            real=True,
        )

        # Forward transform: returns a list-of-lists of coefficient patches
        coeffs_struct = fdct.struct(fdct * img.ravel())

        # ── Collapse coefficients into (H, W, n_angles) energy map ──────────
        # Strategy: reconstruct each angular wedge separately to get a
        # spatial energy map per angle bin.  This is analogous to how the
        # original CurveAlign MATLAB code uses curvelet subbands.
        result = np.zeros((h, w, n_angles), dtype=np.float32)

        for scale_idx, scale_bands in enumerate(coeffs_struct):
            if scale_idx == 0:        # coarsest scale — no orientation
                continue
            n_wedges = len(scale_bands)
            for wedge_idx, wedge_coeffs in enumerate(scale_bands):
                # Map wedge index to one of our n_angles bins
                angle_bin = (wedge_idx * n_angles) // n_wedges
                angle_bin = min(angle_bin, n_angles - 1)

                # Build a coefficient structure with only this wedge active
                zero_struct = [[np.zeros_like(c) for c in band]
                               for band in coeffs_struct]
                zero_struct[scale_idx][wedge_idx] = wedge_coeffs

                # Reconstruct spatial image for this wedge
                reconstructed = fdct.H * fdct.unravel(zero_struct)
                spatial = reconstructed.reshape(h, w).astype(np.float32)

                # Accumulate energy
                result[:, :, angle_bin] += spatial ** 2

        return result

    except Exception as exc:
        warnings.warn(
            f"curvelops 2-D transform failed ({exc}); using NumPy fallback.",
            UserWarning,
            stacklevel=3,
        )
        return None


def _curvelops_curvelet_3d(
    image: np.ndarray,
    n_levels: int,
    n_angles: int,
) -> Optional[np.ndarray]:
    """
    True volumetric 3-D curvelet transform via curvelops (FDCT3D).

    Uses ``curvelops.FDCT3D`` to perform a genuine joint 3-D fast discrete
    curvelet transform.  Fiber structures that span multiple z-planes are
    captured by the 3-D angular wedges.

    The output ``(Z, H, W, n_angles)`` marginalises 3-D wedge energies into
    the angular dimension by projecting 3-D orientations onto the 2-D angular
    bins corresponding to the principal plane (the XY plane is used here as
    it matches the acquisition plane in most microscopy data).

    Returns None on any error.
    """
    try:
        import curvelops  # type: ignore

        vol = image.astype(np.float64)
        z, h, w = vol.shape

        fdct = curvelops.FDCT3D(
            n=(z, h, w),
            nscales=n_levels,
            angles_coarse=n_angles,
            real=True,
        )

        coeffs_struct = fdct.struct(fdct * vol.ravel())

        result = np.zeros((z, h, w, n_angles), dtype=np.float32)

        for scale_idx, scale_bands in enumerate(coeffs_struct):
            if scale_idx == 0:
                continue
            n_wedges = len(scale_bands)
            for wedge_idx, wedge_coeffs in enumerate(scale_bands):
                angle_bin = (wedge_idx * n_angles) // n_wedges
                angle_bin = min(angle_bin, n_angles - 1)

                zero_struct = [[np.zeros_like(c) for c in band]
                               for band in coeffs_struct]
                zero_struct[scale_idx][wedge_idx] = wedge_coeffs

                reconstructed = fdct.H * fdct.unravel(zero_struct)
                spatial = reconstructed.reshape(z, h, w).astype(np.float32)
                result[:, :, :, angle_bin] += spatial ** 2

        return result

    except Exception as exc:
        warnings.warn(
            f"curvelops 3-D transform failed ({exc}); will use fallback.",
            UserWarning,
            stacklevel=3,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB Engine bridge  (legacy / CT-FIRE parity)
# ─────────────────────────────────────────────────────────────────────────────

def _matlab_curvelet_2d(
    image: np.ndarray,
    n_levels: int,
    n_angles: int,
) -> Optional[np.ndarray]:
    """Call the original MATLAB CT-FIRE curvelet function. Returns None on failure."""
    try:
        import matlab.engine  # type: ignore
        eng    = matlab.engine.start_matlab()
        mat_im = matlab.double(image.tolist())
        result = eng.curvelet_transform_2d(mat_im, float(n_levels), float(n_angles))
        return np.array(result, dtype=np.float32)
    except Exception:
        return None


def _matlab_curvelet_3d(
    image: np.ndarray,
    n_levels: int,
    n_angles: int,
) -> Optional[np.ndarray]:
    """Call the original MATLAB CT-FIRE 3-D curvelet function. Returns None on failure."""
    try:
        import matlab.engine  # type: ignore
        eng    = matlab.engine.start_matlab()
        mat_im = matlab.double(image.tolist())
        result = eng.curvelet_transform_3d(mat_im, float(n_levels), float(n_angles))
        return np.array(result, dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NumPy / SciPy fallback  (approximate, always available)
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_curvelet_2d(
    image: np.ndarray,
    n_levels: int,
    n_angles: int,
) -> np.ndarray:
    """
    FFT-based directional filter bank approximating 2-D curvelet responses.

    For each of the ``n_angles`` directions, the image is filtered with a
    bandpass directional Gaussian in the Fourier domain, and the spatial
    energy map is accumulated.  This approximation captures the angular
    selectivity of curvelets but lacks the parabolic scaling law that gives
    true curvelets their anisotropy.

    Suitable for unit tests and development.  **Not** a replacement for
    genuine curvelet coefficients on production SHG / fluorescence data.
    """
    img = image.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()

    h, w   = img.shape
    result = np.zeros((h, w, n_angles), dtype=np.float32)

    fft_img    = np.fft.fft2(img)
    fy         = np.fft.fftfreq(h)[:, None] * np.ones((1, w))
    fx         = np.fft.fftfreq(w)[None, :] * np.ones((h, 1))
    freq_angle = np.degrees(np.arctan2(fy, fx)) % 180.0   # [0, 180)
    freq_mag   = np.sqrt(fx ** 2 + fy ** 2)

    angle_bw = 180.0 / n_angles   # degrees per bin

    for k in range(n_angles):
        center = k * angle_bw

        # Wrapped angular Gaussian
        diff = freq_angle - center
        diff = (diff + 90) % 180 - 90
        angular_f = np.exp(-0.5 * (diff / (angle_bw * 0.6)) ** 2)

        # Multi-scale bandpass (parabolic scaling approximation)
        scale_f = np.zeros_like(freq_mag)
        for lvl in range(1, n_levels + 1):
            lo = 2 ** (lvl - 1) / (2 * max(h, w))
            hi = 2 **  lvl      / (2 * max(h, w))
            scale_f += ((freq_mag >= lo) & (freq_mag < hi)).astype(np.float32)
        scale_f = np.clip(scale_f, 0.0, 1.0)

        filtered = np.abs(np.fft.ifft2(fft_img * angular_f * scale_f)).astype(np.float32)
        result[..., k] = gaussian_filter(filtered, sigma=1.0)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def available_backends() -> dict:
    """
    Report which curvelet backends are currently available.

    Returns
    -------
    dict with keys ``'curvelops'``, ``'matlab'``, ``'numpy'`` mapping to bool.
    """
    return {
        'curvelops': _has_curvelops(),
        'matlab':    _has_matlab_engine(),
        'numpy':     True,   # always available
    }


__all__ = [
    'curvelet_transform_2d',
    'curvelet_transform_3d',
    'available_backends',
]