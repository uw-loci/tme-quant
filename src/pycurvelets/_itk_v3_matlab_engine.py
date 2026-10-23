"""ITK v3 registration engine configured exactly like MATLAB ``imregtform``.

MATLAB's ``imregtform`` (Image Processing Toolbox) is a thin wrapper around a
compiled ITK 5.3 bridge (``libmwimagesitk``) that uses the *classic* (v3)
registration framework:

* ``itk::MultiResolutionImageRegistrationMethod`` (3 pyramid levels)
* ``itk::MultiResolutionPyramidImageFilter`` (shrink 4/2/1)
* ``itk::MattesMutualInformationImageToImageMetric`` (v3, 50 bins, all pixels)
* ``itk::LinearInterpolateImageFunction``
* ``itk::Similarity2DTransform`` / ``itk::AffineTransform``
* ``itk::OnePlusOneEvolutionaryOptimizer`` + ``itk::Statistics::NormalVariateGenerator``

SimpleITK only exposes the v4 framework, whose Mattes MI differs numerically,
so this module drives the v3 classes through the ``itk`` Python package
instead. Everything MATLAB sets in ``imregtform.m`` /
``computeDefaultRegmexSettings.m`` is reproduced here:

* world coordinates = MATLAB 1-based intrinsic coordinates (image origin at
  ``(1, 1)``, unit spacing), so the rotation centre (world origin) and the
  translation parameters mean the same thing as in MATLAB;
* optimizer scales ``[1, 1, 1/diag, 1/diag]`` (similarity) and
  ``[1, 1, 1, 1, 1/diag, 1/diag]`` (affine) with ``diag = hypot(W, H)`` of the
  fixed image extent;
* initial transform = identity + translation aligning the geometric centres;
* ``ShrinkFactor = GrowthFactor ** -0.25``;
* ``MaximumIterations`` applies per pyramid level.

Three things are *not* readable from MATLAB's .m files and were pinned
empirically against MATLAB ``imregtform`` runs (first ES steps with
``MaximumIterations = 1``):

* the RNG seed is ``12345`` (the ITK-example constant; the first normal
  variates of MATLAB's ES are exactly ``1.41727041, 0.04911296, ...``);
* the transform centre is the *fixed image centre* in world coordinates,
  ``((W+1)/2, (H+1)/2)`` with MATLAB's 1-based ``imref2d`` convention;
* a per-pyramid-level refiner (RTTI ``PyramidLevelOptimizerRefiner`` in
  ``libmwimagesitk``) rescales the optimizer before each level: the ES
  initial radius is multiplied by ``shrink_factor**2`` (16, 4, 1) and the
  epsilon by ``10**(levels-1-level)`` (100, 10, 1).

These are still exposed as parameters (``seed``, ``center``,
``level_radius_power``, ``level_epsilon_base``) so the harness can re-verify
them or explore alternatives.

Debugging aids: every call returns a ``dict`` with the per-iteration metric
trace (``trace``), stop conditions, level boundaries and the raw ITK
parameters, so a divergence from MATLAB can be localised to a specific
iteration rather than only to the final image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import itk  # type: ignore[import-untyped]
import numpy as np

# Defaults of ``[optimizer, metric] = imregconfig('multimodal')``.
# MathWorks documents these as properties of the objects that call returns,
# not as arguments of ``imregconfig`` itself:
# https://www.mathworks.com/help/images/ref/imregconfig.html
#
# ``registration.optimizer.OnePlusOneEvolutionary``
MATLAB_DEFAULT_INITIAL_RADIUS = 6.25e-3  # optimizer.InitialRadius
MATLAB_DEFAULT_GROWTH_FACTOR = 1.05  # optimizer.GrowthFactor
MATLAB_DEFAULT_EPSILON = 1.5e-6  # optimizer.Epsilon
MATLAB_DEFAULT_MAX_ITERATIONS = 100  # optimizer.MaximumIterations
# ``registration.metric.MattesMutualInformation``
MATLAB_DEFAULT_HISTOGRAM_BINS = 50  # metric.NumberOfHistogramBins
# ``imregtform`` name-value ``'PyramidLevels'`` (default 3). This is not an
# ``imregconfig`` field:
# https://www.mathworks.com/help/images/ref/imregtform.html
MATLAB_DEFAULT_PYRAMID_LEVELS = 3
# BDcreation does not run these defaults unchanged.
# ``register_bdcreation_reg2_matlab`` divides ``InitialRadius`` by 3.5 and
# sets ``MaximumIterations`` to 700, matching ``BDcreation_reg.m`` /
# ``BDcreation_reg2.m`` after ``imregconfig``.
# ITK examples (and, as far as we can tell, MATLAB's bridge) seed the normal
# variate generator with this constant.
DEFAULT_SEED = 12345


@dataclass
class OnePlusOneConfig:
    """Settings for MATLAB's ``OnePlusOneEvolutionary`` optimizer.

    Attributes
    ----------
    initial_radius : float
        Starting search radius. ``BDcreation_reg2`` divides the default by 3.5.
    growth_factor : float
        How fast a successful step grows the radius.
    epsilon : float
        Stop when the radius falls below this.
    maximum_iterations : int
        Iteration cap. ``BDcreation_reg2`` sets this to 700.
    """

    initial_radius: float = MATLAB_DEFAULT_INITIAL_RADIUS
    growth_factor: float = MATLAB_DEFAULT_GROWTH_FACTOR
    epsilon: float = MATLAB_DEFAULT_EPSILON
    maximum_iterations: int = MATLAB_DEFAULT_MAX_ITERATIONS

    @property
    def shrink_factor(self) -> float:
        """Radius shrink after a failed step.

        MATLAB's dependent property is ``GrowthFactor ** -0.25``.

        Returns
        -------
        float
        """
        return float(self.growth_factor) ** -0.25


@dataclass
class RegmexResult:
    """Output of one ``imregtform``-equivalent call.

    ``fixed_to_moving_3x3`` is the ITK/regmex transform (maps *fixed* world
    points to *moving* world points, world = MATLAB 1-based intrinsic).
    ``moving_to_fixed_1based_3x3`` is its inverse, i.e. MATLAB's ``tform.A``.
    ``forward_2x3_0based`` is the same forward transform re-expressed in the
    0-based pixel-centre convention used by
    :func:`pycurvelets._he_bdc_common.matlab_imwarp_bilinear`.

    Attributes
    ----------
    transform_type : str
        ``'similarity'`` or ``'affine'``.
    parameters : list of float
        Final ITK parameter vector.
    fixed_to_moving_3x3 : numpy.ndarray
        ITK transform, fixed world to moving world.
    moving_to_fixed_1based_3x3 : numpy.ndarray
        Inverse of that matrix. This is MATLAB's ``tform.A``.
    forward_2x3_0based : numpy.ndarray
        Same forward map in 0-based pixel centres.
    final_metric : float
        Last Mattes mutual-information value (ITK returns the negative).
    stop_condition : str
        Why the optimizer stopped.
    trace : list of dict
        One entry per iteration.
    level_starts : list of int
        Trace index where each pyramid level began.
    extras : dict
        Pyramid settings and the ITK version string.
    """

    transform_type: str
    parameters: list[float]
    fixed_to_moving_3x3: np.ndarray
    moving_to_fixed_1based_3x3: np.ndarray
    forward_2x3_0based: np.ndarray
    final_metric: float
    stop_condition: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    level_starts: list[int] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_debug_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flatten this result into a JSON-friendly dict.

        Parameters
        ----------
        prefix : str, optional
            Text prepended to every key. Default is empty.

        Returns
        -------
        dict
            Transform, metric, stop reason, and per-level trace.
        """
        return {
            f"{prefix}transform_type": self.transform_type,
            f"{prefix}parameters": list(self.parameters),
            f"{prefix}fixed_to_moving_3x3": self.fixed_to_moving_3x3.tolist(),
            f"{prefix}matlab_tform_A": self.moving_to_fixed_1based_3x3.tolist(),
            f"{prefix}forward_2x3": self.forward_2x3_0based.tolist(),
            f"{prefix}final_metric": float(self.final_metric),
            f"{prefix}stop_condition": self.stop_condition,
            f"{prefix}n_iterations": len(self.trace),
            f"{prefix}level_starts": list(self.level_starts),
            **{f"{prefix}{k}": v for k, v in self.extras.items()},
        }


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def moving_to_fixed_1based_to_forward_0based(A_1based: np.ndarray) -> np.ndarray:
    """Convert MATLAB's 1-based ``tform.A`` into a 0-based 2x3 forward map.

    The shift is ``t0 = M @ 1 + t - 1``.

    Parameters
    ----------
    A_1based : numpy.ndarray
        3x3 moving-to-fixed matrix on MATLAB's 1-based pixel centres.

    Returns
    -------
    numpy.ndarray
        2x3 map for 0-based pixel centres.
    """
    A = np.asarray(A_1based, dtype=np.float64)
    M = A[:2, :2]
    t = A[:2, 2]
    ones = np.ones(2, dtype=np.float64)
    t0 = M @ ones + t - ones
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = M
    out[:, 2] = t0
    return out


def forward_0based_to_moving_to_fixed_1based(forward_2x3: np.ndarray) -> np.ndarray:
    """Convert a 0-based 2x3 forward map back to MATLAB's 3x3 ``tform.A``.

    Parameters
    ----------
    forward_2x3 : numpy.ndarray
        2x3 map on 0-based pixel centres.

    Returns
    -------
    numpy.ndarray
        3x3 moving-to-fixed matrix on 1-based pixel centres.
    """
    F = np.asarray(forward_2x3, dtype=np.float64)
    M = F[:, :2]
    t0 = F[:, 2]
    ones = np.ones(2, dtype=np.float64)
    t1 = t0 - M @ ones + ones
    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = M
    A[:2, 2] = t1
    return A


def matlab_T_to_A(T: np.ndarray) -> np.ndarray:
    """Turn MATLAB's post-multiply ``tform.T`` into column-form ``tform.A``.

    Parameters
    ----------
    T : numpy.ndarray
        3x3 matrix stored the way ``tform.T`` is stored.

    Returns
    -------
    numpy.ndarray
        Transpose of ``T``.
    """
    return np.asarray(T, dtype=np.float64).T.copy()


# ---------------------------------------------------------------------------
# ITK helpers
# ---------------------------------------------------------------------------


def _to_itk_image(arr: np.ndarray, origin: tuple[float, float] = (1.0, 1.0)) -> Any:
    """Wrap a 2-D float array as an ITK image with unit spacing.

    The default origin ``(1, 1)`` puts pixel centres at 1..N, matching
    MATLAB's default ``imref2d``.

    Parameters
    ----------
    arr : numpy.ndarray
        2-D image.
    origin : tuple of float, optional
        World origin ``(x, y)``. Default is ``(1, 1)``.

    Returns
    -------
    itk.Image
        Double 2-D image.

    Raises
    ------
    ValueError
        If ``arr`` is not 2-D.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {a.shape}")
    img = itk.image_from_array(a)
    img.SetSpacing([1.0, 1.0])
    img.SetOrigin([float(origin[0]), float(origin[1])])
    return img


def _make_scales(values: list[float]) -> Any:
    """Build the ITK optimizer-scales vector MATLAB passes to ``imregtform``.

    Parameters
    ----------
    values : list of float
        One scale per transform parameter.

    Returns
    -------
    itk.OptimizerParameters
        Scales object the evolutionary optimizer expects.
    """
    arr = itk.Array[itk.D](len(values))
    for i, v in enumerate(values):
        arr.SetElement(i, float(v))
    return arr


def _params_from_itk(p: Any, n: int) -> list[float]:
    """Copy the first ``n`` ITK parameters into a Python list.

    Parameters
    ----------
    p : itk parameter object
        Object with ``GetElement``.
    n : int
        How many parameters to copy.

    Returns
    -------
    list of float
    """
    return [float(p.GetElement(i)) for i in range(n)]


def matlab_default_settings(
    transform_type: str,
    moving_shape: tuple[int, int],
    fixed_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Default transform and optimizer scales from ``computeDefaultRegmexSettings``.

    Parameters
    ----------
    transform_type : {'affine', 'similarity', 'rigid', 'translation'}
        Transform whose parameter scales are needed.
    moving_shape, fixed_shape : tuple of int
        ``(height, width)`` of the two images.

    Returns
    -------
    linear_2x2 : numpy.ndarray
        Identity 2x2 matrix.
    translation_2 : numpy.ndarray
        Shift that lines up the two image centres.
    optimizer_scales : list of float
        One scale per ITK parameter. Translation is scaled by the fixed
        image diagonal.

    Raises
    ------
    ValueError
        If ``transform_type`` is not one of the four names above.
    """
    mh, mw = int(moving_shape[0]), int(moving_shape[1])
    fh, fw = int(fixed_shape[0]), int(fixed_shape[1])
    # imref2d(size): XWorldLimits = [0.5, W + 0.5] -> mean = (W + 1) / 2.
    init_translation = np.array(
        [((mw + 1) / 2.0) - ((fw + 1) / 2.0), ((mh + 1) / 2.0) - ((fh + 1) / 2.0)],
        dtype=np.float64,
    )
    fixed_extent = sorted([float(fw), float(fh)])
    max_translation = float(np.hypot(fixed_extent[0], fixed_extent[1]))
    translation_scale = 1.0 / max_translation

    t = transform_type.lower()
    if t == "affine":
        scales = [1.0, 1.0, 1.0, 1.0, translation_scale, translation_scale]
    elif t == "similarity":
        scales = [1.0, 1.0, translation_scale, translation_scale]
    elif t == "rigid":
        scales = [1.0, translation_scale, translation_scale]
    elif t == "translation":
        scales = [translation_scale, translation_scale]
    else:
        raise ValueError(f"Unsupported transform type {transform_type!r}")
    return np.eye(2, dtype=np.float64), init_translation, scales


def _similarity_params_from_matrix(M: np.ndarray, t: np.ndarray) -> list[float]:
    """Read ITK similarity parameters from a 2x2 matrix and a translation.

    Parameters
    ----------
    M : numpy.ndarray
        ``scale * rotation``, shape ``(2, 2)``. The centre is the world origin.
    t : numpy.ndarray
        Translation, shape ``(2,)``.

    Returns
    -------
    list of float
        ``[scale, angle, tx, ty]``.
    """
    scale = float(np.sqrt(abs(np.linalg.det(M))))
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return [scale, angle, float(t[0]), float(t[1])]


def _build_transform(
    transform_type: str,
    linear_2x2: np.ndarray,
    translation_2: np.ndarray,
    center: tuple[float, float],
) -> Any:
    """Build an ITK transform for ``T(p) = M p + t``.

    ITK stores ``T(p) = M (p - c) + c + trans``, so the translation parameter
    is ``t - c + M c``. The built transform is checked against ``M`` and ``t``.

    Parameters
    ----------
    transform_type : {'similarity', 'affine'}
        Which ITK transform to create.
    linear_2x2 : numpy.ndarray
        Linear part ``M``.
    translation_2 : numpy.ndarray
        Translation ``t`` in world coordinates.
    center : tuple of float
        ITK transform centre ``(x, y)``.

    Returns
    -------
    itk.Transform
        Initialized similarity or affine transform.

    Raises
    ------
    RuntimeError
        If the ITK object does not reproduce ``M`` and ``t``.
    ValueError
        If ``transform_type`` is not ``'similarity'`` or ``'affine'``.
    """
    M = np.asarray(linear_2x2, dtype=np.float64)
    t = np.asarray(translation_2, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    trans = t - c + M @ c

    tt = transform_type.lower()
    if tt == "similarity":
        tr = itk.Similarity2DTransform[itk.D].New()
        tr.SetIdentity()
        tr.SetCenter([float(c[0]), float(c[1])])
        scale, angle, _, _ = _similarity_params_from_matrix(M, t)
        tr.SetScale(scale)
        tr.SetAngle(angle)
        tr.SetTranslation([float(trans[0]), float(trans[1])])
        got_M, got_t = _transform_to_matrix(tr)
        if not (np.allclose(got_M, M, atol=1e-9) and np.allclose(got_t, t, atol=1e-9)):
            raise RuntimeError(
                "Similarity2DTransform initialisation mismatch (input is not a pure "
                f"similarity?): wanted M={M.tolist()}, t={t.tolist()}, got "
                f"M={got_M.tolist()}, t={got_t.tolist()}"
            )
        return tr
    if tt == "affine":
        tr = itk.AffineTransform[itk.D, 2].New()
        tr.SetIdentity()
        tr.SetCenter([float(c[0]), float(c[1])])
        tr.SetMatrix(itk.matrix_from_array(np.ascontiguousarray(M, dtype=np.float64)))
        tr.SetTranslation([float(trans[0]), float(trans[1])])
        got_M, got_t = _transform_to_matrix(tr)
        if not (np.allclose(got_M, M, atol=1e-12) and np.allclose(got_t, t, atol=1e-9)):
            raise RuntimeError(
                f"AffineTransform initialisation mismatch: wanted M={M.tolist()}, "
                f"t={t.tolist()}, got M={got_M.tolist()}, t={got_t.tolist()}"
            )
        return tr
    raise ValueError(f"Unsupported transform type {transform_type!r}")


def _transform_to_matrix(transform: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read ``T(p) = M p + t`` by transforming three points.

    Parameters
    ----------
    transform : itk.Transform
        2-D transform.

    Returns
    -------
    M : numpy.ndarray
        2x2 linear part.
    t : numpy.ndarray
        Translation, equal to ``T(0)``.
    """
    p00 = np.array(transform.TransformPoint([0.0, 0.0]), dtype=np.float64)
    p10 = np.array(transform.TransformPoint([1.0, 0.0]), dtype=np.float64)
    p01 = np.array(transform.TransformPoint([0.0, 1.0]), dtype=np.float64)
    M = np.column_stack((p10 - p00, p01 - p00))
    return M, p00


# ---------------------------------------------------------------------------
# imregtform equivalent
# ---------------------------------------------------------------------------


def matlab_imregtform_v3(
    moving: np.ndarray,
    fixed: np.ndarray,
    transform_type: str,
    optimizer: OnePlusOneConfig,
    *,
    initial_fixed_to_moving_3x3: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    number_of_histogram_bins: int = MATLAB_DEFAULT_HISTOGRAM_BINS,
    pyramid_levels: int = MATLAB_DEFAULT_PYRAMID_LEVELS,
    center: str | tuple[float, float] = "fixed_image_center",
    level_radius_power: float = 2.0,
    level_epsilon_base: float = 10.0,
    reseed_per_level: bool = False,
    number_of_threads: int | None = None,
    verbose: bool = False,
) -> RegmexResult:
    """
    Python/ITK equivalent of MATLAB ``imregtform(moving, fixed, type, optimizer,
    metric)`` with ``imregconfig('multimodal')`` metric.

    Parameters
    ----------
    moving, fixed
        2-D float arrays (any range; the metric bins on min/max).
    transform_type
        ``"similarity"`` or ``"affine"``.
    optimizer
        :class:`OnePlusOneConfig` (radius/growth/epsilon/max-iterations).
    initial_fixed_to_moving_3x3
        Optional initial transform in *regmex* convention (fixed world ->
        moving world, 1-based). Equivalent to MATLAB's
        ``'InitialTransformation'`` after ``invert``. ``None`` -> identity +
        geometric-centre alignment (MATLAB default).
    seed
        ``NormalVariateGenerator`` seed.
    center
        ``"fixed_image_center"`` (MATLAB behaviour, world ``((W+1)/2,
        (H+1)/2)`` of the fixed image), ``"origin"`` (ITK default, world
        (0,0)), or an explicit ``(x, y)`` world tuple.
    level_radius_power
        Before each pyramid level the ES radius is set to
        ``initial_radius * shrink**level_radius_power`` where ``shrink`` is the
        level's pyramid factor (4, 2, 1). MATLAB uses 2; 0 disables.
    level_epsilon_base
        Before each level epsilon is set to
        ``epsilon * level_epsilon_base**(levels - 1 - level)``. MATLAB uses 10;
        1 disables.
    reseed_per_level
        Re-initialise the normal variate generator with ``seed`` at every
        level (MATLAB does not; kept for experiments).
    number_of_threads
        ITK global thread count (``None`` = ITK default). Thread count changes
        the floating-point summation order inside the metric; set to 1 for
        bit-stable traces.

    Returns
    -------
    RegmexResult
    """
    t = transform_type.lower()
    if t not in ("similarity", "affine"):
        raise ValueError("transform_type must be 'similarity' or 'affine'")

    moving_np = np.asarray(moving, dtype=np.float64)
    fixed_np = np.asarray(fixed, dtype=np.float64)
    if moving_np.ndim != 2 or fixed_np.ndim != 2:
        raise ValueError("moving and fixed must be 2-D")
    if min(moving_np.shape) < 4 ** (pyramid_levels - 1):
        raise ValueError(
            f"images too small ({moving_np.shape}) for {pyramid_levels} pyramid levels"
        )

    if number_of_threads is not None:
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(int(number_of_threads))

    ImageType = itk.Image[itk.D, 2]
    fixed_itk = _to_itk_image(fixed_np)
    moving_itk = _to_itk_image(moving_np)

    lin0, trans0, scales = matlab_default_settings(t, moving_np.shape, fixed_np.shape)
    if initial_fixed_to_moving_3x3 is not None:
        A0 = np.asarray(initial_fixed_to_moving_3x3, dtype=np.float64)
        lin0 = A0[:2, :2]
        trans0 = A0[:2, 2]

    if isinstance(center, str):
        if center == "origin":
            c = (0.0, 0.0)
        elif center in ("fixed_image_center", "image_center"):
            fh, fw = fixed_np.shape
            c = ((fw + 1) / 2.0, (fh + 1) / 2.0)
        else:
            raise ValueError(
                "center must be 'fixed_image_center', 'origin' or an (x, y) tuple"
            )
    else:
        c = (float(center[0]), float(center[1]))

    transform = _build_transform(t, lin0, trans0, c)
    n_params = transform.GetNumberOfParameters()

    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

    metric = itk.MattesMutualInformationImageToImageMetric[ImageType, ImageType].New()
    metric.SetNumberOfHistogramBins(int(number_of_histogram_bins))
    metric.UseAllPixelsOn()

    generator = itk.NormalVariateGenerator.New()
    generator.Initialize(int(seed))

    opt = itk.OnePlusOneEvolutionaryOptimizer.New()
    opt.SetNormalVariateGenerator(generator)
    opt.Initialize(float(optimizer.initial_radius))
    opt.SetGrowthFactor(float(optimizer.growth_factor))
    opt.SetShrinkFactor(float(optimizer.shrink_factor))
    opt.SetEpsilon(float(optimizer.epsilon))
    opt.SetMaximumIteration(int(optimizer.maximum_iterations))
    opt.MinimizeOn()  # Mattes MI returns -MI
    opt.SetScales(_make_scales(scales))

    fixed_pyramid = itk.MultiResolutionPyramidImageFilter[ImageType, ImageType].New()
    moving_pyramid = itk.MultiResolutionPyramidImageFilter[ImageType, ImageType].New()

    registration = itk.MultiResolutionImageRegistrationMethod[ImageType, ImageType].New()
    registration.SetMetric(metric)
    registration.SetOptimizer(opt)
    registration.SetTransform(transform)
    registration.SetInterpolator(interpolator)
    registration.SetFixedImage(fixed_itk)
    registration.SetMovingImage(moving_itk)
    registration.SetFixedImageRegion(fixed_itk.GetBufferedRegion())
    registration.SetFixedImagePyramid(fixed_pyramid)
    registration.SetMovingImagePyramid(moving_pyramid)
    registration.SetNumberOfLevels(int(pyramid_levels))
    registration.SetInitialTransformParameters(transform.GetParameters())

    trace: list[dict[str, Any]] = []
    level_starts: list[int] = []
    level_settings: list[dict[str, float]] = []
    n_levels = int(pyramid_levels)

    def _on_level() -> None:
        """Reset radius, epsilon, and the random seed at each pyramid level.

        This is the port of MATLAB's ``PyramidLevelOptimizerRefiner``. It
        runs before ``StartOptimization`` on that level.

        Returns
        -------
        None
        """
        level = int(registration.GetCurrentLevel())
        shrink = 2.0 ** (n_levels - 1 - level)  # ITK default schedule 4/2/1
        radius_l = float(optimizer.initial_radius) * shrink ** float(level_radius_power)
        eps_l = float(optimizer.epsilon) * float(level_epsilon_base) ** (n_levels - 1 - level)
        opt.Initialize(radius_l, float(optimizer.growth_factor), float(optimizer.shrink_factor))
        opt.SetEpsilon(eps_l)
        if reseed_per_level:
            generator.Initialize(int(seed))
        level_starts.append(len(trace))
        level_settings.append({"level": level, "shrink": shrink, "radius": radius_l, "epsilon": eps_l})
        if verbose:
            print(f"  -- pyramid level {level + 1}/{n_levels}: radius={radius_l:.4g} epsilon={eps_l:.3g} --")

    def _on_iteration() -> None:
        """Append the current cost and parameters to the trace.

        Returns
        -------
        None
        """
        pos = opt.GetCurrentPosition()
        trace.append(
            {
                "iter": int(opt.GetCurrentIteration()),
                "metric": float(opt.GetCurrentCost()),
                "fnorm": float(opt.GetFrobeniusNorm()),
                "params": _params_from_itk(pos, n_params),
            }
        )
        if verbose:
            print(
                f"    it={opt.GetCurrentIteration():4d} metric={opt.GetCurrentCost():.6f} "
                f"fnorm={opt.GetFrobeniusNorm():.3e}"
            )

    opt.AddObserver(itk.IterationEvent(), _on_iteration)
    registration.AddObserver(itk.MultiResolutionIterationEvent(), _on_level)

    registration.Update()

    final_params = registration.GetLastTransformParameters()
    transform.SetParameters(final_params)
    M, tvec = _transform_to_matrix(transform)
    fixed_to_moving = np.eye(3, dtype=np.float64)
    fixed_to_moving[:2, :2] = M
    fixed_to_moving[:2, 2] = tvec
    moving_to_fixed_1based = np.linalg.inv(fixed_to_moving)
    forward_0based = moving_to_fixed_1based_to_forward_0based(moving_to_fixed_1based)

    return RegmexResult(
        transform_type=t,
        parameters=_params_from_itk(final_params, n_params),
        fixed_to_moving_3x3=fixed_to_moving,
        moving_to_fixed_1based_3x3=moving_to_fixed_1based,
        forward_2x3_0based=forward_0based,
        final_metric=float(opt.GetCurrentCost()),
        stop_condition=str(opt.GetStopConditionDescription()),
        trace=trace,
        level_starts=level_starts,
        extras={
            "seed": int(seed),
            "center": [float(c[0]), float(c[1])],
            "scales": list(scales),
            "initial_radius": float(optimizer.initial_radius),
            "growth_factor": float(optimizer.growth_factor),
            "shrink_factor": float(optimizer.shrink_factor),
            "epsilon": float(optimizer.epsilon),
            "maximum_iterations": int(optimizer.maximum_iterations),
            "pyramid_levels": int(pyramid_levels),
            "level_settings": level_settings,
            "histogram_bins": int(number_of_histogram_bins),
            "itk_version": str(itk.Version.GetITKVersion()),
        },
    )


# ---------------------------------------------------------------------------
# BDcreation_reg2.m registration block
# ---------------------------------------------------------------------------


def register_bdcreation_reg2_matlab(
    he_moving: np.ndarray,
    fixed_shg: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    center: str | tuple[float, float] = "fixed_image_center",
    radius_divisor: float = 3.5,
    maximum_iterations: int = 700,
    run_warmup: bool = False,
    number_of_threads: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reproduce the registration block of ``BDcreation_reg2.m``::

        [optimizer,metric] = imregconfig('multimodal');
        optimizer.InitialRadius = optimizer.InitialRadius/3.5;
        imregister(HEmoving, fixedSHG, 'affine', optimizer, metric);  % discarded
        optimizer.MaximumIterations = 700;
        tformSimilarity = imregtform(HEmoving, fixedSHG, 'similarity', optimizer, metric);
        tform = imregtform(HEmoving, fixedSHG, 'affine', optimizer, metric, ...
                           'InitialTransformation', tformSimilarity);

    The discarded ``imregister`` warm-up has no side effects on the optimizer
    (value class) and is skipped unless ``run_warmup=True``.

    Parameters
    ----------
    he_moving : numpy.ndarray
        Binary or grayscale H&E moving image.
    fixed_shg : numpy.ndarray
        SHG fixed image.
    seed : int, optional
        Random seed for the evolutionary optimizer. Default is 12345.
    center : str or tuple of float, optional
        Transform centre. Default is the fixed-image centre.
    radius_divisor : float, optional
        Divides MATLAB's default initial radius. Default is 3.5.
    maximum_iterations : int, optional
        Iteration cap for both stages. Default is 700.
    run_warmup : bool, optional
        When True, also run the discarded ``imregister`` call. Default is False.
    number_of_threads : int or None, optional
        ITK thread count. ``None`` leaves the ITK default.
    verbose : bool, optional
        Print pyramid-level progress. Default is False.

    Returns
    -------
    forward_2x3 : np.ndarray
        Forward affine (moving -> fixed) in 0-based pixel-centre convention,
        ready for :func:`pycurvelets._he_bdc_common.matlab_imwarp_bilinear`
        via ``_affine_fixed_to_moving_from_forward``.
    debug : dict
        Both stage results (traces, MATLAB-convention ``tform.A`` matrices).
    """
    cfg = OnePlusOneConfig(
        initial_radius=MATLAB_DEFAULT_INITIAL_RADIUS / float(radius_divisor),
        maximum_iterations=int(maximum_iterations),
    )
    debug: dict[str, Any] = {"matlab_engine": "itk_v3", "seed": int(seed)}

    if run_warmup:
        warm_cfg = OnePlusOneConfig(
            initial_radius=cfg.initial_radius,
            maximum_iterations=MATLAB_DEFAULT_MAX_ITERATIONS,
        )
        warm = matlab_imregtform_v3(
            he_moving, fixed_shg, "affine", warm_cfg,
            seed=seed, center=center, number_of_threads=number_of_threads,
            verbose=verbose,
        )
        debug.update(warm.as_debug_dict("warmup_"))

    if verbose:
        print("[matlab-parity] stage 1: similarity")
    sim = matlab_imregtform_v3(
        he_moving, fixed_shg, "similarity", cfg,
        seed=seed, center=center, number_of_threads=number_of_threads,
        verbose=verbose,
    )
    debug.update(sim.as_debug_dict("sim_"))

    if verbose:
        print("[matlab-parity] stage 2: affine (InitialTransformation = similarity)")
    aff = matlab_imregtform_v3(
        he_moving, fixed_shg, "affine", cfg,
        initial_fixed_to_moving_3x3=sim.fixed_to_moving_3x3,
        seed=seed, center=center, number_of_threads=number_of_threads,
        verbose=verbose,
    )
    debug.update(aff.as_debug_dict("aff_"))

    return aff.forward_2x3_0based.copy(), debug
