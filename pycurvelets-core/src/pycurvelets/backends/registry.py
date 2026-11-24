"""Find and load curvelet transform backends through entry points.

A backend package registers a factory (normally its backend class) under the
``pycurvelets.backends`` entry-point group, for example in its
``pyproject.toml``::

    [project.entry-points."pycurvelets.backends"]
    fdct = "pycurvelets_fdct.backend:FDCTBackend"

Which backend runs is decided in this order:

1. the ``backend`` argument passed by the caller,
2. the ``TMEQ_CURVELET_BACKEND`` environment variable,
3. the only installed backend, or ``fdct`` when several are installed.

There is no silent fallback: if the chosen backend is missing or fails to
import, :class:`BackendUnavailableError` is raised with the reason. The choice
and the reason for it are logged at INFO level on the ``pycurvelets.backends``
logger.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import EntryPoint, entry_points

from .protocol import CurveletBackend

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pycurvelets.backends"
BACKEND_ENV_VAR = "TMEQ_CURVELET_BACKEND"
# Used when several backends are installed and none was named, so that
# results do not change for existing FDCT users.
PREFERRED_BACKEND = "fdct"

INSTALL_HINTS = {
    "fdct": "install pycurvelets-fdct (needs a local CurveLab and FFTW build; put CurveLab in utils/ and run bin/install.sh)",
    "udct": "install pycurvelets-udct (pure Python; from a tme-quant checkout: pip install -e ./pycurvelets-udct)",
}


class BackendUnavailableError(RuntimeError):
    """The requested curvelet backend is not installed or cannot be loaded."""


def _install_hint(name: str) -> str:
    return INSTALL_HINTS.get(name, f"install the package that provides the {name!r} backend")


def _backend_entry_points() -> dict[str, EntryPoint]:
    found: dict[str, EntryPoint] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        previous = found.get(ep.name)
        if previous is not None and previous.value != ep.value:
            raise BackendUnavailableError(
                f"Two installed packages register the curvelet backend {ep.name!r}: "
                f"{previous.value} and {ep.value}. Uninstall one of them."
            )
        found[ep.name] = ep
    return found


def available_backends() -> list[str]:
    """Return the names of the installed backends, whether or not they import.

    Returns
    -------
    list of str
        Sorted entry-point names in the ``pycurvelets.backends`` group.
    """
    return sorted(_backend_entry_points())


def resolve_backend_name(name: str | None = None) -> tuple[str, str]:
    """Decide which backend to use without importing it.

    Parameters
    ----------
    name : str, optional
        Backend requested by the caller. Takes precedence over everything else.

    Returns
    -------
    tuple of (str, str)
        The backend name and a short reason for the choice, for logs and errors.

    Raises
    ------
    BackendUnavailableError
        If no name was given and no backend, or several backends without
        ``fdct`` among them, are installed.
    """
    if name:
        return name, "requested by the caller"
    env_name = os.environ.get(BACKEND_ENV_VAR, "").strip()
    if env_name:
        return env_name, f"set by {BACKEND_ENV_VAR}"

    installed = available_backends()
    if not installed:
        hints = "; ".join(f"{n}: {h}" for n, h in sorted(INSTALL_HINTS.items()))
        raise BackendUnavailableError(
            "No curvelet backend is installed. Install one of them, then retry. "
            f"{hints}"
        )
    if len(installed) == 1:
        return installed[0], "the only installed backend"
    if PREFERRED_BACKEND in installed:
        return PREFERRED_BACKEND, f"default when several backends are installed ({installed})"
    raise BackendUnavailableError(
        f"Several curvelet backends are installed ({installed}) and none was chosen. "
        f"Pass backend=... or set {BACKEND_ENV_VAR}."
    )


def load_backend(name: str | None = None) -> CurveletBackend:
    """Load and instantiate a curvelet backend.

    Parameters
    ----------
    name : str, optional
        Backend to load, e.g. ``"fdct"`` or ``"udct"``. When omitted the
        choice follows the order described in the module docstring.

    Returns
    -------
    CurveletBackend
        A new backend instance.

    Raises
    ------
    BackendUnavailableError
        If the backend is not installed, fails to import or construct (the
        original exception is chained as ``__cause__``), or does not satisfy
        :class:`~pycurvelets.backends.protocol.CurveletBackend`.
    """
    resolved, reason = resolve_backend_name(name)
    installed = _backend_entry_points()
    ep = installed.get(resolved)
    if ep is None:
        raise BackendUnavailableError(
            f"Curvelet backend {resolved!r} ({reason}) is not installed. "
            f"Installed backends: {sorted(installed) or 'none'}. "
            f"To add it, {_install_hint(resolved)}."
        )

    try:
        factory = ep.load()
        backend = factory()
    except Exception as exc:
        raise BackendUnavailableError(
            f"Curvelet backend {resolved!r} is installed ({ep.value}) but could not "
            f"be loaded: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(backend, CurveletBackend):
        raise BackendUnavailableError(
            f"Entry point {ep.value} for curvelet backend {resolved!r} returned "
            f"{type(backend).__name__}, which does not implement CurveletBackend "
            "(needs name, forward, scale_geometry and inverse)."
        )

    logger.info("Using curvelet backend %r (%s) from %s", resolved, reason, ep.value)
    return backend


def has_working_backend() -> bool:
    """Return True if at least one installed backend loads.

    Backends that are installed but fail to load are reported as warnings on
    the ``pycurvelets.backends`` logger.
    """
    for backend_name in available_backends():
        try:
            load_backend(backend_name)
        except BackendUnavailableError as exc:
            logger.warning("Curvelet backend %r is installed but unusable: %s", backend_name, exc)
            continue
        return True
    return False
