"""Pluggable curvelet transform backends (FDCT, UDCT, ...).

See :mod:`pycurvelets.backends.protocol` for the interface and
:mod:`pycurvelets.backends.registry` for how a backend is chosen.
"""

from .protocol import Coefficients, CurveletBackend, ScaleGeometry, WedgeGeometry
from .registry import (
    BACKEND_ENV_VAR,
    ENTRY_POINT_GROUP,
    BackendUnavailableError,
    available_backends,
    has_working_backend,
    load_backend,
    resolve_backend_name,
)

__all__ = [
    "BACKEND_ENV_VAR",
    "ENTRY_POINT_GROUP",
    "BackendUnavailableError",
    "Coefficients",
    "CurveletBackend",
    "ScaleGeometry",
    "WedgeGeometry",
    "available_backends",
    "has_working_backend",
    "load_backend",
    "resolve_backend_name",
]
