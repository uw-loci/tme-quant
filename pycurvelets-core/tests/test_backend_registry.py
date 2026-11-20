"""Backend selection and loading through the ``pycurvelets.backends`` entry points.

These tests replace the installed entry points with fake ones, so they run
without any curvelet library installed.
"""

import sys
import types
from importlib.metadata import EntryPoint

import numpy as np
import pytest
from pycurvelets.backends import (
    BACKEND_ENV_VAR,
    ENTRY_POINT_GROUP,
    BackendUnavailableError,
    CurveletBackend,
    ScaleGeometry,
    available_backends,
    has_working_backend,
    load_backend,
    registry,
    resolve_backend_name,
)

FAKE_MODULE = "_tmeq_fake_curvelet_backends"


class _FakeBackend:
    name = "fake"

    def forward(self, img):
        return [[np.asarray(img, dtype=float)]]

    def scale_geometry(self, shape, scale):
        return ScaleGeometry(angle_step_deg=90.0, wedges=[])

    def inverse(self, coefficients, shape):
        return np.zeros(shape)


class _NotABackend:
    name = "broken"


def _failing_factory():
    raise RuntimeError("CurveLab library not found")


@pytest.fixture
def fake_backends(monkeypatch):
    """Install fake entry points; returns a setter taking {name: attribute}."""
    module = types.ModuleType(FAKE_MODULE)
    module.FakeBackend = _FakeBackend
    module.NotABackend = _NotABackend
    module.failing_factory = _failing_factory
    monkeypatch.setitem(sys.modules, FAKE_MODULE, module)
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)

    def install(mapping):
        eps = [
            EntryPoint(name=name, value=target if ":" in target else f"{FAKE_MODULE}:{target}", group=ENTRY_POINT_GROUP)
            for name, target in mapping.items()
        ]
        monkeypatch.setattr(registry, "entry_points", lambda group: eps if group == ENTRY_POINT_GROUP else [])

    return install


def test_available_backends_lists_entry_point_names(fake_backends):
    fake_backends({"udct": "FakeBackend", "fdct": "FakeBackend"})
    assert available_backends() == ["fdct", "udct"]


def test_load_backend_returns_protocol_instance(fake_backends):
    fake_backends({"fake": "FakeBackend"})
    backend = load_backend("fake")
    assert isinstance(backend, CurveletBackend)
    assert backend.name == "fake"


def test_only_installed_backend_is_chosen(fake_backends):
    fake_backends({"udct": "FakeBackend"})
    assert resolve_backend_name() == ("udct", "the only installed backend")


def test_env_var_selects_backend(fake_backends, monkeypatch):
    fake_backends({"fdct": "FakeBackend", "udct": "FakeBackend"})
    monkeypatch.setenv(BACKEND_ENV_VAR, "udct")
    assert resolve_backend_name()[0] == "udct"


def test_explicit_name_beats_env_var(fake_backends, monkeypatch):
    fake_backends({"fdct": "FakeBackend", "udct": "FakeBackend"})
    monkeypatch.setenv(BACKEND_ENV_VAR, "udct")
    assert resolve_backend_name("fdct") == ("fdct", "requested by the caller")


def test_fdct_is_default_when_several_installed(fake_backends):
    fake_backends({"fdct": "FakeBackend", "udct": "FakeBackend"})
    assert resolve_backend_name()[0] == "fdct"


def test_several_backends_without_fdct_must_be_named(fake_backends):
    fake_backends({"udct": "FakeBackend", "other": "FakeBackend"})
    with pytest.raises(BackendUnavailableError, match="none was chosen"):
        resolve_backend_name()


def test_no_backend_installed_lists_install_commands(fake_backends):
    fake_backends({})
    with pytest.raises(BackendUnavailableError, match="pycurvelets-udct"):
        load_backend()


def test_missing_named_backend_reports_installed_ones(fake_backends):
    fake_backends({"udct": "FakeBackend"})
    with pytest.raises(BackendUnavailableError, match=r"'fdct'.*not installed.*\['udct'\]"):
        load_backend("fdct")


def test_import_failure_keeps_original_exception(fake_backends):
    fake_backends({"fdct": "_tmeq_module_that_does_not_exist:Backend"})
    with pytest.raises(BackendUnavailableError, match="could not be loaded") as info:
        load_backend("fdct")
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_constructor_failure_keeps_original_exception(fake_backends):
    fake_backends({"fdct": "failing_factory"})
    with pytest.raises(BackendUnavailableError, match="CurveLab library not found") as info:
        load_backend("fdct")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_object_without_protocol_methods_is_rejected(fake_backends):
    fake_backends({"broken": "NotABackend"})
    with pytest.raises(BackendUnavailableError, match="does not implement CurveletBackend"):
        load_backend("broken")


def test_conflicting_registrations_are_reported(fake_backends, monkeypatch):
    eps = [
        EntryPoint(name="fdct", value=f"{FAKE_MODULE}:FakeBackend", group=ENTRY_POINT_GROUP),
        EntryPoint(name="fdct", value=f"{FAKE_MODULE}:NotABackend", group=ENTRY_POINT_GROUP),
    ]
    monkeypatch.setattr(registry, "entry_points", lambda group: eps)
    with pytest.raises(BackendUnavailableError, match="Two installed packages"):
        available_backends()


def test_has_working_backend(fake_backends):
    fake_backends({})
    assert has_working_backend() is False
    fake_backends({"broken": "NotABackend"})
    assert has_working_backend() is False
    fake_backends({"broken": "NotABackend", "fake": "FakeBackend"})
    assert has_working_backend() is True
