"""Keep pycurvelets-core independent of the curvelet transform packages.

Core finds transforms only through the ``pycurvelets.backends`` entry points,
so it must install and import with no backend present. These tests check
that statically (no import of a provider anywhere in core's source) and at
run time (the pipeline imports in a fresh interpreter where every provider is
unimportable and no backend is registered).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "pycurvelets"

# Top-level modules core must never import; backend projects import them.
FORBIDDEN_IMPORTS = ("curvelops", "curvelets", "pycurvelets_fdct", "pycurvelets_udct")


def _imported_top_level_modules(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


def test_core_source_never_imports_a_curvelet_provider():
    offenders = [
        f"{path.relative_to(CORE_SRC.parent)}:{lineno} imports {module}"
        for path in sorted(CORE_SRC.rglob("*.py"))
        for lineno, module in _imported_top_level_modules(path)
        if module in FORBIDDEN_IMPORTS
    ]
    assert not offenders, "core must reach transforms through pycurvelets.backends:\n" + "\n".join(offenders)


_NO_BACKEND_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {blocked!r}

    class BlockProviders(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"{{name}} is blocked by this test")
            return None

    sys.meta_path.insert(0, BlockProviders())

    from pycurvelets.backends import registry
    registry.entry_points = lambda group: []

    import pycurvelets
    import pycurvelets.get_ct
    import pycurvelets.new_curv
    import pycurvelets.process_image
    from pycurvelets.backends import BackendUnavailableError, load_backend

    assert pycurvelets.HAS_CURVELETS is False, "HAS_CURVELETS should be False with no backend"
    try:
        load_backend()
    except BackendUnavailableError as error:
        print("OK:", error)
    else:
        raise AssertionError("load_backend() should fail with no backend installed")
    """
)


def test_pipeline_imports_without_any_backend():
    script = _NO_BACKEND_SCRIPT.format(blocked=set(FORBIDDEN_IMPORTS))
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK:" in result.stdout
    # The error must tell the user how to get a backend.
    assert "pycurvelets-fdct" in result.stdout or "pycurvelets-udct" in result.stdout, result.stdout
