"""Build configuration for the compiled C++ extension `fiber_backend`.

Project metadata lives in ``pyproject.toml``; this file exists only because
setuptools cannot declare ``ext_modules`` in ``pyproject.toml``. It compiles the
pybind11 backend in ``src/ctfire_py/CPP/`` into ``ctfire_py.fiber_backend``.

The compiled backend is **required** — there is no pure-Python implementation of
the FIRE algorithm. If the C++ toolchain or OpenMP is unavailable the build
aborts with platform-specific instructions rather than installing a
backend-less package.
"""

import sys
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# The four C++ translation units that make up the backend. The pybind11 module
# itself is declared via PYBIND11_MODULE(fiber_backend, ...) in fire_utils.cpp.
CPP_DIR = "src/ctfire_py/CPP"
SOURCES = sorted(glob(f"{CPP_DIR}/*.cpp"))

# Name it as a submodule of the package so the artifact lands at
# src/ctfire_py/fiber_backend.<EXT_SUFFIX> — exactly where the sys.path hack in
# fire_2d_angle.py (and an in-place editable install) expects to import it.
ext_modules = [
    Pybind11Extension(
        "ctfire_py.fiber_backend",
        sources=SOURCES,
        include_dirs=[CPP_DIR],  # for link_fibre.h
        cxx_std=14,
    ),
]


def _toolchain_help() -> str:
    """Platform-specific instructions for installing a C++ + OpenMP toolchain."""
    if sys.platform == "win32":
        return (
            "  Windows:\n"
            "    - Install 'Microsoft C++ Build Tools' (Visual Studio Build Tools\n"
            "      with the 'Desktop development with C++' workload). OpenMP ships\n"
            "      with MSVC. Then re-run the install.\n"
            "    - OR use MSYS2 UCRT64: `pacman -S mingw-w64-ucrt-x86_64-gcc` and\n"
            "      run the install with the UCRT64 Python."
        )
    if sys.platform == "darwin":
        return (
            "  macOS:\n"
            "    - Install the Xcode command-line tools: `xcode-select --install`\n"
            "    - Install OpenMP runtime: `brew install libomp`\n"
            "    Then re-run the install."
        )
    return (
        "  Linux:\n"
        "    - Install a C++ compiler with OpenMP, e.g.\n"
        "      `sudo apt install build-essential libgomp1` (or your distro's\n"
        "      equivalent). Then re-run the install."
    )


class FiberBackendBuildExt(build_ext):
    """Inject OpenMP flags per compiler; fail loudly with guidance if the build can't proceed."""

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == "msvc":
                ext.extra_compile_args += ["/openmp", "/O2"]
            elif sys.platform == "darwin":
                ext.extra_compile_args += ["-O2", "-Xpreprocessor", "-fopenmp"]
                ext.extra_link_args += ["-lomp"]
            else:  # gcc / mingw / other unix
                ext.extra_compile_args += ["-O2", "-fopenmp"]
                ext.extra_link_args += ["-fopenmp"]
        super().build_extensions()

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 — re-raised after guidance
            sys.stderr.write(
                "\n"
                "============================================================\n"
                "ERROR: failed to build the required C++ extension "
                "'fiber_backend'.\n"
                "\n"
                "ctfire_py has no pure-Python fallback, so installation cannot\n"
                "continue without it. This usually means a C++ compiler or\n"
                "OpenMP is missing. To fix:\n"
                "\n"
                f"{_toolchain_help()}\n"
                "\n"
                f"Underlying error: {exc}\n"
                "============================================================\n"
            )
            raise


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": FiberBackendBuildExt},
)
