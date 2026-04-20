cmake_minimum_required(VERSION 3.18)
project(_ctfire_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ──────────────────────────────────────────────────────────────────
# Find via pip-installed pybind11 (preferred) or system install.
# Run: pip install pybind11 scikit-build-core
find_package(pybind11 CONFIG REQUIRED)

# ── Extension module ─────────────────────────────────────────────────────────
pybind11_add_module(_ctfire_cpp
    fire.cpp
    fire_bindings.cpp
)

target_include_directories(_ctfire_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ── Install: place .pyd/.so next to ctfire_utils.py ──────────────────────────
# Output path relative to repo root:
#   src/tme_quant/src/tme_quant/fiber_analysis/utils/
install(TARGETS _ctfire_cpp
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/../utils"
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/../utils"
)

# ── Build instructions ────────────────────────────────────────────────────────
# From the repo root (src/tme_quant/):
#
#   pip install pybind11 scikit-build-core
#   cmake -S src/tme_quant/fiber_analysis/_cpp/ctfire \
#         -B build/ctfire_cpp \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/ctfire_cpp
#   cmake --install build/ctfire_cpp
#
# Or use scikit-build-core via pyproject.toml (add [tool.scikit-build] section).
