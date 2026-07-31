# --- Configuration ---
# configurable variable for python calls
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MAKEFILE_DIR)/../../..)
UV_PYTHON := $(REPO_ROOT)/.venv/bin/python

PYTHON ?= $(if $(wildcard $(UV_PYTHON)),$(UV_PYTHON),python3)
# The output filename automatically grabs the correct suffix (e.g., .cpython-310-darwin.so)
TARGET = fiber_backend$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Compiler setup
CXX = clang++
# -undefined dynamic_lookup is strictly required on macOS so it doesn't freak out about missing Python symbols during compilation
CXXFLAGS = -g -O0 -Wall -shared -std=c++14 -fPIC -undefined dynamic_lookup

# Python and Pybind11 Flags
# Include both pybind11 and Python development headers
PY_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
# CRITICAL: Do NOT use --embed flag on macOS as it causes segfaults during module import
# The -undefined dynamic_lookup flag handles Python symbol resolution correctly
PY_LDFLAGS =

# OpenMP Configuration
OMP_FLAGS = -Xpreprocessor -fopenmp
BREW_LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
OMP_PREFIX ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX),$(BREW_LIBOMP))
OMP_INCLUDES = -I$(OMP_PREFIX)/include
OMP_LIBS = -L$(OMP_PREFIX)/lib -lomp

# Source files 
SRCS = extend_xlink_native.cpp fiberproc_native.cpp findlocmax_native.cpp fire_utils.cpp

# --- Build Rules ---
all: $(TARGET)

# Notice we added link_fibre.h here. It tells `make` to recompile if you edit the header file!
$(TARGET): $(SRCS) link_fibre.h
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) $(OMP_FLAGS) $(OMP_INCLUDES) $(SRCS) $(PY_LDFLAGS) $(OMP_LIBS) -o $(TARGET)

clean:
	rm -f *.so *.pyd

.PHONY: all clean
