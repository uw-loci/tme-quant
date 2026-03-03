# --- Configuration ---
# The output filename automatically grabs the correct suffix (e.g., .cpython-310-darwin.so)
TARGET = fiber_backend$(shell python3-config --extension-suffix)

# Compiler setup
CXX = clang++
# -undefined dynamic_lookup is strictly required on macOS so it doesn't freak out about missing Python symbols during compilation
CXXFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -undefined dynamic_lookup

# Python and Pybind11 Flags
PY_INCLUDES = $(shell python3 -m pybind11 --includes)
# CRITICAL: Do NOT use --embed flag on macOS as it causes segfaults during module import
# The -undefined dynamic_lookup flag handles Python symbol resolution correctly
PY_LDFLAGS =

# OpenMP Configuration (Pulls directly from your active Conda environment)
OMP_FLAGS = -Xpreprocessor -fopenmp
OMP_INCLUDES = -I$(CONDA_PREFIX)/include
OMP_LIBS = -L$(CONDA_PREFIX)/lib -lomp

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