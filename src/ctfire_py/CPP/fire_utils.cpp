#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 1. Forward declare the functions from your other .cpp files
py::tuple extend_xlink_native(int sizex, int sizey, int sizez, py::array_t<float, py::array::c_style> image, py::array_t<int32_t> pts_in, py::dict p);

py::tuple fiberproc_native(int sizex, int sizey, int sizez, py::array_t<float, py::array::c_style> image, py::array_t<int32_t> X_in, py::list F_in, py::array_t<float> R_in, py::dict p);

py::array_t<int32_t> findlocmax_native(int sizex, int sizey, int sizez, py::array_t<float, py::array::c_style | py::array::forcecast> image, int radius, float dmin);


// 2. Map them all in a single module block
PYBIND11_MODULE(fiber_backend, m) {
    m.doc() = "C++ backend for fiber extraction"; 

    m.def("extend_xlink", &extend_xlink_native, "Extend fiber segments and link them");
    m.def("process_fibers", &fiberproc_native, "Process and link fiber segments");
    m.def("find_local_max", &findlocmax_native, "Find local maxima");
}