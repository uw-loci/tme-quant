// pybind11 bindings for _ctfire_cpp
// Compiled module is imported in fiber_analysis/utils/ctfire_utils.py.
// Build: see CMakeLists.txt. Output: _ctfire_cpp.pyd (Windows) / _ctfire_cpp.so (Linux/macOS)
// Install output next to ctfire_utils.py: fiber_analysis/utils/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fire.h"

namespace py = pybind11;

PYBIND11_MODULE(_ctfire_cpp, m) {
    m.doc() = "CT-FIRE FIRE algorithm C++ extension for tme_quant";

    // ── Parameter structs ────────────────────────────────────────────────────

    py::class_<ctfire::FireParams2D>(m, "FireParams2D")
        .def(py::init<>())
        .def_readwrite("pixel_size_um", &ctfire::FireParams2D::pixel_size_um)
        .def_readwrite("min_length_um", &ctfire::FireParams2D::min_length_um)
        .def_readwrite("max_width_um",  &ctfire::FireParams2D::max_width_um)
        .def_readwrite("min_straight",  &ctfire::FireParams2D::min_straight);

    py::class_<ctfire::FireParams3D>(m, "FireParams3D")
        .def(py::init<>())
        .def_readwrite("pixel_size_um", &ctfire::FireParams3D::pixel_size_um)
        .def_readwrite("z_spacing_um",  &ctfire::FireParams3D::z_spacing_um)
        .def_readwrite("min_length_um", &ctfire::FireParams3D::min_length_um)
        .def_readwrite("max_width_um",  &ctfire::FireParams3D::max_width_um)
        .def_readwrite("min_straight",  &ctfire::FireParams3D::min_straight);

    // ── Result structs ───────────────────────────────────────────────────────

    py::class_<ctfire::FiberTrace>(m, "FiberTrace")
        .def_readonly("x",           &ctfire::FiberTrace::x)
        .def_readonly("y",           &ctfire::FiberTrace::y)
        .def_readonly("width",       &ctfire::FiberTrace::width)
        .def_readonly("length",      &ctfire::FiberTrace::length)
        .def_readonly("straightness",&ctfire::FiberTrace::straightness)
        .def_readonly("mean_width",  &ctfire::FiberTrace::mean_width);

    py::class_<ctfire::FiberTrace3D>(m, "FiberTrace3D")
        .def_readonly("x",           &ctfire::FiberTrace3D::x)
        .def_readonly("y",           &ctfire::FiberTrace3D::y)
        .def_readonly("z",           &ctfire::FiberTrace3D::z)
        .def_readonly("width",       &ctfire::FiberTrace3D::width)
        .def_readonly("length",      &ctfire::FiberTrace3D::length)
        .def_readonly("straightness",&ctfire::FiberTrace3D::straightness)
        .def_readonly("mean_width",  &ctfire::FiberTrace3D::mean_width);

    // ── Entry points (called from ctfire_utils.py) ───────────────────────────

    m.def("fire_2d",
        [](py::array_t<uint8_t, py::array::c_style> mask,
           const ctfire::FireParams2D& params) {
            auto buf = mask.request();
            if (buf.ndim != 2)
                throw std::runtime_error("fire_2d: mask must be 2-D (H, W)");
            return ctfire::fire_2d(
                static_cast<const uint8_t*>(buf.ptr),
                static_cast<int>(buf.shape[0]),
                static_cast<int>(buf.shape[1]),
                params
            );
        },
        py::arg("mask"), py::arg("params") = ctfire::FireParams2D{},
        "Run FIRE fiber extraction on a 2-D binary mask.\n\n"
        "Parameters\n----------\n"
        "mask   : np.ndarray, uint8, shape (H, W)  — binary fiber mask\n"
        "params : FireParams2D\n\n"
        "Returns\n-------\n"
        "list[FiberTrace]"
    );

    m.def("fire_3d",
        [](py::array_t<uint8_t, py::array::c_style> mask,
           const ctfire::FireParams3D& params) {
            auto buf = mask.request();
            if (buf.ndim != 3)
                throw std::runtime_error("fire_3d: mask must be 3-D (D, H, W)");
            return ctfire::fire_3d(
                static_cast<const uint8_t*>(buf.ptr),
                static_cast<int>(buf.shape[0]),
                static_cast<int>(buf.shape[1]),
                static_cast<int>(buf.shape[2]),
                params
            );
        },
        py::arg("mask"), py::arg("params") = ctfire::FireParams3D{},
        "Run FIRE fiber extraction on a 3-D binary mask.\n\n"
        "Parameters\n----------\n"
        "mask   : np.ndarray, uint8, shape (D, H, W)  — binary fiber mask\n"
        "params : FireParams3D\n\n"
        "Returns\n-------\n"
        "list[FiberTrace3D]"
    );
}
