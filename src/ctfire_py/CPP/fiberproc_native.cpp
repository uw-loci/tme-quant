#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "link_fibre.h"

namespace py = pybind11;

py::list fiberproc_native(
    int nPt,
    py::array_t<int32_t> X_in,
    py::list F_v_in,
    std::vector<int> nucleation_pts,
    float thresh_linka,
    int sp
) {
    uint64_t nSegments = F_v_in.size();
    using Fibre_T = Fibre<float, 2>;
    std::vector<Fibre_T> F_internal(nSegments);

    // Convert Python input to internal C++ structs
    for (size_t i = 0; i < nSegments; ++i) {
        std::vector<int> indices = F_v_in[i].cast<std::vector<int>>();
        // Parity with your port: indices in Python are already 0-indexed or 
        // we handle the -1 conversion here if they came directly from Matlab data.
        F_internal[i].link_index = indices; 
        
        for (int idx : indices) {
            F_internal[i].link.push_back({ (int)X_in.at(idx, 0), (int)X_in.at(idx, 1) });
        }
    }

    std::vector<std::vector<int>> F_out_vec;
    // Execute the linking logic
    LinkFibreAtNucleationPoint<float, 2> linker(nPt, nucleation_pts, F_internal, F_out_vec, thresh_linka, sp);

    // Convert back to Python List of Lists
    py::list out;
    for (const auto& fiber : F_out_vec) {
        out.append(py::cast(fiber));
    }
    return out;
}
