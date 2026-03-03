#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include "link_fibre.h"

namespace py = pybind11;

// This function processes fibers after initial extraction
// It takes the raw fiber segments and links them at nucleation points
// Returns: X (vertices), F (linked fibers), E (edges), V (vertex info), R (radii)
py::tuple fiberproc_native(
    int sizex, int sizey, int sizez,
    py::array_t<float, py::array::c_style> image,
    py::array_t<int32_t> X_in,
    py::list F_in,
    py::array_t<float> R_in,
    py::dict p
) {
    std::cout << "Image size: " << sizex << " x " << sizey << " x " << sizez << std::endl;
    
    // Extract parameters
    float thresh_linka = p["thresh_linka"].cast<float>();
    int s_fiberdir = p["s_fiberdir"].cast<int>();
    
    const int nX = X_in.shape(0);
    
    // Convert X to internal format (0-based indexing)
    std::vector<std::array<int, 2>> X_2D;
    std::vector<std::array<int, 3>> X_3D;
    
    if (sizex == 1) {
        X_2D.resize(nX);
        #pragma omp parallel for
        for (int i = 0; i < nX; ++i) {
            X_2D[i] = {X_in.at(i, 0) - 1, X_in.at(i, 1) - 1};  // Convert to 0-based
        }
    } else {
        X_3D.resize(nX);
        #pragma omp parallel for
        for (int i = 0; i < nX; ++i) {
            X_3D[i] = {X_in.at(i, 0) - 1, X_in.at(i, 1) - 1, X_in.at(i, 2) - 1};  // Convert to 0-based
        }
    }
    
    // Convert F from Python to internal format
    std::vector<std::vector<int>> F_v;
    std::vector<std::vector<int>> F_f;
    
    int nF = F_in.size();
    F_v.resize(nF);
    F_f.resize(nF);
    
    for (int i = 0; i < nF; ++i) {
        py::dict fiber_dict = F_in[i];
        
        // Extract 'v' field (vertex indices)
        if (fiber_dict.contains("v")) {
            std::vector<int> v_list = fiber_dict["v"].cast<std::vector<int>>();
            F_v[i].resize(v_list.size());
            for (size_t j = 0; j < v_list.size(); ++j) {
                F_v[i][j] = v_list[j] - 1;  // Convert to 0-based indexing
            }
        }
        
        // Extract 'f' field (connected fibers) if it exists
        if (fiber_dict.contains("f")) {
            std::vector<int> f_list = fiber_dict["f"].cast<std::vector<int>>();
            F_f[i].resize(f_list.size());
            for (size_t j = 0; j < f_list.size(); ++j) {
                F_f[i][j] = f_list[j] - 1;  // Convert to 0-based indexing
            }
        }
    }
    
    // Convert R to internal format
    std::vector<float> R_float(nX);
    #pragma omp parallel for
    for (int i = 0; i < nX; ++i) {
        R_float[i] = R_in.at(i);
    }
    
    // TODO: Implement the actual fiber processing logic
    // This would include:
    // 1. Identifying nucleation points
    // 2. Linking fibers at nucleation points using LinkFibreAtNucleationPoint
    // 3. Computing edges (E)
    // 4. Building vertex connectivity information (V)
    // 5. Updating fiber and vertex data structures
    
    std::cout << "Number of input vertices: " << nX << std::endl;
    std::cout << "Number of input fibers: " << nF << std::endl;
    std::cout << "thresh_linka: " << thresh_linka << std::endl;
    std::cout << "s_fiberdir: " << s_fiberdir << std::endl;
    
    // For now, return the input data (placeholder implementation)
    // A complete implementation would process and link the fibers
    
    // Convert X back to numpy array (1-based indexing for output)
    auto py_X = py::array_t<int32_t>({nX, 3});
    auto X_ptr = py_X.mutable_data();
    
    if (sizex == 1) {
        for (int i = 0; i < nX; ++i) {
            X_ptr[i * 3 + 0] = X_2D[i][0] + 1;  // Convert back to 1-based
            X_ptr[i * 3 + 1] = X_2D[i][1] + 1;
            X_ptr[i * 3 + 2] = 1;
        }
    } else {
        for (int i = 0; i < nX; ++i) {
            X_ptr[i * 3 + 0] = X_3D[i][0] + 1;  // Convert back to 1-based
            X_ptr[i * 3 + 1] = X_3D[i][1] + 1;
            X_ptr[i * 3 + 2] = X_3D[i][2] + 1;
        }
    }
    
    // Convert F back to Python list of dicts (1-based indexing for output)
    py::list py_F;
    for (int i = 0; i < nF; ++i) {
        py::dict f_struct;
        
        // Convert v indices back to 1-based
        std::vector<int> v_out(F_v[i].size());
        for (size_t j = 0; j < F_v[i].size(); ++j) {
            v_out[j] = F_v[i][j] + 1;
        }
        f_struct["v"] = py::cast(v_out);
        
        // Convert f indices back to 1-based
        std::vector<int> f_out(F_f[i].size());
        for (size_t j = 0; j < F_f[i].size(); ++j) {
            f_out[j] = F_f[i][j] + 1;
        }
        f_struct["f"] = py::cast(f_out);
        
        py_F.append(f_struct);
    }
    
    // Placeholder for E (edges) - would be computed from fiber connectivity
    py::list py_E;
    
    // Placeholder for V (vertex info) - would contain connectivity information
    py::list py_V;
    for (int i = 0; i < nX; ++i) {
        py::dict v_struct;
        v_struct["fe"] = py::list();  // Fibers ending at this vertex
        v_struct["f"] = py::list();   // All fibers passing through this vertex
        v_struct["vall"] = py::list(); // All vertices in connected fibers
        py_V.append(v_struct);
    }
    
    // Return R as numpy array
    auto py_R = py::array_t<float>(nX);
    auto R_ptr = py_R.mutable_data();
    for (int i = 0; i < nX; ++i) {
        R_ptr[i] = R_float[i];
    }
    
    std::cout << "WARNING: fiberproc_native is currently a placeholder implementation" << std::endl;
    std::cout << "Full fiber processing logic needs to be implemented" << std::endl;
    
    return py::make_tuple(py_X, py_F, py_E, py_V, py_R);
}
