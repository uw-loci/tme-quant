#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <omp.h>
#include <iostream>
#include "link_fibre.h" 

namespace py = pybind11;

// Note: MATLAB is column major, C++/Python is row major
template<typename T, int d>
struct ExtendXLink {
    using Fibre_Type = Fibre<T, d>;

    static T Length(const std::array<T, d>& v) {
        T val = 0;
        for (int i = 0; i < d; ++i) val += v[i] * v[i];
        return sqrt(val);
    }

    static T Dot(const std::array<T, d>& v1, const std::array<T, d>& v2) {
        T val = 0;
        for (int i = 0; i < d; ++i) val += v1[i] * v2[i];
        return val;
    }

    // 2D Constructor - pts in {y, x} format
    explicit ExtendXLink(int sizex, int sizey, T* image, 
                         const std::vector<std::array<int, d>>& pts,
                         int thresh_LMPdist, T thresh_LMP, T thresh_ext, T lambda, 
                         T thresh_linkd, T thresh_linka, int sp,
                         std::vector<std::array<int, d>>& X, std::vector<T>& R, 
                         std::vector<std::vector<int>>& F,
                         std::vector<std::vector<int>>& Xfe, std::vector<std::vector<int>>& Xf,
                         std::vector<std::vector<int>>& Xvall, std::vector<std::vector<int>>& Ff) {
        
        static_assert(d == 2, "2D Implementation Only");

        // Allocate and initialize maps
        int* index_map = new int[sizex * sizey];
        int* nucleation_map = new int[sizex * sizey];
        memset(nucleation_map, 0, sizeof(int) * sizex * sizey);
        memset(index_map, 0, sizeof(int) * sizex * sizey);

        // Mark nucleation points (store index + 1 to distinguish from 0)
        #pragma omp parallel for
        for (int i = 0; i < (int)pts.size(); ++i) {
            nucleation_map[pts[i][0] * sizex + pts[i][1]] = i + 1;
        }

        std::vector<std::vector<Fibre_Type>> fibres(pts.size());

        // Step 1: Find Local Maxima Points (LMP) and extend fibers
        #pragma omp parallel for
        for (int i = 0; i < (int)pts.size(); ++i) {
            const int r_i = ceil(image[pts[i][0] * sizex + pts[i][1]]);
            const std::array<int, d> nucleation{pts[i][0], pts[i][1]};
            const std::array<int, d> b_min = {nucleation[0] - r_i, nucleation[1] - r_i};
            const std::array<int, d> b_max = {nucleation[0] + r_i, nucleation[1] + r_i};

            // Search for LMP on the boundary
            for (int ii = -r_i; ii <= r_i; ++ii) {
                for (int jj = -r_i; jj <= r_i; ++jj) {
                    const std::array<int, d> p{ii + pts[i][0], jj + pts[i][1]};
                    
                    // Bounds check
                    if (p[0] < 0 || p[0] >= sizey || p[1] < 0 || p[1] >= sizex) continue;
                    
                    // Only check boundary points
                    if (p[0] != b_min[0] && p[0] != b_max[0] && 
                        p[1] != b_min[1] && p[1] != b_max[1]) continue;
                    
                    const T d_value = image[p[0] * sizex + p[1]];
                    if (d_value < thresh_LMP) continue;

                    // Check if this is a local maximum
                    bool is_LMP = true;
                    for (int iii = -1; iii <= 1 && is_LMP; ++iii) {
                        for (int jjj = -1; jjj <= 1; ++jjj) {
                            const std::array<int, d> p_n{iii + p[0], jjj + p[1]};
                            if (p_n[0] < 0 || p_n[0] >= sizey || p_n[1] < 0 || p_n[1] >= sizex) continue;
                            if (p_n[0] < b_min[0] || p_n[0] > b_max[0] || 
                                p_n[1] < b_min[1] || p_n[1] > b_max[1]) continue;
                            if (p_n[0] != b_min[0] && p_n[0] != b_max[0] && 
                                p_n[1] != b_min[1] && p_n[1] != b_max[1]) continue;
                            
                            const T d_value_n = image[p_n[0] * sizex + p_n[1]];
                            if (d_value < d_value_n) { 
                                is_LMP = false; 
                                break; 
                            }
                        }
                    }

                    if (is_LMP) {
                        // Check if too close to existing fiber endpoints
                        bool too_close = false;
                        for (size_t b = 0; b < fibres[i].size(); ++b) {
                            if (abs(fibres[i][b].link[1][0] - p[0]) < thresh_LMPdist && 
                                abs(fibres[i][b].link[1][1] - p[1]) < thresh_LMPdist) {
                                too_close = true; 
                                break;
                            }
                        }
                        if (too_close) continue;

                        // Start a new fiber
                        fibres[i].push_back(Fibre_Type{});
                        fibres[i].back().link.push_back(nucleation);
                        fibres[i].back().link.push_back(p);
                        
                        std::array<T, d> dir{T(p[0] - nucleation[0]), T(p[1] - nucleation[1])};
                        T l = Length(dir);
                        dir = {dir[0] / l, dir[1] / l};

                        // Extend the fiber
                        std::array<int, d> p_current(p);
                        bool found_next = true;
                        
                        while (found_next) {
                            found_next = false;  // Reset - will be set to true if we find next point
                            bool found_nucleation = false;
                            std::array<int, d> nucleation_pt{};
                            
                            const int r_curr = ceil(image[p_current[0] * sizex + p_current[1]]);
                            const std::array<int, d> cur_min = {p_current[0] - r_curr, p_current[1] - r_curr};
                            const std::array<int, d> cur_max = {p_current[0] + r_curr, p_current[1] + r_curr};
                            
                            // Collect ALL LMP candidates first (like MATLAB does)
                            struct LMPCandidate {
                                std::array<int, d> pt;
                                T d_value;
                                T dir_dot;
                            };
                            std::vector<LMPCandidate> lmp_candidates;
                            
                            for (int m = -r_curr; m <= r_curr && !found_nucleation; ++m) {
                                for (int n = -r_curr; n <= r_curr; ++n) {
                                    const std::array<int, d> p_cand{m + p_current[0], n + p_current[1]};
                                    if (p_cand[0] < 0 || p_cand[0] >= sizey || 
                                        p_cand[1] < 0 || p_cand[1] >= sizex) continue;
                                    
                                    const uint64_t offset = p_cand[0] * sizex + p_cand[1];
                                    
                                    // Check if we hit another nucleation point
                                    if (nucleation_map[offset] && 
                                        (p_cand[0] != nucleation[0] || p_cand[1] != nucleation[1])) {
                                        nucleation_pt = p_cand;
                                        found_nucleation = true;
                                        break;
                                    }

                                    // Only check boundary points
                                    if (p_cand[0] != cur_min[0] && p_cand[0] != cur_max[0] && 
                                        p_cand[1] != cur_min[1] && p_cand[1] != cur_max[1]) continue;
                                    
                                    const T d_val = image[offset];
                                    if (d_val < thresh_LMP) continue;

                                    // Check if it's a local maximum
                                    bool is_LMP_cand = true;
                                    for (int iii = -1; iii <= 1 && is_LMP_cand; ++iii) {
                                        for (int jjj = -1; jjj <= 1; ++jjj) {
                                            const std::array<int, d> p_n{iii + p_cand[0], jjj + p_cand[1]};
                                            if (p_n[0] < 0 || p_n[0] >= sizey || 
                                                p_n[1] < 0 || p_n[1] >= sizex) continue;
                                            if (p_n[0] < cur_min[0] || p_n[0] > cur_max[0] || 
                                                p_n[1] < cur_min[1] || p_n[1] > cur_max[1]) continue;
                                            if (p_n[0] != cur_min[0] && p_n[0] != cur_max[0] && 
                                                p_n[1] != cur_min[1] && p_n[1] != cur_max[1]) continue;
                                            
                                            const T d_value_n = image[p_n[0] * sizex + p_n[1]];
                                            if (d_val < d_value_n) {
                                                is_LMP_cand = false;
                                                break;
                                            }
                                        }
                                    }

                                    if (is_LMP_cand) {
                                        // Compute direction for this candidate
                                        std::array<T, d> new_dir{T(p_cand[0] - p_current[0]), 
                                                                 T(p_cand[1] - p_current[1])};
                                        T nl = Length(new_dir);
                                        new_dir = {new_dir[0] / nl, new_dir[1] / nl};
                                        
                                        T dir_dot = Dot(new_dir, dir);
                                        
                                        // Add to candidates list (filter by direction LATER like MATLAB)
                                        lmp_candidates.push_back({p_cand, d_val, dir_dot});
                                    }
                                }
                            }
                            
                            // Process LMP candidates (like MATLAB): filter by direction, then select max distance
                            if (!found_nucleation && !lmp_candidates.empty()) {
                                // Check if ANY candidate passes direction threshold
                                T max_dot_all = -2.0;
                                for (const auto& cand : lmp_candidates) {
                                    if (cand.dir_dot > max_dot_all) max_dot_all = cand.dir_dot;
                                }
                                
                                // If best direction passes threshold, select among those passing
                                if (max_dot_all >= thresh_ext) {
                                    T max_d = -1.0;
                                    std::array<int, d> best_pt{};
                                    std::array<T, d> best_dir{};
                                    
                                    for (const auto& cand : lmp_candidates) {
                                        if (cand.dir_dot >= thresh_ext && cand.d_value > max_d) {
                                            max_d = cand.d_value;
                                            best_pt = cand.pt;
                                            
                                            // Recompute direction
                                            std::array<T, d> new_dir{T(cand.pt[0] - p_current[0]), 
                                                                     T(cand.pt[1] - p_current[1])};
                                            T nl = Length(new_dir);
                                            best_dir = {new_dir[0] / nl, new_dir[1] / nl};
                                        }
                                    }
                                    
                                    if (max_d > 0) {
                                        found_next = true;
                                        fibres[i].back().link.push_back(best_pt);
                                        p_current = best_pt;
                                        
                                        // Update direction with decay
                                        dir = {T(1.0 / (1.0 + lambda)) * dir[0] + T(lambda / (1.0 + lambda)) * best_dir[0],
                                               T(1.0 / (1.0 + lambda)) * dir[1] + T(lambda / (1.0 + lambda)) * best_dir[1]};
                                        T dl = Length(dir);
                                        dir = {dir[0] / dl, dir[1] / dl};
                                    }
                                }
                            }
                            
                            if (found_nucleation) {
                                fibres[i].back().link.push_back(nucleation_pt);
                                break;
                            }
                        }
                        fibres[i].back().direction = dir;
                    }
                }
            }
        }

        // Step 2: Remove duplicate fibers using link_map
        const int nNucleation = pts.size();
        std::vector<std::vector<uint64_t>> link_map(nNucleation);

        // Populate link_map
        #pragma omp parallel for
        for (int f = 0; f < (int)fibres.size(); ++f) {
            for (int branch = 0; branch < (int)fibres[f].size(); ++branch) {
                if (fibres[f][branch].link.size() < 2) continue;
                
                uint64_t offset_begin = fibres[f][branch].link[0][0] * sizex + fibres[f][branch].link[0][1];
                uint64_t offset_end = fibres[f][branch].link.back()[0] * sizex + fibres[f][branch].link.back()[1];
                
                if (nucleation_map[offset_end]) {
                    if (offset_begin < offset_end) {
                        bool found = false;
                        #pragma omp critical
                        {
                            for (size_t i = 0; i < link_map[nucleation_map[offset_begin] - 1].size(); ++i) {
                                if (link_map[nucleation_map[offset_begin] - 1][i] == offset_end) {
                                    found = true;
                                    break;
                                }
                            }
                            if (found) {
                                fibres[f][branch].link.clear();
                            } else {
                                link_map[nucleation_map[offset_begin] - 1].push_back(offset_end);
                            }
                        }
                    }
                }
            }
        }

        // Use link_map to delete duplicated fibers (reverse direction)
        #pragma omp parallel for
        for (int f = 0; f < (int)fibres.size(); ++f) {
            for (int branch = 0; branch < (int)fibres[f].size(); ++branch) {
                if (fibres[f][branch].link.size() < 2) continue;
                
                uint64_t offset_begin = fibres[f][branch].link[0][0] * sizex + fibres[f][branch].link[0][1];
                uint64_t offset_end = fibres[f][branch].link.back()[0] * sizex + fibres[f][branch].link.back()[1];
                
                if (nucleation_map[offset_end]) {
                    if (offset_begin > offset_end) {
                        bool found = false;
                        for (size_t i = 0; i < link_map[nucleation_map[offset_end] - 1].size(); ++i) {
                            if (link_map[nucleation_map[offset_end] - 1][i] == offset_begin) {
                                found = true;
                                break;
                            }
                        }
                        if (found) {
                            fibres[f][branch].link.clear();
                        }
                    }
                }
            }
        }

        // Step 3: Populate index_map and X array
        int ncounter = 0;
        for (int f = 0; f < (int)fibres.size(); ++f) {
            for (int branch = 0; branch < (int)fibres[f].size(); ++branch) {
                for (int node = 0; node < (int)fibres[f][branch].link.size(); ++node) {
                    const uint64_t offset = fibres[f][branch].link[node][0] * sizex + 
                                           fibres[f][branch].link[node][1];
                    if (!index_map[offset]) {
                        index_map[offset] = ++ncounter;
                        // Convert to 1-based indexing for output
                        X.push_back(std::array<int, d>{fibres[f][branch].link[node][0] + 1, 
                                                       fibres[f][branch].link[node][1] + 1});
                    }
                }
            }
        }

        // Step 4: Copy radius values (DSM values at vertex locations)
        R.resize(X.size());
        #pragma omp parallel for
        for (int i = 0; i < (int)X.size(); ++i) {
            // Store actual DSM value, not ceil (ceil is only for search radius during extension)
            R[i] = image[(X[i][0] - 1) * sizex + (X[i][1] - 1)];
        }

        // Step 5: Prepare nucleation points for linking (0-based indexing)
        std::vector<int> nucleation_pts(nNucleation);
        #pragma omp parallel for
        for (int i = 0; i < nNucleation; ++i) {
            nucleation_pts[i] = index_map[pts[i][0] * sizex + pts[i][1]] - 1;
        }

        // Step 6: Initialize F_init with fiber segments
        std::vector<Fibre_Type> F_init;
        std::vector<uint64_t> branch_accum(pts.size() + 1);
        branch_accum[0] = 0;
        
        for (int f = 0; f < (int)fibres.size(); ++f) {
            int fibre_count = 0;
            for (int b = 0; b < (int)fibres[f].size(); ++b) {
                if (fibres[f][b].link.size() > 0) ++fibre_count;
            }
            branch_accum[f + 1] = branch_accum[f] + fibre_count;
        }
        
        F_init.resize(branch_accum[pts.size()]);
        
        #pragma omp parallel for
        for (int f = 0; f < (int)fibres.size(); ++f) {
            int branch_count = 0;
            for (int branch = 0; branch < (int)fibres[f].size(); ++branch) {
                for (int node = 0; node < (int)fibres[f][branch].link.size(); ++node) {
                    const uint64_t offset = fibres[f][branch].link[node][0] * sizex + 
                                           fibres[f][branch].link[node][1];
                    F_init[branch_accum[f] + branch_count].link_index.push_back(index_map[offset] - 1);
                    F_init[branch_accum[f] + branch_count].link.push_back(fibres[f][branch].link[node]);
                }
                if (fibres[f][branch].link.size() > 0) {
                    F_init[branch_accum[f] + branch_count].direction = fibres[f][branch].direction;
                    ++branch_count;
                }
            }
        }

        // Step 7: Link fibers at nucleation points
        LinkFibreAtNucleationPoint<T, d>(X.size(), nucleation_pts, F_init, F, thresh_linka, sp);

        std::cout << "Fiber segments: " << F_init.size() << std::endl;
        std::cout << "Linked fibers: " << F.size() << std::endl;

        // Step 8: Build auxiliary data structures (Xfe, Xf, Xvall, Ff)
        Xfe.resize(X.size());
        Xf.resize(X.size());
        Xvall.resize(X.size());
        
        for (int f = 0; f < (int)F.size(); ++f) {
            if (F[f].size()) {
                const int index_begin = F[f][0];
                const int index_end = F[f].back();
                
                if (index_begin - 1 >= (int)X.size() || index_end - 1 >= (int)X.size()) {
                    std::cerr << "Error: Index out of bounds in Xfe" << std::endl;
                    continue;
                }
                
                Xfe[index_begin - 1].push_back(f + 1);
                Xfe[index_end - 1].push_back(f + 1);
            }
            
            for (int node = 0; node < (int)F[f].size(); ++node) {
                Xf[F[f][node] - 1].push_back(f + 1);
                for (int node2 = 0; node2 < (int)F[f].size(); ++node2) {
                    Xvall[F[f][node] - 1].push_back(F[f][node2] - 1);
                }
            }
        }

        // Step 9: Build Ff (fibers connected to each fiber)
        Ff.resize(F.size());
        #pragma omp parallel for
        for (int f = 0; f < (int)F.size(); ++f) {
            for (int node = 0; node < (int)F[f].size(); ++node) {
                for (int f2 = 0; f2 < (int)Xf[F[f][node] - 1].size(); ++f2) {
                    if (Xf[F[f][node] - 1][f2] != f + 1) {
                        bool unique = true;
                        for (int ff = 0; ff < (int)Ff[f].size(); ++ff) {
                            if (Ff[f][ff] == Xf[F[f][node] - 1][f2]) {
                                unique = false;
                                break;
                            }
                        }
                        if (unique) Ff[f].push_back(Xf[F[f][node] - 1][f2]);
                    }
                }
            }
        }

        delete[] index_map;
        delete[] nucleation_map;
    }

    // 3D Constructor stub
    explicit ExtendXLink(int sizex, int sizey, int sizez, T* image,
                         const std::vector<std::array<int, d>>& pts,
                         int thresh_LMPdist, T thresh_LMP, T thresh_ext, T lambda,
                         T thresh_linkd, T thresh_linka, int sp,
                         std::vector<std::array<int, d>>& X, std::vector<T>& R,
                         std::vector<std::vector<int>>& F,
                         std::vector<std::vector<int>>& Xfe, std::vector<std::vector<int>>& Xf,
                         std::vector<std::vector<int>>& Xvall, std::vector<std::vector<int>>& Ff) {
        static_assert(d == 3, "3D Implementation");
        // TODO: Implement 3D version
        throw std::runtime_error("3D ExtendXLink not yet implemented");
    }
};

// Python Wrapper
py::tuple extend_xlink_native(int sizex, int sizey, int sizez,
                               py::array_t<float, py::array::c_style> image,
                               py::array_t<int32_t> pts_in,
                               py::dict p) {

    float* img_ptr = image.mutable_data();
    int nNuc = pts_in.shape(0);
    
    std::vector<std::array<int, 2>> pts(nNuc);
    // Convert from 1-based (MATLAB) to 0-based (C++) indexing
    for (int i = 0; i < nNuc; ++i) {
        pts[i] = {pts_in.at(i, 0) - 1, pts_in.at(i, 1) - 1};
    }

    std::vector<std::array<int, 2>> X;
    std::vector<float> R;
    std::vector<std::vector<int>> F, Xfe, Xf, Xvall, Ff;

    // Extract parameters while holding GIL
    int thresh_LMPdist = p["thresh_LMPdist"].cast<int>();
    float thresh_LMP = p["thresh_LMP"].cast<float>();
    float thresh_ext = p["thresh_ext"].cast<float>();
    float lam_dirdecay = p["lam_dirdecay"].cast<float>();
    float thresh_linkd = p["thresh_linkd"].cast<float>();
    float thresh_linka = p["thresh_linka"].cast<float>();
    int s_fiberdir = std::max(2, p["s_fiberdir"].cast<int>());

    {
        py::gil_scoped_release release;
        // Note: For 2D (sizex==1), pass sizey and sizez as dimensions
        ExtendXLink<float, 2> engine(sizey, sizez, img_ptr, pts,
                                    thresh_LMPdist, thresh_LMP, thresh_ext, 
                                    lam_dirdecay, thresh_linkd, thresh_linka, s_fiberdir,
                                    X, R, F, Xfe, Xf, Xvall, Ff);
    }

    // Convert X to numpy array
    auto py_X = py::array_t<int32_t>({(int)X.size(), 3});
    auto X_ptr = py_X.mutable_data();
    for (int i = 0; i < (int)X.size(); ++i) {
        X_ptr[i * 3 + 0] = X[i][0];
        X_ptr[i * 3 + 1] = X[i][1];
        X_ptr[i * 3 + 2] = 1;
    }

    // Convert F to Python list of dicts
    py::list py_F;
    for (auto& fiber : F) {
        py::dict f_struct;
        f_struct["v"] = py::cast(fiber);
        f_struct["f"] = py::dict();  // Placeholder for Ff
        py_F.append(f_struct);
    }
    
    // Add Ff to each fiber dict
    for (int i = 0; i < (int)Ff.size(); ++i) {
        py_F[i].attr("__setitem__")("f", py::cast(Ff[i]));
    }

    // Convert V (vertices) to Python list of dicts
    py::list py_V;
    for (int i = 0; i < (int)Xfe.size(); ++i) {
        py::dict v_struct;
        v_struct["fe"] = py::cast(Xfe[i]);
        v_struct["f"] = py::cast(Xf[i]);
        v_struct["vall"] = py::cast(Xvall[i]);
        py_V.append(v_struct);
    }

    return py::make_tuple(py_X, py_F, py_V, py::cast(R));
}
