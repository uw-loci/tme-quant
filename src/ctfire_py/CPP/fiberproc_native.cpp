#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

namespace py = pybind11;

// ============================================================================
// Helper Structures and Functions
// ============================================================================

struct Vertex {
    std::vector<int> fe;    // Fiber endpoints at this vertex
    std::vector<int> f;     // All fibers passing through this vertex
    std::vector<int> vall;  // All vertices in connected fibers
};

struct Fiber {
    std::vector<int> v;     // Vertex indices (1-based for Python output)
};

// ============================================================================
// Math Utilities
// ============================================================================

template<typename T>
inline T dot3(const std::array<T, 3>& a, const std::array<T, 3>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

template<typename T>
inline T norm3(const std::array<T, 3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

template<typename T>
inline std::array<T, 3> normalize3(const std::array<T, 3>& v) {
    T len = norm3(v);
    if (len < 1e-10) return {0, 0, 0};
    return {v[0]/len, v[1]/len, v[2]/len};
}

// Calculate Euclidean distance between two points
template<typename T>
inline T distance(const std::array<T, 3>& p1, const std::array<T, 3>& p2) {
    T dx = p1[0] - p2[0];
    T dy = p1[1] - p2[1];
    T dz = p1[2] - p2[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ============================================================================
// getvect: Get fiber direction vector at endpoint
// Based on fiberproc.m lines 74-90
// ============================================================================

std::array<float, 3> getvect(
    const std::vector<std::array<float, 3>>& X,
    int vi,  // Vertex index (0-based)
    const std::vector<int>& fiber,  // 0-based indices
    int sp   // Number of points for direction
) {
    int len = fiber.size();
    std::array<float, 3> vect = {0, 0, 0};
    
    if (fiber[0] == vi) {  // Fiber starts at vertex i
        int ii = std::min(sp, len) - 1;
        int vj = fiber[ii];
        vect = {X[vj][0] - X[vi][0], X[vj][1] - X[vi][1], X[vj][2] - X[vi][2]};
    } else if (fiber[len-1] == vi) {  // Fiber ends at vertex i
        int ii = std::max(0, len - sp);
        int vj = fiber[ii];
        vect = {X[vj][0] - X[vi][0], X[vj][1] - X[vi][1], X[vj][2] - X[vi][2]};
    }
    
    return normalize3(vect);
}

// ============================================================================
// mergefiber: Merge two fibers at their common vertex
// Based on fiberproc.m lines 94-137 and mergefiber.m
// ============================================================================

void mergefiber(
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    int f1,  // Fiber indices (0-based)
    int f2
) {
    const auto& fiber1 = F[f1].v;
    const auto& fiber2 = F[f2].v;
    
    if (fiber1.empty() || fiber2.empty()) return;
    
    std::vector<int> fmerge;
    int vm = -1;  // Middle vertex
    int ve = -1;  // End vertex
    
    // Determine how fibers connect and merge them
    if (fiber1[0] == fiber2[0]) {
        // Both start at same vertex: reverse fiber2, then append fiber1
        vm = fiber1[0];
        ve = fiber2[fiber2.size()-1];
        for (int i = fiber2.size()-1; i > 0; --i) fmerge.push_back(fiber2[i]);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (fiber1[0] == fiber2[fiber2.size()-1]) {
        // fiber1 start == fiber2 end
        vm = fiber1[0];
        ve = fiber2[0];
        for (int i = 0; i < (int)fiber2.size()-1; ++i) fmerge.push_back(fiber2[i]);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (fiber1[fiber1.size()-1] == fiber2[0]) {
        // fiber1 end == fiber2 start
        vm = fiber1[fiber1.size()-1];
        ve = fiber2[fiber2.size()-1];
        for (int v : fiber1) fmerge.push_back(v);
        for (int i = 1; i < (int)fiber2.size(); ++i) fmerge.push_back(fiber2[i]);
    } else if (fiber1[fiber1.size()-1] == fiber2[fiber2.size()-1]) {
        // Both end at same vertex
        vm = fiber1[fiber1.size()-1];
        ve = fiber2[0];
        for (int v : fiber1) fmerge.push_back(v);
        for (int i = fiber2.size()-2; i >= 0; --i) fmerge.push_back(fiber2[i]);
    } else {
        // Fibers don't share an end vertex - shouldn't happen
        return;
    }
    
    // Update fiber lists
    F[f1].v = fmerge;
    F[f2].v.clear();
    
    // Update vertex connectivity
    if (vm >= 0 && vm < (int)V.size()) {
        // Remove both fibers from middle vertex endpoint list
        auto& fe = V[vm].fe;
        fe.erase(std::remove(fe.begin(), fe.end(), f1), fe.end());
        fe.erase(std::remove(fe.begin(), fe.end(), f2), fe.end());
    }
    
    if (ve >= 0 && ve < (int)V.size()) {
        // Remove f2 and add f1 to end vertex
        auto& fe = V[ve].fe;
        fe.erase(std::remove(fe.begin(), fe.end(), f2), fe.end());
        if (std::find(fe.begin(), fe.end(), f1) == fe.end()) {
            fe.push_back(f1);
        }
    }
    
    // Update all vertices in fiber2 to reference fiber1 instead
    for (int vi : fiber2) {
        if (vi >= 0 && vi < (int)V.size()) {
            auto& f_list = V[vi].f;
            auto& vall_list = V[vi].vall;
            
            // Replace f2 with f1
            for (int& fi : f_list) {
                if (fi == f2) fi = f1;
            }
            
            // Remove f2 and add f1 (avoiding duplicates)
            f_list.erase(std::remove(f_list.begin(), f_list.end(), f2), f_list.end());
            if (std::find(f_list.begin(), f_list.end(), f1) == f_list.end()) {
                f_list.push_back(f1);
            }
            
            // Update vall: remove fiber2 vertices, add fiber1 vertices
            for (int v : fiber2) {
                vall_list.erase(std::remove(vall_list.begin(), vall_list.end(), v), vall_list.end());
            }
            for (int v : fiber1) {
                if (std::find(vall_list.begin(), vall_list.end(), v) == vall_list.end()) {
                    vall_list.push_back(v);
                }
            }
        }
    }
}

// ============================================================================
// findclose: Find vertices within distance r of vertex i
// Based on fiberproc/findclose.m
// ============================================================================

std::vector<int> findclose(
    const std::vector<std::array<float, 3>>& X,
    int i,  // Vertex index (0-based)
    float r,  // Search radius
    const std::vector<std::array<int, 3>>& dim  // Image dimensions [K, M, N]
) {
    std::vector<int> iclose;
    
    if (i < 0 || i >= (int)X.size()) return iclose;
    
    const auto& xi = X[i];
    
    // Find all vertices within Euclidean distance r
    for (int j = 0; j < (int)X.size(); ++j) {
        if (j == i) continue;
        
        float dist = distance(xi, X[j]);
        if (dist <= r) {
            iclose.push_back(j);
        }
    }
    
    return iclose;
}

// ============================================================================
// mergefiber_sep: Merge two fibers that don't share an end vertex
// Based on fiberproc.m mergefiber_sep (lines 139-163)
// Concatenates fiber1 and fiber2 end-to-end according to which ends connect.
// e1, e2 in {0, 1}: 0 = fiber start, 1 = fiber end (0-based analog of MATLAB 1/2).
// After merging, F[f1] contains both fibers; F[f2] is left empty.
// ============================================================================

void mergefiber_sep(
    std::vector<Fiber>& F,
    int f1, int e1,
    int f2, int e2
) {
    if (f1 < 0 || f1 >= (int)F.size() || f2 < 0 || f2 >= (int)F.size()) return;
    if (f1 == f2) return;
    if (F[f1].v.empty() || F[f2].v.empty()) return;

    const auto& fiber1 = F[f1].v;
    const auto& fiber2 = F[f2].v;
    std::vector<int> fmerge;
    fmerge.reserve(fiber1.size() + fiber2.size());

    if (e1 == 0 && e2 == 0) {
        // Both connect at their starts: reverse(fiber2) ++ fiber1
        for (auto it = fiber2.rbegin(); it != fiber2.rend(); ++it) fmerge.push_back(*it);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (e1 == 0 && e2 == 1) {
        // fiber1 start, fiber2 end: fiber2 ++ fiber1
        for (int v : fiber2) fmerge.push_back(v);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (e1 == 1 && e2 == 0) {
        // fiber1 end, fiber2 start: fiber1 ++ fiber2
        for (int v : fiber1) fmerge.push_back(v);
        for (int v : fiber2) fmerge.push_back(v);
    } else { // e1 == 1 && e2 == 1
        // Both connect at their ends: fiber1 ++ reverse(fiber2)
        for (int v : fiber1) fmerge.push_back(v);
        for (auto it = fiber2.rbegin(); it != fiber2.rend(); ++it) fmerge.push_back(*it);
    }

    F[f1].v = std::move(fmerge);
    F[f2].v.clear();
}

// ============================================================================
// fiberlink: Link fibers at endpoints if they have similar orientation
// Based on fiberproc.m lines 263-332
// ============================================================================

void fiberlink(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    float thresha,  // Angle threshold (cosine, e.g., -0.866 for ~150 degrees)
    int sp          // Number of points for direction calculation
) {
    for (int i = 0; i < (int)F.size(); ++i) {
        if (F[i].v.size() < 2) continue;
        
        // Check both endpoints of fiber i
        std::vector<int> vrange = {F[i].v[0], F[i].v[(int)F[i].v.size()-1]};
        
        for (int vi : vrange) {
            if (vi < 0 || vi >= (int)V.size()) continue;
            
            const auto& fe = V[vi].fe;  // Fibers ending at this vertex
            int n = fe.size();
            
            if (n == 2) {
                // Exactly 2 fibers meet: check if they should be merged
                int fj = fe[0];
                int fk = fe[1];
                
                if (fj < 0 || fj >= (int)F.size() || fk < 0 || fk >= (int)F.size()) continue;
                if (F[fj].v.empty() || F[fk].v.empty()) continue;
                
                auto vect1 = getvect(X, vi, F[fj].v, sp);
                auto vect2 = getvect(X, vi, F[fk].v, sp);
                float a = dot3(vect1, vect2);
                
                if (a < thresha) {
                    mergefiber(F, V, fj, fk);
                }
            } else if (n > 2) {
                // More than 2 fibers meet: find best pair to merge
                for (int j = 0; j < n-1; ++j) {
                    int fj = fe[j];
                    if (fj < 0 || fj >= (int)F.size() || F[fj].v.empty()) continue;
                    
                    auto vect1 = getvect(X, vi, F[fj].v, sp);
                    
                    for (int k = j+1; k < n; ++k) {
                        int fk = fe[k];
                        if (fk < 0 || fk >= (int)F.size() || F[fk].v.empty()) continue;
                        
                        auto vect2 = getvect(X, vi, F[fk].v, sp);
                        float a = dot3(vect1, vect2);
                        
                        if (a < thresha) {
                            mergefiber(F, V, fj, fk);
                            break;  // Only merge one pair per iteration
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// fiberlinkgap: Fuse fibers end-to-end across gaps if they have similar
// orientation. Faithful port of fiberproc.m lines 395-529.
//
// For every fiber endpoint, look for nearby endpoints of other fibers that are
// not already directly connected, score each candidate pair with
// max(dot(dir_fi, dir_fek), dot(dir_fi, (xk - xj)/|xk - xj|)) (smaller = more
// anti-parallel = better match), and fuse the best-scoring pair via
// mergefiber_sep when the score is below thresh_linka. Each fiber end may be
// consumed at most once per invocation.
// ============================================================================

void fiberlinkgap(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    const std::array<int, 3>& sized,
    int sp,             // anglecomp_space (p.s_fiberdir)
    float thresh_linkd, // max gap distance
    float thresh_linka  // max score to accept a fuse (cosine, typically negative)
) {
    const int nF = (int)F.size();
    const int nV = (int)V.size();

    int nFibersInitial = 0;
    for (const auto& f : F) if (!f.v.empty()) nFibersInitial++;

    // Step 1: cache fiber endpoint positions and direction vectors.
    // dir[i][0] points from fiber start into the fiber; dir[i][1] points from
    // fiber end into the fiber (matches fiberproc.m getvect convention).
    std::vector<std::array<std::array<float, 3>, 2>> fiber_dir(nF);
    std::vector<std::array<std::array<float, 3>, 2>> fiber_pos(nF);

    for (int fi = 0; fi < nF; ++fi) {
        if (F[fi].v.size() < 2) continue;
        int v1 = F[fi].v.front();
        int v2 = F[fi].v.back();
        if (v1 < 0 || v1 >= (int)X.size() || v2 < 0 || v2 >= (int)X.size()) continue;
        fiber_pos[fi][0] = X[v1];
        fiber_pos[fi][1] = X[v2];
        fiber_dir[fi][0] = getvect(X, v1, F[fi].v, sp);
        fiber_dir[fi][1] = getvect(X, v2, F[fi].v, sp);
    }

    // Step 2: for each fiber/end, find the best fuse partner and record it.
    struct FuseEntry { int f1; int e1; int f2; int e2; };
    std::vector<FuseEntry> fuse;
    std::vector<std::array<uint8_t, 2>> fuseflag(nF, {0, 0});

    // Dim argument kept only to match the current findclose signature.
    std::vector<std::array<int, 3>> dim_vec = {{sized[0], sized[1], sized[2]}};

    for (int fi = 0; fi < nF; ++fi) {
        if (F[fi].v.size() < 2) continue;

        // Build the set of fibers that share any vertex with fi so we can
        // exclude direct neighbors (analog of MATLAB's fconnect).
        std::unordered_set<int> fconnect;
        for (int vi : F[fi].v) {
            if (vi < 0 || vi >= nV) continue;
            for (int ff : V[vi].f) fconnect.insert(ff);
        }

        for (int j = 0; j < 2; ++j) {
            if (fuseflag[fi][j]) continue;

            int vj = (j == 0) ? F[fi].v.front() : F[fi].v.back();
            if (vj < 0 || vj >= (int)X.size()) continue;

            const auto& xj = fiber_pos[fi][j];
            const auto& vectj = fiber_dir[fi][j];

            auto vclose = findclose(X, vj, thresh_linkd, dim_vec);
            if (vclose.empty()) continue;

            float best_score = std::numeric_limits<float>::infinity();
            int best_fk = -1;
            int best_l = -1;

            for (int vk : vclose) {
                if (vk < 0 || vk >= nV) continue;
                for (int fek : V[vk].fe) {
                    if (fek <= fi) continue;                   // check each pair once
                    if (fconnect.count(fek) > 0) continue;      // skip direct neighbors
                    if (fek >= nF || F[fek].v.empty()) continue;

                    int m;
                    if (vk == F[fek].v.front()) m = 0;
                    else if (vk == F[fek].v.back()) m = 1;
                    else continue;                              // vk must be an end of fek

                    // Note: MATLAB does NOT skip already-flagged candidates during
                    // scoring; it picks the best candidate and then rejects via the
                    // fuseflag check below. Do not add an early skip here.

                    float d1 = dot3(vectj, fiber_dir[fek][m]);

                    const auto& xk = X[vk];
                    std::array<float, 3> xkj = {xk[0] - xj[0], xk[1] - xj[1], xk[2] - xj[2]};
                    float nkj = norm3(xkj);
                    float d2;
                    if (nkj < 1e-10f) {
                        // coincident endpoints - treat as poorly informative
                        d2 = 1.0f;
                    } else {
                        std::array<float, 3> xkj_n = {xkj[0]/nkj, xkj[1]/nkj, xkj[2]/nkj};
                        d2 = dot3(vectj, xkj_n);
                    }

                    float score = std::max(d1, d2);
                    if (score < best_score) {
                        best_score = score;
                        best_fk = fek;
                        best_l = m;
                    }
                }
            }

            if (best_fk >= 0 && best_score < thresh_linka) {
                if (fuseflag[fi][j] == 0 && fuseflag[best_fk][best_l] == 0) {
                    fuseflag[fi][j] = 1;
                    fuseflag[best_fk][best_l] = 1;
                    fuse.push_back({fi, j, best_fk, best_l});
                }
            }
        }
    }

    // Step 3: perform the fuses, remapping subsequent entries when a fiber
    // index is absorbed. See fiberproc.m lines 488-517 for the remap logic.
    int n_fused = 0;
    for (int i = 0; i < (int)fuse.size(); ++i) {
        int f1 = fuse[i].f1, e1 = fuse[i].e1;
        int f2 = fuse[i].f2, e2 = fuse[i].e2;

        if (f1 == f2) continue;
        if (f1 < 0 || f1 >= (int)F.size() || f2 < 0 || f2 >= (int)F.size()) continue;
        if (F[f1].v.empty() || F[f2].v.empty()) continue;

        mergefiber_sep(F, f1, e1, f2, e2);
        n_fused++;

        // The remaining free end of the absorbed fiber f2 now lives at merged
        // end e1 of f1 (derivation: see mergefiber_sep cases).
        for (int k = i + 1; k < (int)fuse.size(); ++k) {
            if (fuse[k].f1 == f2) { fuse[k].f1 = f1; fuse[k].e1 = e1; }
            if (fuse[k].f2 == f2) { fuse[k].f2 = f1; fuse[k].e2 = e1; }
        }
    }

    int nFibersFinal = 0;
    for (const auto& f : F) if (!f.v.empty()) nFibersFinal++;

    std::cout << "  fiberlinkgap: fused " << n_fused << " pairs ("
              << nFibersInitial << " -> " << nFibersFinal << " fibers)" << std::endl;
}

// ============================================================================
// Main fiberproc_native function
// Based on fiberproc.m lines 1-60
// ============================================================================

py::tuple fiberproc_native(
    int sizex, int sizey, int sizez,
    py::array_t<float, py::array::c_style> image,
    py::array_t<int32_t> X_in,
    py::list F_in,
    py::array_t<float> R_in,
    py::dict p
) {
    std::cout << "fiberproc_native: Processing fibers..." << std::endl;
    std::cout << "Image size: " << sizex << " x " << sizey << " x " << sizez << std::endl;
    
    // image is accepted for signature compatibility with MATLAB's
    // fiberproc(X, F, R, size(dsm), p) but fiberproc itself no longer needs the
    // distance-transform pixels (the fuse step is purely geometric).
    (void)image;
    const int nX = X_in.shape(0);
    
    // Extract parameters
    float thresh_linka = p["thresh_linka"].cast<float>();
    int s_fiberdir = p["s_fiberdir"].cast<int>();
    float thresh_linkd = p.contains("thresh_linkd") ? p["thresh_linkd"].cast<float>() : 15.0f;
    
    // Convert X to internal format (0-based indexing)
    std::vector<std::array<float, 3>> X(nX);
    for (int i = 0; i < nX; ++i) {
        X[i] = {
            (float)(X_in.at(i, 0) - 1),  // Convert to 0-based
            (float)(X_in.at(i, 1) - 1),
            (float)(X_in.at(i, 2) - 1)
        };
    }
    
    // Convert F from Python to internal format (0-based indexing)
    std::vector<Fiber> F;
    for (int i = 0; i < F_in.size(); ++i) {
        py::dict fiber_dict = F_in[i];
        Fiber fib;
        
        if (fiber_dict.contains("v")) {
            std::vector<int> v_list = fiber_dict["v"].cast<std::vector<int>>();
            for (int vi : v_list) {
                fib.v.push_back(vi - 1);  // Convert to 0-based
            }
        }
        
        F.push_back(fib);
    }
    
    // Build vertex connectivity structure V
    std::vector<Vertex> V(nX);
    for (int fi = 0; fi < (int)F.size(); ++fi) {
        if (F[fi].v.size() < 2) continue;
        
        // Mark endpoints
        int v_start = F[fi].v[0];
        int v_end = F[fi].v[F[fi].v.size()-1];
        
        if (v_start >= 0 && v_start < nX) {
            V[v_start].fe.push_back(fi);
        }
        if (v_end >= 0 && v_end < nX) {
            V[v_end].fe.push_back(fi);
        }
        
        // Mark all vertices in fiber
        for (int vi : F[fi].v) {
            if (vi >= 0 && vi < nX) {
                V[vi].f.push_back(fi);
                for (int vj : F[fi].v) {
                    if (std::find(V[vi].vall.begin(), V[vi].vall.end(), vj) == V[vi].vall.end()) {
                        V[vi].vall.push_back(vj);
                    }
                }
            }
        }
    }
    
    std::cout << "  Initial: " << nX << " vertices, " << F.size() << " fibers" << std::endl;
    
    // LINK FIBERS OF LIKE ORIENTATION AT ENDPOINTS (5 iterations)
    // Based on fiberproc.m lines 14-20
    const int NN = 5;
    std::cout << "  Linking fibers (" << NN << " iterations)..." << std::endl;
    
    for (int iter = 0; iter < NN; ++iter) {
        int fibers_before = 0;
        for (const auto& f : F) if (!f.v.empty()) fibers_before++;
        
        fiberlink(X, F, V, thresh_linka, s_fiberdir);
        
        int fibers_after = 0;
        for (const auto& f : F) if (!f.v.empty()) fibers_after++;
        
        std::cout << "    Iteration " << (iter+1) << ": " << fibers_before 
                  << " -> " << fibers_after << " fibers" << std::endl;
    }
    
    // LINK FIBERS ACROSS GAPS (fuse end-to-end)
    // Faithful port of MATLAB fiberproc.m line 25:
    //     [X F V] = fiberlinkgap(X, F, V, size_im, p.s_fiberdir,
    //                            p.thresh_linkd, p.thresh_linka, plotflag);
    std::cout << "  Linking fibers across gaps..." << std::endl;
    std::array<int, 3> sized = {sizez, sizey, sizex};
    fiberlinkgap(X, F, V, sized, s_fiberdir, thresh_linkd, thresh_linka);

    // Rebuild V from the post-fuse F (fiberlinkgap does not maintain V).
    for (auto& v : V) { v.fe.clear(); v.f.clear(); v.vall.clear(); }
    for (int fi = 0; fi < (int)F.size(); ++fi) {
        if (F[fi].v.size() < 2) continue;
        int vs = F[fi].v.front();
        int ve = F[fi].v.back();
        if (vs >= 0 && vs < (int)V.size()) V[vs].fe.push_back(fi);
        if (ve >= 0 && ve < (int)V.size()) V[ve].fe.push_back(fi);
        for (int vi : F[fi].v) {
            if (vi < 0 || vi >= (int)V.size()) continue;
            V[vi].f.push_back(fi);
            for (int vj : F[fi].v) {
                if (std::find(V[vi].vall.begin(), V[vi].vall.end(), vj) == V[vi].vall.end()) {
                    V[vi].vall.push_back(vj);
                }
            }
        }
    }
    
    // BUILD OUTPUT
    // Convert back to 1-based indexing for Python
    
    // X output (only used vertices)
    auto py_X = py::array_t<int32_t>({nX, 3});
    auto X_ptr = py_X.mutable_data();
    for (int i = 0; i < nX; ++i) {
        X_ptr[i*3 + 0] = (int32_t)(X[i][0] + 1);
        X_ptr[i*3 + 1] = (int32_t)(X[i][1] + 1);
        X_ptr[i*3 + 2] = (int32_t)(X[i][2] + 1);
    }
    
    // F output (non-empty fibers, 1-based indices)
    py::list py_F;
    for (const auto& fiber : F) {
        if (fiber.v.empty()) continue;
        
        py::dict f_struct;
        std::vector<int> v_out;
        for (int vi : fiber.v) {
            v_out.push_back(vi + 1);  // Convert to 1-based
        }
        f_struct["v"] = py::cast(v_out);
        py_F.append(f_struct);
    }
    
    // E output (edges)
    py::list py_E;
    for (const auto& fiber : F) {
        if (fiber.v.size() >= 2) {
            py::list edge;
            edge.append(fiber.v[0] + 1);  // Start vertex (1-based)
            edge.append(fiber.v[fiber.v.size()-1] + 1);  // End vertex (1-based)
            py_E.append(edge);
        }
    }
    
    // V output (vertex connectivity, 1-based indices)
    py::list py_V;
    for (const auto& v : V) {
        py::dict v_struct;
        
        std::vector<int> fe_out, f_out, vall_out;
        for (int fi : v.fe) fe_out.push_back(fi + 1);
        for (int fi : v.f) f_out.push_back(fi + 1);
        for (int vi : v.vall) vall_out.push_back(vi + 1);
        
        v_struct["fe"] = py::cast(fe_out);
        v_struct["f"] = py::cast(f_out);
        v_struct["vall"] = py::cast(vall_out);
        py_V.append(v_struct);
    }
    
    // R output (radii - unchanged)
    auto py_R = py::array_t<float>(nX);
    auto R_ptr = py_R.mutable_data();
    for (int i = 0; i < nX; ++i) {
        R_ptr[i] = R_in.at(i);
    }
    
    int final_fiber_count = 0;
    for (const auto& f : F) if (!f.v.empty()) final_fiber_count++;
    
    std::cout << "  Final: " << nX << " vertices, " << final_fiber_count << " fibers" << std::endl;
    std::cout << "fiberproc_native: Complete!" << std::endl;
    
    return py::make_tuple(py_X, py_F, py_E, py_V, py_R);
}
