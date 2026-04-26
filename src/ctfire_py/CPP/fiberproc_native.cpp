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
#include <limits>
#include <unordered_set>
#include <unordered_map>

namespace py = pybind11;

// ============================================================================
// C++ port of MATLAB `fiberproc.m` (curvelets/src/FIRE/fiberproc/fiberproc.m).
//
// Orchestrates:
//   [X F V R] = trimxfv(X, F, [], R);
//   [X F V R] = remove_repeat(X, F, V, R);
//   for i=1:5
//       [X F V R] = fiberlink(X, F, V, R, thresh_linka, s_fiberdir);
//       [X F V R] = remove_repeat(X, F, V, R);
//   end
//   [X F V]   = fiberlinkgap(X, F, V, size_im, s_fiberdir, thresh_linkd,
//                            thresh_linka);
//   [X F V R] = fiberremove(X, F, V, R, thresh_flen, thresh_numv);
// ============================================================================

// ============================================================================
// Structures
// ============================================================================

struct Vertex {
    std::vector<int> fe;    // Fibers that have this vertex as an endpoint
    std::vector<int> f;     // All fibers that pass through this vertex
    std::vector<int> vall;  // All vertices in connected fibers (filled on output)
};

struct Fiber {
    std::vector<int> v;     // 0-based vertex indices into X
    std::vector<int> f;     // 0-based fiber indices this fiber shares a vertex with
    float           len = 0.0f; // physical length (set by fiberremove_cpp)
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

template<typename T>
inline T point_distance(const std::array<T, 3>& p1, const std::array<T, 3>& p2) {
    T dx = p1[0] - p2[0];
    T dy = p1[1] - p2[1];
    T dz = p1[2] - p2[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static float fiber_length(const std::vector<std::array<float, 3>>& X,
                          const std::vector<int>& v) {
    float len = 0.0f;
    for (size_t i = 1; i < v.size(); ++i) {
        int a = v[i-1], b = v[i];
        if (a < 0 || b < 0) continue;
        if (a >= (int)X.size() || b >= (int)X.size()) continue;
        len += point_distance(X[a], X[b]);
    }
    return len;
}

// ============================================================================
// getvect: Fiber direction at an endpoint (fiberproc.m lines 74-90).
// ============================================================================

static std::array<float, 3> getvect(
    const std::vector<std::array<float, 3>>& X,
    int vi,                         // vertex index (0-based)
    const std::vector<int>& fiber,  // 0-based vertex indices
    int sp                          // sample distance along fiber
) {
    int len = (int)fiber.size();
    std::array<float, 3> vect = {0, 0, 0};
    if (len < 2) return vect;

    if (fiber.front() == vi) {
        int ii = std::min(sp, len) - 1;
        int vj = fiber[ii];
        vect = {X[vj][0] - X[vi][0], X[vj][1] - X[vi][1], X[vj][2] - X[vi][2]};
    } else if (fiber.back() == vi) {
        int ii = std::max(0, len - sp);
        int vj = fiber[ii];
        vect = {X[vj][0] - X[vi][0], X[vj][1] - X[vi][1], X[vj][2] - X[vi][2]};
    }
    return normalize3(vect);
}

// ============================================================================
// trimxfv_cpp: compact F (drop v.size()<2), rebuild V, populate F[fi].f.
// Port of fiberproc/trimxfv.m, minus vertex renumbering.
//
// We deliberately do NOT compact X or renumber vertices: X is owned by the
// caller, unused entries are harmless for the downstream algorithms because
// V[unused_v].f is always empty, and keeping indices stable makes reasoning
// about remove_repeat's coincident-vertex hashing trivial.
// ============================================================================

static void trimxfv_cpp(const std::vector<std::array<float, 3>>& X,
                        std::vector<Fiber>& F,
                        std::vector<Vertex>& V) {
    std::vector<Fiber> F_new;
    F_new.reserve(F.size());
    for (auto& f : F) {
        if (f.v.size() >= 2) {
            f.f.clear();
            F_new.push_back(std::move(f));
        }
    }
    F = std::move(F_new);

    V.assign(X.size(), Vertex{});
    for (int fi = 0; fi < (int)F.size(); ++fi) {
        const auto& f = F[fi];
        int v1 = f.v.front();
        int v2 = f.v.back();
        if (v1 >= 0 && v1 < (int)V.size()) V[v1].fe.push_back(fi);
        if (v2 >= 0 && v2 < (int)V.size()) V[v2].fe.push_back(fi);
        for (int vj : f.v) {
            if (vj < 0 || vj >= (int)V.size()) continue;
            V[vj].f.push_back(fi);
        }
    }

    // F[fi].f: fibers that share any vertex with fi, excluding fi itself.
    // MATLAB does NOT deduplicate this list (see trimxfv.m lines 59-67); the
    // multiplicity matters for fiberremove's `length(fconn) == 2` check.
    for (int fi = 0; fi < (int)F.size(); ++fi) {
        std::vector<int>& fconn = F[fi].f;
        for (int vj : F[fi].v) {
            if (vj < 0 || vj >= (int)V.size()) continue;
            for (int fk : V[vj].f) {
                if (fk != fi) fconn.push_back(fk);
            }
        }
    }
}

// ============================================================================
// mergefiber: merge f2 into f1 when they share an endpoint vertex.
// Port of fiberproc.m lines 94-137.
// ============================================================================

static void mergefiber(
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    int f1,
    int f2
) {
    if (f1 == f2) return;
    if (f1 < 0 || f2 < 0 || f1 >= (int)F.size() || f2 >= (int)F.size()) return;
    const auto& fiber1 = F[f1].v;
    const auto& fiber2 = F[f2].v;
    if (fiber1.empty() || fiber2.empty()) return;

    std::vector<int> fmerge;
    int vm = -1;
    int ve = -1;

    if (fiber1.front() == fiber2.front()) {
        vm = fiber1.front();
        ve = fiber2.back();
        for (int i = (int)fiber2.size() - 1; i > 0; --i) fmerge.push_back(fiber2[i]);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (fiber1.front() == fiber2.back()) {
        vm = fiber1.front();
        ve = fiber2.front();
        for (int i = 0; i < (int)fiber2.size() - 1; ++i) fmerge.push_back(fiber2[i]);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (fiber1.back() == fiber2.front()) {
        vm = fiber1.back();
        ve = fiber2.back();
        for (int v : fiber1) fmerge.push_back(v);
        for (int i = 1; i < (int)fiber2.size(); ++i) fmerge.push_back(fiber2[i]);
    } else if (fiber1.back() == fiber2.back()) {
        vm = fiber1.back();
        ve = fiber2.front();
        for (int v : fiber1) fmerge.push_back(v);
        for (int i = (int)fiber2.size() - 2; i >= 0; --i) fmerge.push_back(fiber2[i]);
    } else {
        return;
    }

    std::vector<int> fiber2_copy = fiber2;  // snapshot before we mutate F
    F[f1].v = std::move(fmerge);
    F[f2].v.clear();

    if (vm >= 0 && vm < (int)V.size()) {
        auto& fe = V[vm].fe;
        fe.erase(std::remove(fe.begin(), fe.end(), f1), fe.end());
        fe.erase(std::remove(fe.begin(), fe.end(), f2), fe.end());
    }
    if (ve >= 0 && ve < (int)V.size()) {
        auto& fe = V[ve].fe;
        fe.erase(std::remove(fe.begin(), fe.end(), f2), fe.end());
        if (std::find(fe.begin(), fe.end(), f1) == fe.end()) {
            fe.push_back(f1);
        }
    }

    for (int vi : fiber2_copy) {
        if (vi < 0 || vi >= (int)V.size()) continue;
        auto& f_list = V[vi].f;
        f_list.erase(std::remove(f_list.begin(), f_list.end(), f2), f_list.end());
        if (std::find(f_list.begin(), f_list.end(), f1) == f_list.end()) {
            f_list.push_back(f1);
        }
    }
}

// ============================================================================
// mergefiber_sep: fuse f2 into f1 across a gap (no shared vertex).
// Port of fiberproc.m lines 139-163.
// e1, e2 in {0, 1}: 0 = fiber start, 1 = fiber end (0-based analog of MATLAB 1/2).
// ============================================================================

static void mergefiber_sep(
    std::vector<Fiber>& F,
    int f1, int e1,
    int f2, int e2
) {
    if (f1 == f2) return;
    if (f1 < 0 || f2 < 0 || f1 >= (int)F.size() || f2 >= (int)F.size()) return;
    if (F[f1].v.empty() || F[f2].v.empty()) return;

    const auto& fiber1 = F[f1].v;
    const auto& fiber2 = F[f2].v;
    std::vector<int> fmerge;
    fmerge.reserve(fiber1.size() + fiber2.size());

    if (e1 == 0 && e2 == 0) {
        for (auto it = fiber2.rbegin(); it != fiber2.rend(); ++it) fmerge.push_back(*it);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (e1 == 0 && e2 == 1) {
        for (int v : fiber2) fmerge.push_back(v);
        for (int v : fiber1) fmerge.push_back(v);
    } else if (e1 == 1 && e2 == 0) {
        for (int v : fiber1) fmerge.push_back(v);
        for (int v : fiber2) fmerge.push_back(v);
    } else {
        for (int v : fiber1) fmerge.push_back(v);
        for (auto it = fiber2.rbegin(); it != fiber2.rend(); ++it) fmerge.push_back(*it);
    }

    F[f1].v = std::move(fmerge);
    F[f2].v.clear();
}

// ============================================================================
// findclose: vertices within Euclidean distance r of vertex i.
// Port of fiberproc/findclose.m (we use a brute-force scan; input sizes here
// are modest and this matches what MATLAB does in practice).
// ============================================================================

static std::vector<int> findclose(
    const std::vector<std::array<float, 3>>& X,
    int i,
    float r
) {
    std::vector<int> iclose;
    if (i < 0 || i >= (int)X.size()) return iclose;
    const auto& xi = X[i];
    for (int j = 0; j < (int)X.size(); ++j) {
        if (j == i) continue;
        if (point_distance(xi, X[j]) <= r) iclose.push_back(j);
    }
    return iclose;
}

// ============================================================================
// remove_repeat_cpp: port of fiberproc/remove_repeat.m.
//
// Iteratively:
//   (a) collapse coincident X coordinates (spatial hash on X / max(X) * 1000)
//   (b) drop consecutive duplicate vertices within each fiber
//   (c) drop an endpoint vertex if it also appears in the fiber's interior
//   (d) trimxfv (compacts F, rebuilds V, populates F.f)
//   (e) when two fibers share >=2 vertices, keep one and split the other
//   (f) trimxfv again
// ...until length(F) stops changing.
// ============================================================================

static void remove_repeat_cpp(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    float thresh_emerge = std::numeric_limits<float>::infinity()
) {
    size_t lenold = std::numeric_limits<size_t>::max();
    size_t len = F.size();

    while (lenold != len) {
        lenold = len;

        // (a) Hash X coords, collect pairs with identical buckets.
        float max_X = 0.0f;
        for (const auto& x : X)
            for (int k = 0; k < 3; ++k) max_X = std::max(max_X, x[k]);

        std::vector<std::array<int64_t, 3>> Xr(X.size());
        std::array<int64_t, 3> s = {1, 1, 1};
        if (max_X > 0.0f) {
            for (size_t i = 0; i < X.size(); ++i) {
                for (int k = 0; k < 3; ++k) {
                    int64_t val = (int64_t)std::ceil((double)X[i][k] / (double)max_X * 1000.0);
                    if (val < 1) val = 1;
                    Xr[i][k] = val;
                    if (val > s[k]) s[k] = val;
                }
            }
        } else {
            for (auto& xr : Xr) xr = {1, 1, 1};
        }

        std::vector<std::pair<int64_t, int>> indexed(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            int64_t flat = (Xr[i][0] - 1)
                         + (Xr[i][1] - 1) * s[0]
                         + (Xr[i][2] - 1) * s[0] * s[1];
            indexed[i] = {flat, (int)i};
        }
        std::sort(indexed.begin(), indexed.end());

        std::vector<std::pair<int, int>> same_pairs;
        for (size_t i = 1; i < indexed.size(); ++i) {
            if (indexed[i].first == indexed[i-1].first) {
                same_pairs.push_back({indexed[i-1].second, indexed[i].second});
            }
        }

        for (const auto& pair : same_pairs) {
            int vi = pair.first;
            int vj = pair.second;
            if (vi < 0 || vi >= (int)V.size()) continue;
            std::vector<int> f_list = V[vi].f;  // snapshot per MATLAB semantics
            for (int fj : f_list) {
                if (fj < 0 || fj >= (int)F.size()) continue;
                for (int& vk : F[fj].v) {
                    if (vk == vi) vk = vj;
                }
            }
        }

        // (b) Drop consecutive duplicate vertices.
        for (auto& f : F) {
            if (f.v.size() < 2) continue;
            std::vector<int> new_v;
            new_v.reserve(f.v.size());
            new_v.push_back(f.v[0]);
            for (size_t i = 1; i < f.v.size(); ++i) {
                if (f.v[i] != f.v[i-1]) new_v.push_back(f.v[i]);
            }
            f.v = std::move(new_v);
        }

        // (c) Drop endpoint vertex if it appears in the fiber's interior.
        for (auto& f : F) {
            if (f.v.size() < 2) continue;
            int ve1 = f.v.front();
            bool in_middle = false;
            for (size_t i = 1; i < f.v.size(); ++i) {
                if (f.v[i] == ve1) { in_middle = true; break; }
            }
            if (in_middle && !f.v.empty()) f.v.erase(f.v.begin());
            if (f.v.size() < 2) continue;

            int ve2 = f.v.back();
            in_middle = false;
            for (size_t i = 0; i + 1 < f.v.size(); ++i) {
                if (f.v[i] == ve2) { in_middle = true; break; }
            }
            if (in_middle && !f.v.empty()) f.v.pop_back();
        }

        // (d) trimxfv: rebuild V, drop short fibers, populate F.f.
        trimxfv_cpp(X, F, V);

        // (e) Find pairs of fibers that share >=2 vertices; keep one, split
        //     the other. MATLAB iterates V(ii).f snapshot; we do the same.
        for (int ii = 0; ii < (int)V.size(); ++ii) {
            std::vector<int> f_here = V[ii].f;   // snapshot
            if (f_here.size() < 2) continue;
            for (size_t i = 0; i + 1 < f_here.size(); ++i) {
                int fi = f_here[i];
                if (fi < 0 || fi >= (int)F.size() || F[fi].v.empty()) continue;
                for (size_t j = i + 1; j < f_here.size(); ++j) {
                    int fj = f_here[j];
                    if (fi == fj) continue;
                    if (fj < 0 || fj >= (int)F.size() || F[fj].v.empty()) continue;

                    const auto& vi_list = F[fi].v;
                    const auto& vj_list = F[fj].v;
                    std::unordered_set<int> vi_set(vi_list.begin(), vi_list.end());

                    int min_ij = -1, max_ij = -1;
                    for (int k = 0; k < (int)vj_list.size(); ++k) {
                        if (vi_set.count(vj_list[k])) {
                            if (min_ij < 0) min_ij = k;
                            max_ij = k;
                        }
                    }
                    if (min_ij < 0 || max_ij <= min_ij) continue;  // need >=2 shared

                    int v1 = vj_list[min_ij];
                    int v2 = vj_list[max_ij];
                    float dist = point_distance(X[v1], X[v2]);
                    if (dist >= thresh_emerge) continue;

                    // MATLAB: if min(ij)>1  => preserve vj(1:min(ij)) as new fiber
                    if (min_ij > 0) {
                        Fiber nf;
                        nf.v.assign(vj_list.begin(), vj_list.begin() + min_ij + 1);
                        F.push_back(std::move(nf));
                    }
                    // MATLAB: if max(ij)<length(vj) => preserve vj(max(ij):end)
                    if (max_ij < (int)vj_list.size() - 1) {
                        Fiber nf;
                        nf.v.assign(vj_list.begin() + max_ij, vj_list.end());
                        F.push_back(std::move(nf));
                    }
                    F[fj].v.clear();
                }
            }
        }

        // (f) trimxfv again.
        trimxfv_cpp(X, F, V);
        len = F.size();
    }
}

// ============================================================================
// fiberlink: link fibers that meet at a vertex with near-opposite orientation.
// Port of fiberproc.m lines 263-343. The n > 2 branch implements MATLAB's
// `min2(A)` optimal strategy: compute full pairwise cosine matrix, repeatedly
// merge the globally-most-anti-parallel pair, zero out the merged rows/cols,
// and continue while the running minimum is below thresha.
// ============================================================================

static void fiberlink(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    float thresha,
    int sp
) {
    for (int i = 0; i < (int)F.size(); ++i) {
        if (F[i].v.size() < 2) continue;

        std::array<int, 2> vrange = {F[i].v.front(), F[i].v.back()};
        for (int vi : vrange) {
            if (vi < 0 || vi >= (int)V.size()) continue;

            std::vector<int> fe = V[vi].fe;     // snapshot
            int n = (int)fe.size();

            if (n == 2) {
                int fj = fe[0], fk = fe[1];
                if (fj < 0 || fj >= (int)F.size() || F[fj].v.empty()) continue;
                if (fk < 0 || fk >= (int)F.size() || F[fk].v.empty()) continue;
                auto vect1 = getvect(X, vi, F[fj].v, sp);
                auto vect2 = getvect(X, vi, F[fk].v, sp);
                float a = dot3(vect1, vect2);
                if (a < thresha) {
                    mergefiber(F, V, fj, fk);
                }
            } else if (n > 2) {
                // Compute pairwise cosine matrix A (upper triangle), mirroring
                // MATLAB's lazy-allocation where lower triangle stays 0.
                std::vector<std::vector<float>> A(n, std::vector<float>(n, 0.0f));
                std::vector<std::array<float, 3>> vects(n, std::array<float, 3>{0, 0, 0});
                std::vector<uint8_t> valid(n, 0);
                for (int j = 0; j < n; ++j) {
                    int fj = fe[j];
                    if (fj < 0 || fj >= (int)F.size() || F[fj].v.empty()) continue;
                    vects[j] = getvect(X, vi, F[fj].v, sp);
                    valid[j] = 1;
                }
                for (int j = 0; j < n - 1; ++j) {
                    if (!valid[j]) continue;
                    for (int k = j + 1; k < n; ++k) {
                        if (!valid[k]) continue;
                        A[j][k] = dot3(vects[j], vects[k]);
                    }
                }

                // Repeatedly merge the most anti-parallel pair. Zeroed rows/cols
                // cannot re-enter the min since 0 >= thresha (thresha < 0).
                while (true) {
                    float a_min = std::numeric_limits<float>::infinity();
                    int l = -1, m = -1;
                    for (int j = 0; j < n; ++j) {
                        for (int k = 0; k < n; ++k) {
                            if (A[j][k] < a_min) {
                                a_min = A[j][k];
                                l = j; m = k;
                            }
                        }
                    }
                    if (a_min >= thresha) break;
                    if (l < 0 || m < 0) break;

                    int f1 = fe[l], f2 = fe[m];
                    if (f1 >= 0 && f2 >= 0 && f1 < (int)F.size() && f2 < (int)F.size()
                        && !F[f1].v.empty() && !F[f2].v.empty()) {
                        mergefiber(F, V, f1, f2);
                    }
                    for (int j = 0; j < n; ++j) {
                        A[l][j] = 0.0f; A[m][j] = 0.0f;
                        A[j][l] = 0.0f; A[j][m] = 0.0f;
                    }
                }
            }
        }
    }

    trimxfv_cpp(X, F, V);
}

// ============================================================================
// fiberlinkgap: fuse fibers across small gaps when orientations align.
// Port of fiberproc.m lines 395-529.
// ============================================================================

static void fiberlinkgap(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    int sp,
    float thresh_linkd,
    float thresh_linka
) {
    const int nF = (int)F.size();
    const int nV = (int)V.size();

    int nFibersInitial = 0;
    for (const auto& f : F) if (!f.v.empty()) nFibersInitial++;

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

    struct FuseEntry { int f1; int e1; int f2; int e2; };
    std::vector<FuseEntry> fuse;
    std::vector<std::array<uint8_t, 2>> fuseflag(nF, {0, 0});

    for (int fi = 0; fi < nF; ++fi) {
        if (F[fi].v.size() < 2) continue;

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

            auto vclose = findclose(X, vj, thresh_linkd);
            if (vclose.empty()) continue;

            float best_score = std::numeric_limits<float>::infinity();
            int best_fk = -1;
            int best_l = -1;

            for (int vk : vclose) {
                if (vk < 0 || vk >= nV) continue;
                for (int fek : V[vk].fe) {
                    if (fek <= fi) continue;
                    if (fconnect.count(fek) > 0) continue;
                    if (fek >= nF || F[fek].v.empty()) continue;

                    int m;
                    if (vk == F[fek].v.front()) m = 0;
                    else if (vk == F[fek].v.back()) m = 1;
                    else continue;

                    float d1 = dot3(vectj, fiber_dir[fek][m]);

                    const auto& xk = X[vk];
                    std::array<float, 3> xkj = {xk[0] - xj[0], xk[1] - xj[1], xk[2] - xj[2]};
                    float nkj = norm3(xkj);
                    float d2;
                    if (nkj < 1e-10f) {
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

    int n_fused = 0;
    for (int i = 0; i < (int)fuse.size(); ++i) {
        int f1 = fuse[i].f1, e1 = fuse[i].e1;
        int f2 = fuse[i].f2, e2 = fuse[i].e2;

        if (f1 == f2) continue;
        if (f1 < 0 || f1 >= (int)F.size() || f2 < 0 || f2 >= (int)F.size()) continue;
        if (F[f1].v.empty() || F[f2].v.empty()) continue;

        mergefiber_sep(F, f1, e1, f2, e2);
        n_fused++;

        for (int k = i + 1; k < (int)fuse.size(); ++k) {
            if (fuse[k].f1 == f2) { fuse[k].f1 = f1; fuse[k].e1 = e1; }
            if (fuse[k].f2 == f2) { fuse[k].f2 = f1; fuse[k].e2 = e1; }
        }
    }

    // MATLAB fiberlinkgap ends with `trimxfv(X, F, V)`.
    trimxfv_cpp(X, F, V);

    int nFibersFinal = 0;
    for (const auto& f : F) if (!f.v.empty()) nFibersFinal++;

    std::cout << "  fiberlinkgap: fused " << n_fused << " pairs ("
              << nFibersInitial << " -> " << nFibersFinal << " fibers)" << std::endl;
}

// ============================================================================
// fiberremove_cpp: remove short, dangling, or short-parallel fibers.
// Port of fiberproc/fiberremove.m.
// ============================================================================

static void fiberremove_cpp(
    std::vector<std::array<float, 3>>& X,
    std::vector<Fiber>& F,
    std::vector<Vertex>& V,
    float thresh_flen,
    int thresh_numv
) {
    (void)thresh_numv;  // MATLAB comments out the numv guard; parity with that.

    for (auto& f : F) {
        f.len = fiber_length(X, f.v);
    }

    std::set<int> fremove;

    for (int fi = (int)F.size() - 1; fi >= 0; --fi) {
        if (F[fi].v.size() < 2) continue;
        if (F[fi].len > thresh_flen) continue;

        // Count unique vertices on fi with degree > 1 (where degree = V[vi].f.size()).
        std::set<int> vconn;
        for (int vi : F[fi].v) {
            if (vi < 0 || vi >= (int)V.size()) continue;
            if (V[vi].f.size() > 1) vconn.insert(vi);
        }
        if (vconn.size() <= 1) {
            fremove.insert(fi);
        }

        // MATLAB also drops short "two short fibers running along a 3rd fiber" patterns.
        const auto& fconn = F[fi].f;
        if (fconn.size() == 2) {
            int f2 = fconn[0];
            int f3 = fconn[1];
            if (f2 < 0 || f2 >= (int)F.size() || f3 < 0 || f3 >= (int)F.size()) continue;

            const auto& fconn2 = F[f2].f;
            const auto& fconn3 = F[f3].f;

            bool f3_in_fconn2 = std::find(fconn2.begin(), fconn2.end(), f3) != fconn2.end();
            bool f2_in_fconn3 = std::find(fconn3.begin(), fconn3.end(), f2) != fconn3.end();

            if (fconn2.size() == 2 && f3_in_fconn2 && F[f2].len <= thresh_flen) {
                fremove.insert(fi);
                fremove.insert(f2);
            } else if (fconn3.size() == 2 && f2_in_fconn3 && F[f3].len <= thresh_flen) {
                fremove.insert(fi);
                fremove.insert(f3);
            }
        }
    }

    for (int fi : fremove) {
        if (fi >= 0 && fi < (int)F.size()) F[fi].v.clear();
    }

    trimxfv_cpp(X, F, V);
}

// ============================================================================
// Main entry: fiberproc_native - MATLAB-faithful fiberproc.m pipeline.
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

    (void)image;  // fiberproc.m no longer uses dsm pixels once fiber_gapfill is gone.
    (void)sizez;

    const int nX = X_in.shape(0);

    float thresh_linka = p["thresh_linka"].cast<float>();
    int s_fiberdir = p["s_fiberdir"].cast<int>();
    float thresh_linkd = p.contains("thresh_linkd") ? p["thresh_linkd"].cast<float>() : 15.0f;
    float thresh_flen = p.contains("thresh_flen") ? p["thresh_flen"].cast<float>() : 15.0f;
    int thresh_numv = p.contains("thresh_numv") ? p["thresh_numv"].cast<int>() : 3;

    // Convert X to internal format (0-based float coords).
    std::vector<std::array<float, 3>> X(nX);
    for (int i = 0; i < nX; ++i) {
        X[i] = {
            (float)(X_in.at(i, 0) - 1),
            (float)(X_in.at(i, 1) - 1),
            (float)(X_in.at(i, 2) - 1)
        };
    }

    // Convert F from Python to internal format (0-based vertex indices).
    std::vector<Fiber> F;
    F.reserve(F_in.size());
    for (int i = 0; i < (int)F_in.size(); ++i) {
        py::dict fiber_dict = F_in[i];
        Fiber fib;
        if (fiber_dict.contains("v")) {
            std::vector<int> v_list = fiber_dict["v"].cast<std::vector<int>>();
            fib.v.reserve(v_list.size());
            for (int vi : v_list) fib.v.push_back(vi - 1);
        }
        F.push_back(std::move(fib));
    }

    std::vector<Vertex> V;

    // --- Pipeline, matching fiberproc.m exactly. ---
    std::cout << "  initial trimxfv + remove_repeat..." << std::endl;
    trimxfv_cpp(X, F, V);
    remove_repeat_cpp(X, F, V);
    std::cout << "    after initial cleanup: " << F.size() << " fibers" << std::endl;

    const int NN = 5;
    std::cout << "  linking fibers (" << NN << " iterations, fiberlink + remove_repeat)..." << std::endl;
    for (int iter = 0; iter < NN; ++iter) {
        size_t before = F.size();
        fiberlink(X, F, V, thresh_linka, s_fiberdir);
        size_t mid = F.size();
        remove_repeat_cpp(X, F, V);
        std::cout << "    iter " << (iter + 1) << ": fiberlink "
                  << before << " -> " << mid
                  << ", remove_repeat " << mid << " -> " << F.size()
                  << " fibers" << std::endl;
    }

    std::cout << "  linking fibers across gaps..." << std::endl;
    fiberlinkgap(X, F, V, s_fiberdir, thresh_linkd, thresh_linka);

    std::cout << "  fiberremove (short / dangling fibers)..." << std::endl;
    size_t before_fr = F.size();
    fiberremove_cpp(X, F, V, thresh_flen, thresh_numv);
    std::cout << "    fiberremove: " << before_fr << " -> " << F.size() << " fibers" << std::endl;

    // --- Populate V.vall (needed for downstream consumers) once at the end. ---
    for (int fi = 0; fi < (int)F.size(); ++fi) {
        const auto& fv = F[fi].v;
        for (int vj : fv) {
            if (vj < 0 || vj >= (int)V.size()) continue;
            auto& vall = V[vj].vall;
            for (int vk : fv) {
                if (std::find(vall.begin(), vall.end(), vk) == vall.end()) {
                    vall.push_back(vk);
                }
            }
        }
    }

    // --- Build output arrays (convert back to 1-based). ---
    auto py_X = py::array_t<int32_t>({nX, 3});
    auto X_ptr = py_X.mutable_data();
    for (int i = 0; i < nX; ++i) {
        X_ptr[i*3 + 0] = (int32_t)(X[i][0] + 1);
        X_ptr[i*3 + 1] = (int32_t)(X[i][1] + 1);
        X_ptr[i*3 + 2] = (int32_t)(X[i][2] + 1);
    }

    py::list py_F;
    for (const auto& fiber : F) {
        if (fiber.v.empty()) continue;
        py::dict f_struct;
        std::vector<int> v_out;
        v_out.reserve(fiber.v.size());
        for (int vi : fiber.v) v_out.push_back(vi + 1);
        f_struct["v"] = py::cast(v_out);
        py_F.append(f_struct);
    }

    py::list py_E;
    for (const auto& fiber : F) {
        if (fiber.v.size() >= 2) {
            py::list edge;
            edge.append(fiber.v.front() + 1);
            edge.append(fiber.v.back() + 1);
            py_E.append(edge);
        }
    }

    py::list py_V;
    for (const auto& v : V) {
        py::dict v_struct;
        std::vector<int> fe_out, f_out, vall_out;
        fe_out.reserve(v.fe.size());
        f_out.reserve(v.f.size());
        vall_out.reserve(v.vall.size());
        for (int fi : v.fe) fe_out.push_back(fi + 1);
        for (int fi : v.f) f_out.push_back(fi + 1);
        for (int vi : v.vall) vall_out.push_back(vi + 1);
        v_struct["fe"] = py::cast(fe_out);
        v_struct["f"] = py::cast(f_out);
        v_struct["vall"] = py::cast(vall_out);
        py_V.append(v_struct);
    }

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
