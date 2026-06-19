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
#include <unordered_set>
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

    // Straight-line continuity check matching MATLAB's ind_btw_nodes +
    // findLMP.m line 55-60:
    //   for k=size(LMP,1):-1:1
    //       ind = ind_btw_nodes(u, LMP(k,:), size(d));
    //       if any(d(ind) < LMPthresh); LMP(k,:) = []; end
    //   end
    // ind_btw_nodes samples one pixel per unit length between p and q
    // (len = max(1, ||p-q||); x = 0:1/len:1; P = round((1-x)*p + x*q)).
    // Returns true iff every sampled pixel has image[pixel] >= thresh.
    // Previously the C++ extend_xlink LMP search skipped this check,
    // letting LMPs across low-distance valleys become fibers and inflate
    // the network on both initial and continuation passes.
    static bool line_clear_above_thresh(const std::array<int, 2>& p,
                                        const std::array<int, 2>& q,
                                        const T* image, int sizex, int sizey,
                                        T thresh) {
        const T dr = T(p[0] - q[0]);
        const T dc = T(p[1] - q[1]);
        T len = std::sqrt(dr * dr + dc * dc);
        if (len < T(1)) len = T(1);
        const int nsteps = (int)std::floor(len); // x = 0, 1/len, ..., floor(len)/len
        for (int k = 0; k <= nsteps; ++k) {
            const T t = T(k) / len;
            const int r = (int)std::floor((T(1) - t) * p[0] + t * q[0] + T(0.5));
            const int c = (int)std::floor((T(1) - t) * p[1] + t * q[1] + T(0.5));
            if (r < 0 || r >= sizey || c < 0 || c >= sizex) continue;
            if (image[r * sizex + c] < thresh) return false;
        }
        return true;
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
            // Match MATLAB: r = max(2, ceil(d(...))) for initial radius
            // This ensures a minimum search radius even for low distance values
            const int r_i = std::max(2, (int)ceil(image[pts[i][0] * sizex + pts[i][1]]));
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

                    // Check if this is a local maximum among boundary neighbors only.
                    //
                    // MATLAB's getdB pads d with zpad=3 zeros, making the volume
                    // [7, J+6, I+6].  For 2D images the nucleation sits at z=4
                    // (the image plane); the search box extends ±r in z, giving
                    // z1≤1 and z2≥7 for r≥3.  The z-face "side" pixels land in
                    // zero-padded planes (fail thresh_LMP), so only the x-face and
                    // y-face pixels at z=4 contribute LMPs.
                    //
                    // For an x-face pixel (col=x1 or x2), getdB's dBxn contains
                    // offsets only in the y (±ys=±7) and z (±zs=±1) directions —
                    // NOT in x.  So the pixel is compared against its SAME-FACE
                    // y-neighbors, not against interior (col±1) pixels.  The z±1
                    // neighbors are in zero-padded planes and never beat it.
                    //
                    // This is equivalent to: compare each boundary candidate only
                    // against its neighbours that are ALSO on the same boundary face.
                    // For a square box that means the same-col neighbors (for left/
                    // right faces) or same-row neighbors (for top/bottom faces), which
                    // is exactly what the boundary-only restriction below achieves for
                    // all practical radii (r≥2, so faces are ≥4 px apart and other-face
                    // pixels never appear in the 8-neighbourhood).
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
                        // MATLAB findLMP.m:55-60 continuity check: discard
                        // any LMP separated from the center by a sub-
                        // threshold valley.
                        if (!line_clear_above_thresh(nucleation, p, image,
                                                     sizex, sizey, thresh_LMP)) {
                            continue;
                        }

                        // Check if too close to existing fiber endpoints
                        // Match MATLAB: use Euclidean distance norm(p1-p2) < LMPdist
                        bool too_close = false;
                        for (size_t b = 0; b < fibres[i].size(); ++b) {
                            T dx = p[0] - fibres[i][b].link[1][0];
                            T dy = p[1] - fibres[i][b].link[1][1];
                            T dist = sqrt(dx*dx + dy*dy);
                            if (dist < thresh_LMPdist) {
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

                                    // Check if it's a local maximum among boundary neighbors only.
                                    // Same face-local logic as the initial LMP search above.
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
                                        // MATLAB findLMP.m:55-60 continuity
                                        // check also runs in continuation
                                        // (findLMP is called with LMPdist=0
                                        // so the line-trace gate is always
                                        // active). Skip candidates whose
                                        // straight line from p_current
                                        // crosses a sub-threshold region.
                                        if (!line_clear_above_thresh(p_current, p_cand, image,
                                                                     sizex, sizey, thresh_LMP)) {
                                            continue;
                                        }

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

        // Step 2: Remove duplicate fibers that share the same unordered
        // endpoint pixel-pair.
        //
        // MATLAB equivalent (extend_xlink.m:143-156):
        //     A = spalloc(n,n,10*n);
        //     for fi=1:length(F)
        //         v1 = F(fi).v(1);  v2 = F(fi).v(end);
        //         if A(v1,v2)==1, fremove(fi) = 1; end
        //         A(v1,v2) = 1;  A(v2,v1) = 1;   % symmetric
        //     end
        //
        // The previous C++ used two parallel passes (forward /
        // offset_begin<offset_end, reverse / offset_begin>offset_end) that
        // had three divergences from MATLAB:
        //   1. The reverse pass never wrote to link_map, so two fibers
        //      both running in the reverse direction with the same
        //      (begin, end) pair were never deduped.
        //   2. Self-loops (offset_begin == offset_end) fell through both
        //      branches and were never deduped.
        //   3. The nucleation_map[offset_end] gate over-restricted dedup:
        //      this C++ collapses non-nucleation vertices at shared pixel
        //      offsets via index_map, so MATLAB's unconditional A(v1,v2)
        //      dedup should map to an unconditional pixel-offset dedup.
        // Fix: single serial pass with a symmetric hashed-pair key that
        // also handles self-loops.
        std::unordered_set<uint64_t> seen_pairs;
        seen_pairs.reserve(fibres.size() * 2);
        const uint64_t pack_shift = 32;  // pixel offsets fit in 32 bits for
                                          // any realistic image (sizex<=2^16)
        for (int f = 0; f < (int)fibres.size(); ++f) {
            for (int branch = 0; branch < (int)fibres[f].size(); ++branch) {
                auto& br = fibres[f][branch];
                if (br.link.size() < 2) continue;

                const uint64_t ob = static_cast<uint64_t>(br.link[0][0]) * sizex +
                                    static_cast<uint64_t>(br.link[0][1]);
                const uint64_t oe = static_cast<uint64_t>(br.link.back()[0]) * sizex +
                                    static_cast<uint64_t>(br.link.back()[1]);
                const uint64_t lo = ob < oe ? ob : oe;  // symmetric key
                const uint64_t hi = ob < oe ? oe : ob;
                const uint64_t key = (lo << pack_shift) | hi;

                if (seen_pairs.count(key)) {
                    br.link.clear();  // duplicate pair -> drop this fiber
                } else {
                    seen_pairs.insert(key);
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
                        // Store 0-based pixel coords
                        X.push_back(std::array<int, d>{fibres[f][branch].link[node][0],
                                                       fibres[f][branch].link[node][1]});
                    }
                }
            }
        }

        // Step 4: Copy radius values (DSM values at vertex locations)
        R.resize(X.size());
        #pragma omp parallel for
        for (int i = 0; i < (int)X.size(); ++i) {
            // Store actual DSM value, not ceil (X coords are 0-based)
            R[i] = image[X[i][0] * sizex + X[i][1]];
        }

        // Step 5: Prepare nucleation points for linking (0-based indexing)
        const int nNucleation = (int)pts.size();
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

        // Step 7: MATLAB-faithful: no orientation-based linking here.
        // MATLAB's extend_xlink.m only emits the individual F_init branches
        // (after the (v_start, v_end) pair dedup applied above) -- fiber
        // linking at shared nucleation points is handled later by fiberlink
        // inside fiberproc. See /src/FIRE/xlink/extend_xlink.m lines 142-167.
        // Previous code ran LinkFibreAtNucleationPoint here, which double-
        // merged fibers and produced far fewer, longer fibers than MATLAB,
        // leaving fiberremove with too little short-fiber work to do.
        F.clear();
        F.reserve(F_init.size());
        for (int f = 0; f < (int)F_init.size(); ++f) {
            if (F_init[f].link_index.size() < 2) continue; // trimxfv analogue
            std::vector<int> v;
            v.reserve(F_init[f].link_index.size());
            for (int idx : F_init[f].link_index) {
                v.push_back(idx); // 0-based vertex indices
            }
            F.push_back(std::move(v));
        }
        (void)thresh_linka; (void)sp; (void)nucleation_pts;

        std::cout << "Fiber segments: " << F_init.size() << std::endl;
        std::cout << "Fibers (no link, MATLAB-faithful): " << F.size() << std::endl;

        // Step 8: Build auxiliary data structures (Xfe, Xf, Xvall, Ff)
        Xfe.resize(X.size());
        Xf.resize(X.size());
        Xvall.resize(X.size());
        
        for (int f = 0; f < (int)F.size(); ++f) {
            if (F[f].size()) {
                const int index_begin = F[f][0];
                const int index_end = F[f].back();
                
                if (index_begin >= (int)X.size() || index_end >= (int)X.size()) {
                    std::cerr << "Error: Index out of bounds in Xfe" << std::endl;
                    continue;
                }
                
                Xfe[index_begin].push_back(f);
                Xfe[index_end].push_back(f);
            }
            
            for (int node = 0; node < (int)F[f].size(); ++node) {
                Xf[F[f][node]].push_back(f);
                for (int node2 = 0; node2 < (int)F[f].size(); ++node2) {
                    Xvall[F[f][node]].push_back(F[f][node2]);
                }
            }
        }

        // Step 9: Build Ff (fibers connected to each fiber)
        Ff.resize(F.size());
        #pragma omp parallel for
        for (int f = 0; f < (int)F.size(); ++f) {
            for (int node = 0; node < (int)F[f].size(); ++node) {
                for (int f2 = 0; f2 < (int)Xf[F[f][node]].size(); ++f2) {
                    if (Xf[F[f][node]][f2] != f) {
                        bool unique = true;
                        for (int ff = 0; ff < (int)Ff[f].size(); ++ff) {
                            if (Ff[f][ff] == Xf[F[f][node]][f2]) {
                                unique = false;
                                break;
                            }
                        }
                        if (unique) Ff[f].push_back(Xf[F[f][node]][f2]);
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

    // Validate input image size (matching findlocmax safety check)
    if ((uint64_t)image.size() != (uint64_t)sizex * sizey * sizez) {
        throw std::runtime_error("Input image size does not match dimensions. Expected " + 
                                std::to_string(sizex * sizey * sizez) + " elements, got " + 
                                std::to_string(image.size()));
    }

    float* img_ptr = image.mutable_data();
    int nNuc = pts_in.shape(0);
    
    std::vector<std::array<int, 2>> pts(nNuc);
    // Input pts are already 0-based
    for (int i = 0; i < nNuc; ++i) {
        pts[i] = {pts_in.at(i, 0), pts_in.at(i, 1)};
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
        // Note: For 2D (sizex==1), the engine's 2D ctor takes (sizex=width, sizey=height);
        // sizez holds width and sizey holds height here, so pass (sizez, sizey).
        ExtendXLink<float, 2> engine(sizez, sizey, img_ptr, pts,
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
