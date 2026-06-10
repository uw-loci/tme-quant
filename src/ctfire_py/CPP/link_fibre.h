#ifndef LINK_FIBRE_H
#define LINK_FIBRE_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// Platform-specific OpenMP includes (matching original MATLAB/C++ code)
#if __APPLE__
    #include <omp.h>
#elif _WIN64
    #include <omp.h>
#else
    #include <omp.h>
#endif

// Note: MATLAB is column major, but we're working with row-major in C++/Python
template<typename T, int d>
struct Fibre {
    std::array<T, d> direction;
    std::vector<std::array<int, d>> link;
    std::vector<int> link_index;
};

struct LinkTo {
    int f;
    bool isStart;
};

struct Terminal {
    int f;
    bool isStart;
};

template<typename T, int d>
struct LinkFibreAtNucleationPoint {
    T Dot(const std::array<T, d>& v1, const std::array<T, d>& v2) {
        T val = 0;
        for (int v = 0; v < d; ++v) val += v1[v] * v2[v];
        return val;
    }

    static T Length(const std::array<T, d>& v) {
        T val = 0;
        for (int i = 0; i < d; ++i) val += v[i] * v[i];
        return sqrt(val);
    }

    static void Normalize(std::array<T, d>& v) {
        T inv_len = T(1.0) / Length(v);
        for (int i = 0; i < d; ++i) v[i] *= inv_len;
    }

    using Fibre_Type = Fibre<T, d>;

    // Input arrays use 0-based indexing (Python/C++ convention)
    // Output array uses 1-based indexing (MATLAB convention for compatibility)
    explicit LinkFibreAtNucleationPoint(const int nPt,
                                        const std::vector<int>& nucleation_pts,
                                        const std::vector<Fibre_Type>& F_in,
                                        std::vector<std::vector<int>>& F_out,
                                        const T thresh_linka, const int sp) {
        
        // Initialize nucleation index mapping
        std::vector<int> nucleationIndex(nPt, -1);
        const int nNucleation = nucleation_pts.size();
        
        #pragma omp parallel for
        for (int i = 0; i < nNucleation; ++i) {
            if (nucleation_pts[i] >= 0 && nucleation_pts[i] < nPt) {
                nucleationIndex[nucleation_pts[i]] = i;
            }
        }

        // Build list of segments at each nucleation point
        std::vector<std::vector<Terminal>> pt_segments(nNucleation);
        const int nSegments = F_in.size();

        // Validate and categorize segments
        for (int f = 0; f < nSegments; ++f) {
            // Error checking: ensure consistency
            if (F_in[f].link_index.size() != F_in[f].link.size()) {
                std::cerr << "Warning: Inconsistent size in segment " << f << std::endl;
            }
            
            if (F_in[f].link_index.size() < 2) {
                std::cerr << "Warning: Segment " << f << " contains less than 2 nodes" << std::endl;
                continue;
            }
            
            const int start = F_in[f].link_index[0];
            const int end = F_in[f].link_index.back();

            // Bounds checking
            if (start >= nPt) {
                std::cerr << "Error: Nodal index overflow at segment " << f << " start" << std::endl;
                continue;
            }
            if (end >= nPt) {
                std::cerr << "Error: Nodal index overflow at segment " << f << " end" << std::endl;
                continue;
            }

            if (start >= 0 && nucleationIndex[start] != -1) {
                pt_segments[nucleationIndex[start]].push_back({f, true});
            }
            if (end >= 0 && nucleationIndex[end] != -1) {
                pt_segments[nucleationIndex[end]].push_back({f, false});
            }
        }

        // Initialize fiber-to-fiber connection maps
        std::vector<LinkTo> f_to_f_start(nSegments, {-1, false});
        std::vector<LinkTo> f_to_f_end(nSegments, {-1, false});

        // Find connections at each nucleation point
        #pragma omp parallel for
        for (int i = 0; i < nNucleation; ++i) {
            // For each nucleation point, check each pair of segments that join here
            const int nSegmentsOnPt = pt_segments[i].size();
            std::vector<bool> linked(nSegmentsOnPt, false);

            for (int f1 = 0; f1 < nSegmentsOnPt; ++f1) {
                if (linked[f1]) continue;
                
                auto& seg1 = F_in[pt_segments[i][f1].f];
                std::array<T, d> d1;

                // Calculate direction vector for segment 1
                if (pt_segments[i][f1].isStart) {
                    int start = 0;
                    int end = std::min(sp, (int)seg1.link.size()) - 1;
                    for (int v = 0; v < d; ++v) {
                        d1[v] = T(seg1.link[end][v] - seg1.link[start][v]);
                    }
                } else {
                    int start = std::max(1, (int)seg1.link.size() - sp) - 1;
                    int end = seg1.link.size() - 1;
                    for (int v = 0; v < d; ++v) {
                        d1[v] = -T(seg1.link[end][v] - seg1.link[start][v]);
                    }
                }
                Normalize(d1);

                // Try to find a matching segment
                for (int f2 = f1 + 1; f2 < nSegmentsOnPt; ++f2) {
                    if (linked[f2]) continue;
                    
                    auto& seg2 = F_in[pt_segments[i][f2].f];
                    std::array<T, d> d2;

                    // Calculate direction vector for segment 2
                    if (pt_segments[i][f2].isStart) {
                        int start = 0;
                        int end = std::min(sp, (int)seg2.link.size()) - 1;
                        for (int v = 0; v < d; ++v) {
                            d2[v] = T(seg2.link[end][v] - seg2.link[start][v]);
                        }
                    } else {
                        int start = std::max(1, (int)seg2.link.size() - sp) - 1;
                        int end = seg2.link.size() - 1;
                        for (int v = 0; v < d; ++v) {
                            d2[v] = -T(seg2.link[end][v] - seg2.link[start][v]);
                        }
                    }
                    Normalize(d2);

                    T cos_angle = Dot(d1, d2);
                    
                    // If angle is within threshold, link the segments
                    if (cos_angle < thresh_linka) {
                        linked[f1] = true;
                        linked[f2] = true;
                        
                        auto& t1 = pt_segments[i][f1];
                        auto& t2 = pt_segments[i][f2];

                        // Record bidirectional connections
                        if (t1.isStart) {
                            f_to_f_start[t1.f] = {t2.f, t2.isStart};
                        } else {
                            f_to_f_end[t1.f] = {t2.f, t2.isStart};
                        }

                        if (t2.isStart) {
                            f_to_f_start[t2.f] = {t1.f, t1.isStart};
                        } else {
                            f_to_f_end[t2.f] = {t1.f, t1.isStart};
                        }
                        
                        // TODO: Use some local metric to pick which one to link
                        // Currently using first match, could be improved with better heuristics
                        break;
                    }
                }
            }
        }

        // Find all fibers with at least one free end
        std::vector<int> free_ends;
        for (int f = 0; f < nSegments; ++f) {
            if (f_to_f_start[f].f == -1 || f_to_f_end[f].f == -1) {
                free_ends.push_back(f);
            }
        }

        // Construct long fibers by following connections
        std::vector<bool> used(nSegments, false);
        F_out.clear();
        
        for (int f_idx : free_ends) {
            if (used[f_idx]) continue;
            
            std::vector<int> current_f;
            bool linked_start = (f_to_f_start[f_idx].f == -1);
            
            int curr = f_idx;
            while (true) {
                if (used[curr]) {
                    std::cerr << "Error: Fiber " << curr << " already used" << std::endl;
                    break;
                }
                used[curr] = true;
                
                if (linked_start) {
                    // Add indices in forward order, converting to 1-based indexing
                    for (int idx : F_in[curr].link_index) {
                        current_f.push_back(idx + 1);  // Convert 0-based to 1-based
                    }
                    if (f_to_f_end[curr].f == -1) break;
                    linked_start = f_to_f_end[curr].isStart;
                    curr = f_to_f_end[curr].f;
                } else {
                    // Add indices in reverse order, converting to 1-based indexing
                    for (int j = F_in[curr].link_index.size() - 1; j >= 0; --j) {
                        current_f.push_back(F_in[curr].link_index[j] + 1);  // Convert 0-based to 1-based
                    }
                    if (f_to_f_start[curr].f == -1) break;
                    linked_start = f_to_f_start[curr].isStart;
                    curr = f_to_f_start[curr].f;
                }
            }
            
            F_out.push_back(current_f);
        }
    }
};

#endif
