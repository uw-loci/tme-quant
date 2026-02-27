#ifndef LINK_FIBRE_H
#define LINK_FIBRE_H

#include <vector>
#include <array>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iostream>

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

    explicit LinkFibreAtNucleationPoint(const int nPt,
                                        const std::vector<int>& nucleation_pts,
                                        const std::vector<Fibre_Type>& F_in,
                                        std::vector<std::vector<int>>& F_out,
                                        const T thresh_linka, const int sp) {
        
        std::vector<int> nucleationIndex(nPt, -1);
        const int nNucleation = nucleation_pts.size();
        
        #pragma omp parallel for
        for (int i = 0; i < nNucleation; ++i) {
            if (nucleation_pts[i] >= 0 && nucleation_pts[i] < nPt) 
                nucleationIndex[nucleation_pts[i]] = i;
        }

        std::vector<std::vector<Terminal>> pt_segments(nNucleation);
        const int nSegments = F_in.size();

        for (int f = 0; f < nSegments; ++f) {
            if (F_in[f].link_index.size() < 2) continue;
            const int start = F_in[f].link_index[0];
            const int end = F_in[f].link_index.back();

            if (start < nPt && nucleationIndex[start] != -1) 
                pt_segments[nucleationIndex[start]].push_back({f, true});
            if (end < nPt && nucleationIndex[end] != -1) 
                pt_segments[nucleationIndex[end]].push_back({f, false});
        }

        std::vector<LinkTo> f_to_f_start(nSegments, {-1, false});
        std::vector<LinkTo> f_to_f_end(nSegments, {-1, false});

        #pragma omp parallel for
        for (int i = 0; i < nNucleation; ++i) {
            const int nSegmentsOnPt = pt_segments[i].size();
            std::vector<bool> linked(nSegmentsOnPt, false);

            for (int f1 = 0; f1 < nSegmentsOnPt; ++f1) {
                if (linked[f1]) continue;
                auto& seg1 = F_in[pt_segments[i][f1].f];
                std::array<T, d> d1;

                if (pt_segments[i][f1].isStart) {
                    int end = std::min(sp, (int)seg1.link.size()) - 1;
                    for (int v = 0; v < d; ++v) d1[v] = T(seg1.link[end][v] - seg1.link[0][v]);
                } else {
                    int start = std::max(1, (int)seg1.link.size() - sp) - 1;
                    int end = seg1.link.size() - 1;
                    for (int v = 0; v < d; ++v) d1[v] = -T(seg1.link[end][v] - seg1.link[start][v]);
                }
                Normalize(d1);

                for (int f2 = f1 + 1; f2 < nSegmentsOnPt; ++f2) {
                    if (linked[f2]) continue;
                    auto& seg2 = F_in[pt_segments[i][f2].f];
                    std::array<T, d> d2;

                    if (pt_segments[i][f2].isStart) {
                        int end = std::min(sp, (int)seg2.link.size()) - 1;
                        for (int v = 0; v < d; ++v) d2[v] = T(seg2.link[end][v] - seg2.link[0][v]);
                    } else {
                        int start = std::max(1, (int)seg2.link.size() - sp) - 1;
                        int end = seg2.link.size() - 1;
                        for (int v = 0; v < d; ++v) d2[v] = -T(seg2.link[end][v] - seg2.link[start][v]);
                    }
                    Normalize(d2);

                    if (Dot(d1, d2) < thresh_linka) {
                        linked[f1] = linked[f2] = true;
                        auto& t1 = pt_segments[i][f1];
                        auto& t2 = pt_segments[i][f2];

                        if (t1.isStart) f_to_f_start[t1.f] = {t2.f, t2.isStart};
                        else f_to_f_end[t1.f] = {t2.f, t2.isStart};

                        if (t2.isStart) f_to_f_start[t2.f] = {t1.f, t1.isStart};
                        else f_to_f_end[t2.f] = {t1.f, t1.isStart};
                        break;
                    }
                }
            }
        }

        std::vector<int> free_ends;
        for (int f = 0; f < nSegments; ++f) {
            if (f_to_f_start[f].f == -1 || f_to_f_end[f].f == -1) free_ends.push_back(f);
        }

        std::vector<bool> used(nSegments, false);
        for (int f_idx : free_ends) {
            if (used[f_idx]) continue;
            std::vector<int> current_f;
            bool linked_start = (f_to_f_start[f_idx].f == -1);

            int curr = f_idx;
            while (true) {
                used[curr] = true;
                if (linked_start) {
                    for (int idx : F_in[curr].link_index) current_f.push_back(idx);
                    if (f_to_f_end[curr].f == -1) break;
                    linked_start = f_to_f_end[curr].isStart;
                    curr = f_to_f_end[curr].f;
                } else {
                    for (int j = F_in[curr].link_index.size() - 1; j >= 0; --j)
                        current_f.push_back(F_in[curr].link_index[j]);
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