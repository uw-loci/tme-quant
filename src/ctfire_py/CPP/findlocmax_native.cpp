#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <array>
#include <random>
#include <omp.h>
#include <stdexcept>

namespace py = pybind11;

template<typename T, int d>
struct FindLocalMax {
    static constexpr T epsilon = 1e-3;

    // 2D constructor — matches original MEX logic exactly
    FindLocalMax(int sizey, int sizez, T* image,
                 std::vector<std::array<int, d>>& pts, int radius, T dmin)
    {
        static_assert(d == 2, "Dimension must be 2");

        printf("sizey=%d sizez=%d total=%llu\n", sizey, sizez, (uint64_t)sizey * sizez);
        fflush(stdout);

        const int nThreads = omp_get_max_threads();

        // Phase 1: add epsilon perturbation using MT19937 seeded at 100.
        //
        // MATLAB's findlocmax.m does:
        //   s = RandStream('twister','Seed',100);  % mt19937ar, seed 100
        //   RandStream.setGlobalStream(s);
        //   d = d + 1e-3 * rand(size(d));          % fills in COLUMN-MAJOR order
        //
        // Two subtleties must be reproduced exactly:
        //
        // 1. MATLAB rand() uses genrand_res53 (Matsumoto & Nishimura reference):
        //      a = mt_out1 >> 5;  b = mt_out2 >> 6;
        //      value = (a * 67108864.0 + b) / 9007199254740992.0;
        //    This consumes TWO uint32 outputs per draw, producing a double.
        //    Using uniform_real_distribution<float> (one uint32 per draw) gives
        //    a completely different sequence from the second draw onward.
        //
        // 2. MATLAB fills in COLUMN-MAJOR order for a [K=height, J=width] matrix:
        //      outer loop = col  (slower, varies last)
        //      inner loop = row  (faster, varies first)
        //    So pixel (row, col) receives draw number (col * height + row + 1).
        //    The draw-assignment loop order below is kept identical to MATLAB's to
        //    preserve which draw lands on which logical pixel.
        //
        //    The flat array passed from Python is ROW-MAJOR (numpy C-order,
        //    dsm.flatten() default), so pixel (row, col) is actually at flat position:
        //      flat_idx = row * sizez + col     (sizez = I = width, the row stride)
        //    Phase 1 and Phase 2 must address the same memory position for pixel
        //    (row, col), so both use this row-major formula.
        {
            std::mt19937 rng(100);
            // Outer loop over columns (MATLAB's slower/outer dimension for a
            // [height, width] matrix stored column-major) — loop order preserved
            // for RNG-draw parity even though the destination is row-major.
            for (int col = 0; col < sizez; ++col) {   // sizez = I = width
                // Inner loop over rows (MATLAB's faster/inner dimension).
                for (int row = 0; row < sizey; ++row) {   // sizey = J = height
                    // Row-major flat index for Python's C-contiguous dsm array.
                    const uint64_t flat_idx = (uint64_t)row * sizez + col;
                    // genrand_res53: two uint32 outputs → double in [0,1),
                    // matching MATLAB's rand() output exactly for seed 100.
                    const uint32_t a = rng() >> 5;   // top 27 bits
                    const uint32_t b = rng() >> 6;   // top 26 bits
                    const double draw = (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
                    image[flat_idx] += static_cast<T>(epsilon * draw);
                }
            }
        }

        // Phase 2: find local maxima
        std::vector<std::vector<std::array<int, d>>> thread_buffer(nThreads);

        #pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < sizez; ++i) {
            const int tid = omp_get_thread_num();
            for (int j = 0; j < sizey; ++j) {
                const uint64_t offset = (uint64_t)sizez * j + i; // row-major: row*width + col, matches C-order dsm_flat
                if (image[offset] < dmin) continue;

                bool local_max = true;
                for (int ii = -radius; ii <= radius && local_max; ++ii) {
                    const int z = ii + i;
                    if (z >= 0 && z < sizez) {
                        for (int jj = -radius; jj <= radius && local_max; ++jj) {
                            if (ii == 0 && jj == 0) continue;
                            const int y = jj + j;
                            if (y >= 0 && y < sizey) {
                                uint64_t neighbor_offset = (uint64_t)sizez * y + z;
                                if (image[offset] <= image[neighbor_offset])
                                    local_max = false;
                            }
                        }
                    }
                }
                // Output as (row, col): i ranges over columns, j ranges over rows.
                if (local_max)
                    thread_buffer[tid].push_back({j, i});
            }
        }

        pts.clear();
        for (int t = 0; t < nThreads; ++t)
            for (auto& p : thread_buffer[t])
                pts.push_back(p);
    }

    // 3D constructor — stub, matches original
    FindLocalMax(int sizex, int sizey, int sizez, T* image,
                 std::vector<std::array<int, d>>& pts, int radius, T dmin)
    {
        static_assert(d == 3, "Dimension must be 3");
    }
};

py::array_t<int32_t> findlocmax_native(
    int sizex, int sizey, int sizez,
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    int radius, float dmin)
{
    if ((uint64_t)image.size() != (uint64_t)sizex * sizey * sizez)
        throw std::runtime_error("Input image size does not match dimensions.");

    float* img_ptr = image.mutable_data();
    std::vector<std::array<int, 2>> pts;

    if (sizex == 1) {
        {
            py::gil_scoped_release release;
            FindLocalMax<float, 2>(sizey, sizez, img_ptr, pts, radius, dmin);
        }

        auto result = py::array_t<int32_t>({(int)pts.size(), 3});
        auto out_ptr = result.mutable_data();
        int N = (int)pts.size();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            out_ptr[i * 3 + 0] = pts[i][0]; // row index (0-based)
            out_ptr[i * 3 + 1] = pts[i][1]; // col index (0-based)
            out_ptr[i * 3 + 2] = 0;         // x = 0 (since sizex == 1)
        }
        return result;
    } else {
        throw std::runtime_error("3D FindLocalMax not yet implemented.");
    }
}