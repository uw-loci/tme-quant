#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <array>
#include <omp.h>
#include <stdexcept>

namespace py = pybind11;

template<typename T, int d>
struct FindLocalMax {
    static constexpr T epsilon = 1e-3;

    // Match original: signed int seed, signed int return — preserves overflow behavior
    inline int fastrand(int& g_seed) {
        g_seed = (214013 * g_seed + 2531011);
        return (g_seed >> 16) & 0x7FFF;
    }

    // 2D constructor — matches original MEX logic exactly
    FindLocalMax(int sizey, int sizez, T* image,
                 std::vector<std::array<int, d>>& pts, int radius, T dmin)
    {
        static_assert(d == 2, "Dimension must be 2");

        printf("sizey=%d sizez=%d total=%llu\n", sizey, sizez, (uint64_t)sizey * sizez);
        fflush(stdout);

        // FIX 1: Use actual thread count, not hardcoded 1
        // (hardcoded 1 only sized the seeds/buffer vectors, but OMP still spawned many threads)
        const int nThreads = omp_get_max_threads();

        // Seed per thread
        std::vector<int> seeds(nThreads, 0);
        for (int i = 1; i < nThreads; ++i)
            seeds[i] = fastrand(seeds[i-1]);

        // Phase 1: add epsilon perturbation
        #pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < sizez; ++i) {
            const int tid = omp_get_thread_num();
            for (int j = 0; j < sizey; ++j) {
                const uint64_t offset = (uint64_t)sizey * i + j; 
                image[offset] += epsilon * T(fastrand(seeds[tid])) / T(0x7FFF);
            }
        }

        // Phase 2: find local maxima
        std::vector<std::vector<std::array<int, d>>> thread_buffer(nThreads);

        #pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < sizez; ++i) {
            const int tid = omp_get_thread_num();
            for (int j = 0; j < sizey; ++j) {
                const uint64_t offset = (uint64_t)sizey * i + j; // FIX: was sizey * j + i
                if (image[offset] < dmin) continue;

                bool local_max = true;
                for (int ii = -radius; ii <= radius && local_max; ++ii) {
                    const int z = ii + i;
                    if (z >= 0 && z < sizez) {
                        for (int jj = -radius; jj <= radius && local_max; ++jj) {
                            if (ii == 0 && jj == 0) continue;
                            const int y = jj + j;
                            if (y >= 0 && y < sizey) {
                                uint64_t neighbor_offset = (uint64_t)z * sizey + y;
                                if (image[offset] <= image[neighbor_offset])
                                    local_max = false;
                            }
                        }
                    }
                }
                // +1 for 1-based indexing (matches MATLAB MEX output)
                if (local_max)
                    thread_buffer[tid].push_back({i + 1, j + 1});
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
            out_ptr[i * 3 + 0] = pts[i][0]; // z index (1-based)
            out_ptr[i * 3 + 1] = pts[i][1]; // y index (1-based)
            out_ptr[i * 3 + 2] = 1;         // x = 1 (since sizex == 1)
        }
        return result;
    } else {
        throw std::runtime_error("3D FindLocalMax not yet implemented.");
    }
}