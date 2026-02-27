#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <array>
#include <omp.h>

namespace py = pybind11;

// --- KEEPING YOUR LOGIC HEART UNTOUCHED ---
template<typename T, int d>
struct FindLocalMax {
    static constexpr T epsilon = 1e-3;
    inline int fastrand(uint32_t& g_seed) { 
        g_seed = (214013 * g_seed + 2531011); 
        return (g_seed >> 16) & 0x7FFF; 
    }
    
    // 3D Constructor (Placeholder logic as per your snippet)
    FindLocalMax(int sizex, int sizey, int sizez, T* image, std::vector<std::array<int, d>>& pts, int radius, T dmin) {
        static_assert(d == 3, "Dimension must be 3");
        // Logic for 3D would go here
    }

    // 2D Constructor
    FindLocalMax(int sizey, int sizez, T* image, std::vector<std::array<int, d>>& pts, int radius, T dmin) {
        static_assert(d == 2, "Dimension must be 2");
        const int nThreads = omp_get_max_threads();
        std::vector<uint32_t> seeds(nThreads, 0); 
        for(int i = 1; i < nThreads; ++i) seeds[i] = fastrand(seeds[0]);

        #pragma omp parallel for
        for(int i = 0; i < sizez; ++i){
            const int tid = omp_get_thread_num();
            for(int j = 0; j < sizey; ++j){
                const uint64_t offset = (uint64_t)sizey * i + j;
                image[offset] += epsilon * T(fastrand(seeds[tid])) / T(0x7FFF);
            }
        }

        std::vector<std::vector<std::array<int, d>>> thread_buffer(nThreads);
        #pragma omp parallel for
        for(int i = 0; i < sizez; ++i){
            const int tid = omp_get_thread_num();
            for(int j = 0; j < sizey; ++j){
                const uint64_t offset = (uint64_t)sizey * i + j;
                if(image[offset] < dmin) continue;
                bool local_max = true;
                for(int ii = -radius; ii <= radius && local_max; ++ii){
                    const int z = ii + i;
                    if(z >= 0 && z < sizez){
                        for(int jj = -radius; jj <= radius && local_max; ++jj){
                            if(ii == 0 && jj == 0) continue;
                            const int y = jj + j;                        
                            if(y >= 0 && y < sizey){
                                uint64_t neighbor_offset = (uint64_t)z * sizey + y;
                                if(image[offset] <= image[neighbor_offset])
                                    local_max = false;
                            }
                        }
                    }
                }
                if(local_max) { 
                    // Note: We keep +1 for Matlab parity if needed, 
                    // but usually Python prefers 0-based indexing.
                    thread_buffer[tid].push_back({i + 1, j + 1}); 
                }
            }
        }

        pts.clear();
        for(int t = 0; t < nThreads; ++t)
            pts.insert(pts.end(), thread_buffer[t].begin(), thread_buffer[t].end());
    }
};

// --- NEW PYTHON WRAPPER (REPLACES mexFunction) ---

py::array_t<int32_t> findlocmax_native(
    int sizex, int sizey, int sizez, 
    py::array_t<float, py::array::c_style | py::array::forcecast> image, 
    int radius, float dmin) 
{
    // Validate image size
    if (image.size() != (uint64_t)sizex * sizey * sizez) {
        throw std::runtime_error("Input image size does not match dimensions.");
    }

    // Get raw pointer to the image data
    float* img_ptr = image.mutable_data();

    std::vector<std::array<int, 2>> pts;
    std::vector<std::array<int, 3>> pts_3D;

    // Direct translation of your switch logic
    if (sizex == 1) {
        {
            py::gil_scoped_release release;
            FindLocalMax<float, 2>(sizey, sizez, img_ptr, pts, radius, dmin);
        }
        
        // Prepare output array: [N x 3] as in your MEX code
        auto result = py::array_t<int32_t>({ (int)pts.size(), 3 });
        auto out_ptr = result.mutable_data();
        int N = pts.size();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            out_ptr[i * 3 + 0] = pts[i][0]; // Row coordinate
            out_ptr[i * 3 + 1] = pts[i][1]; // Col coordinate
            out_ptr[i * 3 + 2] = 1;         // Dummy 3rd dim for parity
        }
        return result;
    } else {
        // Placeholder for 3D logic
        throw std::runtime_error("3D FindLocalMax not yet implemented in wrapper.");
    }
}