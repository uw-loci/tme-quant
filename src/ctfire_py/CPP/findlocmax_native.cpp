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
    
    inline uint32_t fastrand(uint32_t& g_seed) { 
        g_seed = (214013 * g_seed + 2531011); 
        return (g_seed >> 16) & 0x7FFF; 
    } 

    FindLocalMax(int sizey, int sizez, T* image, std::vector<std::array<int, d>>& pts, int radius, T dmin) {
        static_assert(d == 2, "Dimension must be 2");

        // CRITICAL FIX: Declare vector INSIDE the parallel block
        #pragma omp parallel 
        {
            std::vector<std::array<int, d>> local_max_points;
            uint32_t seed = 2531011 + omp_get_thread_num();

            #pragma omp for
            for(int i = 0; i < sizez; ++i){
                for(int j = 0; j < sizey; ++j){
                    const uint64_t offset = (uint64_t)sizey * i + j;
                    image[offset] += epsilon * T(fastrand(seed)) / T(0x7FFF);
                }
            }

            #pragma omp for
            for(int i = 0; i < sizez; ++i){
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
                                    if(image[offset] <= image[neighbor_offset]){
                                        local_max = false;
                                    }
                                }
                            }
                        }
                    }
                    if(local_max) { 
                        local_max_points.push_back({i + 1, j + 1}); 
                    }
                }
            }

            #pragma omp critical
            {
                pts.insert(pts.end(), local_max_points.begin(), local_max_points.end());
            }
        }
    }
    
    FindLocalMax(int sizex, int sizey, int sizez, T* image, std::vector<std::array<int, d>>& pts, int radius, T dmin) {
        static_assert(d == 3, "Dimension must be 3");
    }
};

py::array_t<int32_t> findlocmax_native(
    int sizex, int sizey, int sizez, 
    py::array_t<float, py::array::c_style | py::array::forcecast> image, 
    int radius, float dmin) 
{
    if (image.size() != (uint64_t)sizex * sizey * sizez) {
        throw std::runtime_error("Input image size does not match dimensions.");
    }

    float* img_ptr = image.mutable_data();
    std::vector<std::array<int, 2>> pts;

    if (sizex == 1) {
        {
            py::gil_scoped_release release;
            FindLocalMax<float, 2>(sizey, sizez, img_ptr, pts, radius, dmin);
        }
        
        auto result = py::array_t<int32_t>({ (int)pts.size(), 3 });
        auto out_ptr = result.mutable_data();
        int N = pts.size();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            out_ptr[i * 3 + 0] = pts[i][0]; 
            out_ptr[i * 3 + 1] = pts[i][1]; 
            out_ptr[i * 3 + 2] = 1;         
        }
        return result;
    } else {
        throw std::runtime_error("3D FindLocalMax not yet implemented.");
    }
}