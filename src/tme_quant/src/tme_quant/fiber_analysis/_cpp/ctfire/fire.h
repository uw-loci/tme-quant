#pragma once
// FIRE (Fiber Extraction) algorithm — C++ implementation
// Called from Python via fire_bindings.cpp (pybind11).
//
// Reference: Bredfeldt et al. (2014) J Biomed Opt 19(1):016007
// C++ source: https://github.com/uw-loci/curvelets/tree/master/src/CurveAlign_CT-FIRE/ctFIRE/CPP

#include <cstdint>
#include <vector>

namespace ctfire {

struct FiberTrace {
    std::vector<float> x;        // centerline x-coordinates (px)
    std::vector<float> y;        // centerline y-coordinates (px)
    std::vector<float> width;    // local fiber width at each centerline point (µm)
    float length;                // arc length (µm)
    float straightness;          // end-to-end / arc length, in [0, 1]
    float mean_width;            // mean width along centerline (µm)
};

struct FiberTrace3D {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> width;
    float length;
    float straightness;
    float mean_width;
};

struct FireParams2D {
    float pixel_size_um  = 1.0f;  // µm per pixel
    float min_length_um  = 5.0f;  // minimum fiber arc length to keep
    float max_width_um   = 20.0f; // maximum fiber half-width to keep
    float min_straight   = 0.0f;  // straightness threshold (0 = keep all)
};

struct FireParams3D {
    float pixel_size_um  = 1.0f;
    float z_spacing_um   = 1.0f;
    float min_length_um  = 5.0f;
    float max_width_um   = 20.0f;
    float min_straight   = 0.0f;
};

// TODO: implement
std::vector<FiberTrace> fire_2d(
    const uint8_t* mask,   // H × W binary mask, row-major
    int height, int width,
    const FireParams2D& params
);

// TODO: implement
std::vector<FiberTrace3D> fire_3d(
    const uint8_t* mask,   // D × H × W binary mask, row-major
    int depth, int height, int width,
    const FireParams3D& params
);

} // namespace ctfire
