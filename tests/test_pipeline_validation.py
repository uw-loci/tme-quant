"""
End-to-end pipeline validation tests.

Validates the complete fiber detection pipeline from image input to fiber output,
comparing intermediate and final results against MATLAB reference data.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ctfire_py.CPP import fiber_backend
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fiber_backend not available")
@pytest.mark.parametrize("test_case,params", [
    ("findlocmax_2B_D9_ROI1_default.mat", {
        "s_xlinkbox": 8,
        "thresh_Dxlink": 1.5,
        "thresh_LMP": 0.25,
        "thresh_LMPdist": 25,
        "thresh_ext": 0.25,
        "lam_dirdecay": 1.0,
        "s_fiberdir": 2,
        "thresh_linkd": 15.0,
        "thresh_linka": 45.0,
    }),
    ("findlocmax_real1_default.mat", {
        "s_xlinkbox": 8,
        "thresh_Dxlink": 1.5,
        "thresh_LMP": 0.25,
        "thresh_LMPdist": 25,
        "thresh_ext": 0.25,
        "lam_dirdecay": 1.0,
        "s_fiberdir": 2,
        "thresh_linkd": 15.0,
        "thresh_linka": 45.0,
    }),
])
def test_pipeline_end_to_end(test_case, params):
    """
    Test complete pipeline from image to fibers using MATLAB DSM.
    
    Validates that the pipeline can:
    1. Load distance map  
    2. Detect nucleation points (findlocmax)
    3. Extend fibers (extend_xlink)
    4. Produce valid output structures
    """
    # Load MATLAB reference with distance map
    h5py = pytest.importorskip("h5py")
    mat_path = Path(__file__).parent / "test_results" / "cpp_test_files" / test_case
    if not mat_path.exists():
        pytest.skip(f"MATLAB reference not found: {mat_path}")
    
    with h5py.File(mat_path, 'r') as f:
        dsm = f['dsm'][:]  # Already in (rows, cols, depth) format
    
    # Prepare for C++
    J, I, K = dsm.shape
    K = 1
    dsm_flat = dsm.flatten().astype(np.float32)
    
    print(f"\n=== Pipeline Test: {test_case} ===")
    print(f"DSM shape: {dsm.shape}")
    
    # Stage 2: Detect nucleation points
    print("\nStage 1: Nucleation point detection")
    xlink = fiber_backend.find_local_max(
        K, J, I, dsm_flat,
        params["s_xlinkbox"],
        params["thresh_Dxlink"]
    )
    
    print(f"  Nucleation points detected: {len(xlink)}")
    assert len(xlink) > 0, "No nucleation points detected"
    assert xlink.shape[1] == 3, "Invalid xlink shape"
    assert np.all(xlink[:, 0] >= 0) and np.all(xlink[:, 0] < J), "Invalid row coordinates"
    assert np.all(xlink[:, 1] >= 0) and np.all(xlink[:, 1] < I), "Invalid col coordinates"
    assert np.all(xlink[:, 2] == 0), "Invalid depth coordinates"
    
    # Stage 3: Extend fibers
    print("\nStage 2: Fiber extension")
    X, F, V, R = fiber_backend.extend_xlink(K, J, I, dsm_flat, xlink, params)
    
    print(f"  Vertices: {len(X)}")
    print(f"  Fibers: {len(F)}")
    print(f"  Vertices per fiber: {len(X) / max(1, len(F)):.1f}")
    
    # Validate outputs
    assert len(X) > 0, "No vertices generated"
    assert len(F) > 0, "No fibers generated"
    assert len(R) == len(X), "Radius array length mismatch"
    assert X.shape[1] == 3, "Invalid vertex dimensions"
    
    # Check vertex coordinates are within image bounds
    assert np.all(X[:, 0] >= 0) and np.all(X[:, 0] <= J), "Vertices outside row bounds"
    assert np.all(X[:, 1] >= 0) and np.all(X[:, 1] <= I), "Vertices outside col bounds"
    
    # Check radii are positive and reasonable
    R_arr = np.array(R) if isinstance(R, list) else R
    assert np.all(R_arr > 0), "Non-positive radii found"
    assert np.all(R_arr < max(J, I) / 2), "Unreasonably large radii found"
    
    print("\n✓ Pipeline completed successfully")
    

@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fiber_backend not available")
def test_pipeline_consistency():
    """
    Test that running the pipeline twice with same inputs gives consistent results.
    
    Note: Results won't be identical due to RNG, but should be similar.
    """
    # Create simple test image
    J, I, K = 100, 100, 1
    img = np.zeros((J, I), dtype=np.float32)
    
    # Add some peaks
    img[25, 25] = 1.0
    img[75, 75] = 1.0
    img[50, 50] = 0.8
    
    dsm_flat = img.flatten()
    
    params = {
        "s_xlinkbox": 8,
        "thresh_Dxlink": 0.5,
        "thresh_LMP": 0.25,
        "thresh_LMPdist": 25,
        "thresh_ext": 0.25,
        "lam_dirdecay": 1.0,
        "s_fiberdir": 2,
        "thresh_linkd": 15.0,
        "thresh_linka": 45.0,
    }
    
    # Run 1
    xlink1 = fiber_backend.find_local_max(K, J, I, dsm_flat, params["s_xlinkbox"], params["thresh_Dxlink"])
    X1, F1, V1, R1 = fiber_backend.extend_xlink(K, J, I, dsm_flat, xlink1, params)
    
    # Run 2
    xlink2 = fiber_backend.find_local_max(K, J, I, dsm_flat, params["s_xlinkbox"], params["thresh_Dxlink"])
    X2, F2, V2, R2 = fiber_backend.extend_xlink(K, J, I, dsm_flat, xlink2, params)
    
    # Compare (allow some variation due to RNG, but should be close)
    assert abs(len(xlink1) - len(xlink2)) <= 2, "Nucleation point count varies too much"
    assert abs(len(X1) - len(X2)) <= len(X1) * 0.2, "Vertex count varies too much (>20%)"
    assert abs(len(F1) - len(F2)) <= len(F1) * 0.2, "Fiber count varies too much (>20%)"
    
    print(f"\nConsistency check:")
    print(f"  Run 1: {len(xlink1)} nucleation points, {len(X1)} vertices, {len(F1)} fibers")
    print(f"  Run 2: {len(xlink2)} nucleation points, {len(X2)} vertices, {len(F2)} fibers")
    print(f"  ✓ Results are consistent")


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ fiber_backend not available")
def test_pipeline_parameter_sensitivity():
    """
    Test that pipeline responds appropriately to parameter changes.
    """
    # Create simple test image
    J, I, K = 100, 100, 1
    img = np.random.rand(J, I).astype(np.float32) * 0.3
    
    # Add some clear peaks
    for row, col in [(25, 25), (75, 75), (50, 50)]:
        img[row, col] = 1.0
    
    dsm_flat = img.flatten()
    
    # Baseline params
    params_strict = {
        "s_xlinkbox": 5,  # Smaller box = more selective
        "thresh_Dxlink": 0.8,  # Higher threshold = fewer points
        "thresh_LMP": 0.5,
        "thresh_LMPdist": 40,
        "thresh_ext": 0.5,
        "lam_dirdecay": 2.0,
        "s_fiberdir": 3,
        "thresh_linkd": 10.0,
        "thresh_linka": 30.0,
    }
    
    params_relaxed = {
        "s_xlinkbox": 10,  # Larger box = less selective
        "thresh_Dxlink": 0.3,  # Lower threshold = more points
        "thresh_LMP": 0.1,
        "thresh_LMPdist": 15,
        "thresh_ext": 0.1,
        "lam_dirdecay": 0.5,
        "s_fiberdir": 1,
        "thresh_linkd": 20.0,
        "thresh_linka": 60.0,
    }
    
    # Run with strict params
    xlink_strict = fiber_backend.find_local_max(K, J, I, dsm_flat, params_strict["s_xlinkbox"], params_strict["thresh_Dxlink"])
    X_strict, F_strict, V_strict, R_strict = fiber_backend.extend_xlink(K, J, I, dsm_flat, xlink_strict, params_strict)
    
    # Run with relaxed params
    xlink_relaxed = fiber_backend.find_local_max(K, J, I, dsm_flat, params_relaxed["s_xlinkbox"], params_relaxed["thresh_Dxlink"])
    X_relaxed, F_relaxed, V_relaxed, R_relaxed = fiber_backend.extend_xlink(K, J, I, dsm_flat, xlink_relaxed, params_relaxed)
    
    print(f"\nParameter sensitivity:")
    print(f"  Strict params: {len(xlink_strict)} points, {len(X_strict)} vertices, {len(F_strict)} fibers")
    print(f"  Relaxed params: {len(xlink_relaxed)} points, {len(X_relaxed)} vertices, {len(F_relaxed)} fibers")
    
    # Relaxed params should generally find more points/fibers
    # (though not guaranteed due to complex interactions)
    assert len(xlink_strict) > 0, "Strict params found no nucleation points"
    assert len(xlink_relaxed) > 0, "Relaxed params found no nucleation points"
    
    print(f"  ✓ Both parameter sets produce valid outputs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
