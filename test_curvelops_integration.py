#!/usr/bin/env python3
"""
Test script for Curvelops integration in CurveAlign Python API.
"""

import numpy as np
import curvealign_py as curvealign

def test_curvelops_integration():
    """Test the Curvelops integration with the CurveAlign API."""
    print("=== Testing Curvelops Integration ===")
    
    # Check Curvelops status
    from curvealign_py.core.algorithms.fdct_wrapper import get_curvelops_status
    status = get_curvelops_status()
    
    print(f"Curvelops Status:")
    print(f"  Available: {status['available']}")
    print(f"  Backend: {status['backend']}")
    if 'version' in status and status['version']:
        print(f"  Version: {status['version']}")
    if 'functional' in status:
        print(f"  Functional: {status['functional']}")
    if 'error' in status:
        print(f"  Error: {status['error']}")
    
    # Create test image
    print(f"\n=== Testing FDCT Functions ===")
    image = np.random.rand(128, 128)
    print(f"Test image shape: {image.shape}")
    
    # Test forward FDCT
    print("Testing forward FDCT...")
    try:
        from curvealign_py.core.algorithms.fdct_wrapper import apply_fdct
        coeffs = apply_fdct(image)
        print(f"  Forward FDCT successful: {len(coeffs)} scales")
        for i, scale in enumerate(coeffs):
            print(f"    Scale {i}: {len(scale)} wedges")
    except Exception as e:
        print(f"  Forward FDCT failed: {e}")
        return False
    
    # Test inverse FDCT
    print("Testing inverse FDCT...")
    try:
        from curvealign_py.core.algorithms.fdct_wrapper import apply_ifdct
        reconstructed = apply_ifdct(coeffs, img_shape=image.shape)
        print(f"  Inverse FDCT successful: {reconstructed.shape}")
        
        # Check reconstruction quality
        mse = np.mean((image - reconstructed)**2)
        print(f"  Reconstruction MSE: {mse:.6f}")
        
    except Exception as e:
        print(f"  Inverse FDCT failed: {e}")
        return False
    
    # Test parameter extraction
    print("Testing parameter extraction...")
    try:
        from curvealign_py.core.algorithms.fdct_wrapper import extract_parameters
        X_rows, Y_cols = extract_parameters(coeffs, img_shape=image.shape)
        print(f"  Parameter extraction successful: {len(X_rows)} scales")
    except Exception as e:
        print(f"  Parameter extraction failed: {e}")
        return False
    
    # Test full CurveAlign analysis
    print(f"\n=== Testing Full CurveAlign Analysis ===")
    try:
        result = curvealign.analyze_image(image)
        print(f"  Analysis successful: {len(result.curvelets)} curvelets detected")
        print(f"  Mean angle: {result.stats['mean_angle']:.1f}°")
        print(f"  Alignment: {result.stats['alignment']:.3f}")
        
        # Test visualization
        overlay = curvealign.overlay(image, result.curvelets)
        print(f"  Visualization successful: {overlay.shape}")
        
    except Exception as e:
        print(f"  Full analysis failed: {e}")
        return False
    
    print(f"\n=== All Tests Passed! ===")
    return True

if __name__ == "__main__":
    success = test_curvelops_integration()
    exit(0 if success else 1)
