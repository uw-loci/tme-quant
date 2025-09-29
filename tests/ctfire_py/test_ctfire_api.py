#!/usr/bin/env python3
"""
Test suite for CT-FIRE core API functionality.
"""

import numpy as np
from pathlib import Path

def test_ctfire_types():
    """Test CT-FIRE type definitions."""
    print("Testing CT-FIRE types...")
    
    from ctfire_py.types.core import Fiber, FiberNetwork, FiberGraph
    from ctfire_py.types.config import CTFireOptions
    from ctfire_py.types.results import CTFireResult, FiberMetrics
    
    # Test Fiber creation
    fiber = Fiber(
        points=[(10, 20), (15, 25), (20, 30)],
        length=15.0,
        width=2.5,
        angle_deg=45.0,
        straightness=0.8,
        endpoints=((10, 20), (20, 30)),
        curvature=0.2
    )
    assert fiber.length == 15.0
    assert fiber.angle_deg == 45.0
    print("  PASS: Fiber type working")
    
    # Test CTFireOptions
    options = CTFireOptions(
        run_mode="ctfire",
        thresh_flen=30.0,
        sigma_im=1.2
    )
    assert options.run_mode == "ctfire"
    assert options.thresh_flen == 30.0
    print("  PASS: CTFireOptions type working")
    
    print("  PASS: All CT-FIRE types working")


def test_ctfire_algorithms():
    """Test CT-FIRE algorithm functions."""
    print("\nTesting CT-FIRE algorithms...")
    
    from ctfire_py.core.algorithms import extract_fibers_fire, enhance_image_with_curvelets
    from ctfire_py.types.config import CTFireOptions
    
    # Test image
    test_image = np.random.rand(64, 64)
    
    # Test curvelet enhancement
    try:
        enhanced = enhance_image_with_curvelets(test_image, keep=0.01)
        assert enhanced.shape == test_image.shape
        print("  PASS: Curvelet enhancement working")
    except ImportError:
        print("  SKIP: Curvelet enhancement - curvealign not available")
    
    # Test FIRE algorithm
    options = CTFireOptions(thresh_flen=10.0, thresh_numv=3)
    fibers = extract_fibers_fire(test_image, options)
    assert isinstance(fibers, list)
    print(f"  PASS: FIRE algorithm working - extracted {len(fibers)} fibers")
    
    print("  PASS: All CT-FIRE algorithms working")


def test_ctfire_processors():
    """Test CT-FIRE processor functions."""
    print("\nTesting CT-FIRE processors...")
    
    from ctfire_py.core.processors import (
        analyze_image_ctfire, analyze_fiber_network,
        compute_fiber_metrics, compute_ctfire_statistics
    )
    
    # Test image
    test_image = np.random.rand(64, 64)
    
    # Test main processor
    result = analyze_image_ctfire(test_image)
    assert hasattr(result, 'fibers')
    assert hasattr(result, 'network')
    assert hasattr(result, 'stats')
    print(f"  PASS: Main CT-FIRE processor - found {len(result.fibers)} fibers")
    
    # Test network analysis
    if result.fibers:
        network = analyze_fiber_network(result.fibers)
        assert hasattr(network, 'intersections')
        assert hasattr(network, 'connectivity')
        print(f"  PASS: Network analysis - {len(network.intersections)} intersections")
        
        # Test metrics computation
        metrics = compute_fiber_metrics(result.fibers)
        assert len(metrics.lengths) == len(result.fibers)
        print(f"  PASS: Fiber metrics - {len(metrics.lengths)} fibers measured")
    
    print("  PASS: All CT-FIRE processors working")


def main():
    """Run all CT-FIRE core tests."""
    print("Starting CT-FIRE core API tests...\n")
    
    try:
        test_ctfire_types()
        test_ctfire_algorithms()
        test_ctfire_processors()
        
        print("\nAll CT-FIRE core tests completed successfully!")
        print("PASS: CT-FIRE API is fully functional")
        return True
        
    except Exception as e:
        print(f"\nCT-FIRE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
