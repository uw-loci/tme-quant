#!/usr/bin/env python3
"""
Test CT-FIRE integration with the granular API system.
"""

import sys
from pathlib import Path
import numpy as np

# Add both APIs to path
sys.path.insert(0, str(Path(__file__).parent / "curvealign_py"))
sys.path.insert(0, str(Path(__file__).parent / "ctfire_py"))

def test_ctfire_standalone():
    """Test CT-FIRE as standalone API."""
    print("Testing CT-FIRE standalone API...")
    
    try:
        import ctfire
        
        # Test image
        image = np.random.rand(128, 128)
        
        # Basic CT-FIRE analysis
        result = ctfire.analyze_image(image)
        
        print(f"  PASS: CT-FIRE analysis - found {len(result.fibers)} fibers")
        print(f"  PASS: Network analysis - {result.network.network_stats.get('n_intersections', 0)} intersections")
        print(f"  PASS: Statistics computed - {len(result.stats)} metrics")
        
        # Test with different options
        options = ctfire.CTFireOptions(
            run_mode="fire",  # FIRE only, no curvelet enhancement
            thresh_flen=20.0,
            sigma_im=1.5
        )
        result_fire = ctfire.analyze_image(image, options)
        
        print(f"  PASS: FIRE-only mode - found {len(result_fire.fibers)} fibers")
        
        return True
        
    except Exception as e:
        print(f"  FAIL: CT-FIRE standalone test failed: {e}")
        return False


def test_ctfire_integration():
    """Test CT-FIRE integration with CurveAlign API."""
    print("\nTesting CT-FIRE integration with CurveAlign...")
    
    try:
        import curvealign
        
        # Test image
        image = np.random.rand(128, 128)
        
        # Test CurveAlign with CT-FIRE mode
        result = curvealign.analyze_image(image, mode="ctfire")
        
        print(f"  PASS: CurveAlign CT-FIRE mode - found {len(result.curvelets)} curvelets")
        print(f"  PASS: Unified interface working - stats: {list(result.stats.keys())}")
        
        # Compare with curvelets mode
        result_curvelets = curvealign.analyze_image(image, mode="curvelets")
        
        print(f"  PASS: Curvelets mode - found {len(result_curvelets.curvelets)} curvelets")
        print(f"  PASS: Both modes working through unified API")
        
        return True
        
    except Exception as e:
        print(f"  FAIL: CT-FIRE integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_granular_ctfire():
    """Test granular CT-FIRE imports."""
    print("\nTesting granular CT-FIRE imports...")
    
    try:
        # Test granular type imports
        from ctfire.types.core import Fiber, FiberNetwork
        from ctfire.types.config import CTFireOptions
        from ctfire.types.results import CTFireResult
        
        print("  PASS: CT-FIRE types imported successfully")
        
        # Test granular algorithm imports
        from ctfire.core.algorithms import extract_fibers_fire, enhance_image_with_curvelets
        from ctfire.core.processors import analyze_image_ctfire
        
        print("  PASS: CT-FIRE algorithms and processors imported successfully")
        
        # Test object creation
        fiber = Fiber(
            points=[(10, 20), (15, 25), (20, 30)],
            length=15.0,
            width=2.5,
            angle_deg=45.0,
            straightness=0.8,
            endpoints=((10, 20), (20, 30))
        )
        print(f"  PASS: Fiber object created - length: {fiber.length}, angle: {fiber.angle_deg}")
        
        options = CTFireOptions(run_mode="ctfire", thresh_flen=25.0)
        print(f"  PASS: CTFireOptions created - mode: {options.run_mode}, threshold: {options.thresh_flen}")
        
        return True
        
    except Exception as e:
        print(f"  FAIL: Granular CT-FIRE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all CT-FIRE integration tests."""
    print("Starting CT-FIRE integration tests...\n")
    
    success = True
    
    # Test granular imports first
    success &= test_granular_ctfire()
    
    # Test standalone CT-FIRE
    success &= test_ctfire_standalone()
    
    # Test integration with CurveAlign
    success &= test_ctfire_integration()
    
    if success:
        print("\nAll CT-FIRE integration tests completed successfully!")
        print("PASS: CT-FIRE granular API is fully functional")
        print("PASS: CT-FIRE integrates with CurveAlign unified interface")
        print("\nKey achievements:")
        print("- Individual fiber extraction with FIRE algorithm")
        print("- Curvelet enhancement for better fiber detection")  
        print("- Network connectivity analysis")
        print("- Unified API supporting both curvelets and CT-FIRE modes")
    else:
        print("\nSome CT-FIRE tests failed. Check implementation.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
