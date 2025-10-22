#!/usr/bin/env python3
"""
Test suite for unified CurveAlign + CT-FIRE API integration.

This module tests the integration between CurveAlign and CT-FIRE
APIs to ensure they work together seamlessly.
"""

import sys
from pathlib import Path
import numpy as np

# Add both APIs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "curvealign_py"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ctfire_py"))

def test_unified_api_modes():
    """Test unified API with both analysis modes."""
    print("Testing unified API modes...")
    
    import curvealign_py as curvealign
    
    # Test image
    image = np.random.rand(128, 128)
    
    # Test curvelets mode
    result_curvelets = curvealign.analyze_image(image, mode="curvelets")
    assert hasattr(result_curvelets, 'curvelets')
    print(f"  PASS: Curvelets mode - {len(result_curvelets.curvelets)} curvelets")
    
    # Test CT-FIRE mode
    result_ctfire = curvealign.analyze_image(image, mode="ctfire")
    assert hasattr(result_ctfire, 'curvelets')
    print(f"  PASS: CT-FIRE mode - {len(result_ctfire.curvelets)} curvelets (from fibers)")
    
    # Both should have same result structure
    assert type(result_curvelets) == type(result_ctfire)
    # Stats may differ between modes, but both should have stats
    assert len(result_curvelets.stats) > 0
    assert len(result_ctfire.stats) > 0
    print("  PASS: Unified result structure with mode-specific stats")


def test_mode_comparison():
    """Test comparison between analysis modes."""
    print("\nTesting mode comparison...")
    
    import curvealign_py as curvealign
    
    # Test image with clear structure
    image = np.zeros((100, 100))
    image[40:60, 10:90] = 1.0  # Horizontal fiber
    image[10:90, 40:60] = 1.0  # Vertical fiber
    
    # Compare modes
    result_curvelets = curvealign.analyze_image(image, mode="curvelets")
    result_ctfire = curvealign.analyze_image(image, mode="ctfire")
    
    print(f"  Curvelets mode: {len(result_curvelets.curvelets)} features")
    print(f"  CT-FIRE mode: {len(result_ctfire.curvelets)} features")
    
    # Check if we're using real CurveLab or placeholder
    import os
    using_real_curvelab = os.getenv('TMEQ_RUN_CURVELETS', '0') == '1'
    
    if using_real_curvelab:
        # With real CurveLab, we should detect features
        assert len(result_curvelets.curvelets) > 0 or len(result_ctfire.curvelets) > 0, \
            "Real CurveLab should detect features in structured image"
        print("  PASS: Real CurveLab detected features")
    else:
        # With placeholder, empty results are acceptable
        assert isinstance(result_curvelets.curvelets, list)
        assert isinstance(result_ctfire.curvelets, list)
        print("  PASS: Placeholder mode - empty results acceptable")


def test_batch_processing():
    """Test batch processing with both modes."""
    print("\nTesting batch processing...")
    
    import curvealign_py as curvealign
    
    # Create test images
    images = [np.random.rand(64, 64) for _ in range(3)]
    
    # Test batch with curvelets
    results_curvelets = curvealign.batch_analyze(images, mode="curvelets")
    assert len(results_curvelets) == 3
    print(f"  PASS: Batch curvelets - processed {len(results_curvelets)} images")
    
    # Test batch with CT-FIRE
    results_ctfire = curvealign.batch_analyze(images, mode="ctfire")
    assert len(results_ctfire) == 3
    print(f"  PASS: Batch CT-FIRE - processed {len(results_ctfire)} images")


def main():
    """Run all unified API tests."""
    print("Starting unified API integration tests...\n")
    
    try:
        test_unified_api_modes()
        test_mode_comparison()
        test_batch_processing()
        
        print("\nAll unified API tests completed successfully!")
        print("PASS: CurveAlign + CT-FIRE integration working")
        return True
        
    except Exception as e:
        print(f"\nUnified API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
