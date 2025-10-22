#!/usr/bin/env python3
"""
Basic Curvelops integration test for CI.
Tests core functionality without brittle numerical validations.
"""

import numpy as np
import curvealign_py as curvealign


def test_curvelops_installation():
    """Test that Curvelops can be imported and basic status check works."""
    from curvealign_py.core.algorithms.fdct_wrapper import get_curvelops_status
    
    status = get_curvelops_status()
    
    # Test that status check works (regardless of availability)
    assert isinstance(status, dict)
    assert 'available' in status
    assert 'backend' in status
    
    # If available, should have version and functional info
    if status['available']:
        assert 'version' in status or 'functional' in status


def test_curvelops_api_functions():
    """Test that Curvelops API functions can be called without errors."""
    from curvealign_py.core.algorithms.fdct_wrapper import apply_fdct, apply_ifdct
    
    # Create simple test image
    image = np.random.rand(32, 32)
    
    # Test forward transform
    coeffs = apply_fdct(image)
    assert isinstance(coeffs, list)
    assert len(coeffs) > 0
    
    # Test inverse transform
    reconstructed = apply_ifdct(coeffs, img_shape=image.shape)
    assert reconstructed.shape == image.shape
    assert isinstance(reconstructed, np.ndarray)


def test_curvelops_integration():
    """Test that Curvelops integrates with main API."""
    import os
    using_real_curvelab = os.getenv('TMEQ_RUN_CURVELETS', '0') == '1'
    
    # Create simple test image
    image = np.random.rand(64, 64)
    
    # Test main API works (regardless of backend)
    result = curvealign.analyze_image(image)
    
    # Test result structure
    assert hasattr(result, 'curvelets')
    assert hasattr(result, 'stats')
    assert isinstance(result.curvelets, list)
    assert isinstance(result.stats, dict)
    
    # Test that stats contain expected keys
    expected_stats = ['mean_angle', 'std_angle', 'alignment', 'density', 'total_curvelets']
    for stat in expected_stats:
        assert stat in result.stats
    
    if using_real_curvelab:
        # With real CurveLab, we might get some curvelets
        print(f"  Real CurveLab mode: {len(result.curvelets)} curvelets detected")
    else:
        # With placeholder, empty results are expected
        print(f"  Placeholder mode: {len(result.curvelets)} curvelets (expected)")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
