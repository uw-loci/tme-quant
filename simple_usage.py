#!/usr/bin/env python3
"""
Simple usage examples for pycurvelets (manually converted API from MATLAB CurveAlign).

Requires: curvelops (for curvelet transform), pycurvelets (this package).
Run from repo root: python simple_usage.py
"""
import numpy as np

try:
    from pycurvelets.models import (
        CurveletControlParameters,
        FeatureControlParameters,
        ImageInputParameters,
        BoundaryParameters,
        FiberAnalysisParameters,
        OutputControlParameters,
        AdvancedAnalysisOptions,
    )
    from pycurvelets.get_ct import get_ct
    from pycurvelets.new_curv import new_curv
    from pycurvelets.process_image import process_image
    HAS_PYCURVELETS = True
except ImportError as e:
    HAS_PYCURVELETS = False
    print(f"pycurvelets not available: {e}")


def example_get_ct():
    """Extract curvelets from an image using get_ct."""
    if not HAS_PYCURVELETS:
        return
    # Create a simple test image (e.g. 128x128)
    img = np.random.rand(128, 128).astype(np.float64) * 255
    curve_cp = CurveletControlParameters(keep=0.05, scale=1.0, radius=10.0)
    feature_cp = FeatureControlParameters(
        minimum_nearest_fibers=2,
        minimum_box_size=32,
        fiber_midpoint_estimate=1,
    )
    fiber_structure, density_df, alignment_df, _ = get_ct(img, curve_cp, feature_cp)
    print(f"get_ct: {len(fiber_structure)} curvelets extracted")
    if len(fiber_structure) > 0:
        print(f"  angles: min={fiber_structure['angle'].min():.1f}, max={fiber_structure['angle'].max():.1f}")


def example_new_curv():
    """Extract curvelets using new_curv (lower-level)."""
    if not HAS_PYCURVELETS:
        return
    img = np.random.rand(64, 64).astype(np.float64) * 255
    curve_cp = CurveletControlParameters(keep=0.1, scale=1.0, radius=5.0)
    in_curves, coeffs, inc = new_curv(img, curve_cp)
    print(f"new_curv: {len(in_curves)} curvelets, inc={inc:.4f}")


def example_process_image():
    """Run full process_image pipeline (requires curvelops)."""
    if not HAS_PYCURVELETS:
        return
    import tempfile
    import os
    img = np.random.rand(128, 128).astype(np.float64) * 255
    with tempfile.TemporaryDirectory() as tmp:
        image_params = ImageInputParameters(img=img, img_name="test")
        fiber_params = FiberAnalysisParameters(fiber_mode=0, keep=0.05)
        output_params = OutputControlParameters(
            output_directory=tmp,
            make_associations=False,
            make_map=False,
            make_overlay=False,
            make_feature_file=True,
        )
        result = process_image(image_params, fiber_params, output_params)
        if result and "fib_feat_df" in result:
            print(f"process_image: wrote features, {len(result['fib_feat_df'])} rows")
        else:
            print("process_image: no result (curvelops may be required)")


if __name__ == "__main__":
    print("pycurvelets simple usage examples\n" + "=" * 40)
    example_get_ct()
    example_new_curv()
    example_process_image()
    print("\nDone.")
