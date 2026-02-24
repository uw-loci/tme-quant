"""
Example 3: Fiber-based cell guidance analysis
"""

params = TMEAnalysisParams(
    mode=AnalysisMode.FIBER_BASED,
    interaction_strategy=InteractionStrategy.NEAREST,  # Find nearest cell
    cell_fiber_distance=50.0,
    compute_orientation=True,
    compute_spatial=True
)

result = tme_analyzer.analyze(
    cells=cells,
    fibers=fibers,
    params=params,
    analysis_id="fiber_guidance_001"
)

# Analyze fiber alignment
orientation = result.orientation_features

print(f"Fiber-cell alignment:")
print(f"  Parallel: {orientation['parallel_ratio']:.1%}")
print(f"  Perpendicular: {orientation['perpendicular_ratio']:.1%}")
print(f"  Coherence: {orientation['orientation_coherence']:.3f}")