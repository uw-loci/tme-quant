"""
Example 4: Analysis within custom user-defined ROIs
"""

from shapely.geometry import box

# Define custom ROI (e.g., manually annotated region)
custom_roi = box(100, 100, 500, 500)  # Bounding box

custom_rois = [
    ROI.from_shapely(custom_roi, roi_id="invasive_front")
]

params = TMEAnalysisParams(
    mode=AnalysisMode.ROI_BASED,
    interaction_distance=50.0,
    compute_tacs=False,  # Not tumor-based
    compute_morphology=True,
    compute_density=True
)

result = tme_analyzer.analyze(
    cells=cells,
    fibers=fibers,
    custom_rois=custom_rois,
    params=params,
    analysis_id="roi_custom_001"
)

print(f"ROI Analysis:")
print(f"  Cell density: {result.density_features['cell_density']:.2f} cells/mm²")
print(f"  Fiber density: {result.density_features['fiber_density']:.2f} fibers/mm²")