"""
Complete workflow: Analyze fiber orientation with respect to tumor boundary.
"""

from tme_quant.core.project import TMEProject
from tme_quant.core.tme_models.tumor_model import TumorRegion
from tme_quant.fiber_analysis.config.orientation_params import (
    OrientationParams, OrientationMode
)
from tme_quant.fiber_analysis.config.extraction_params import (
    ExtractionParams, ExtractionMode
)

# ============================================================
# STEP 1: Create project and load image
# ============================================================

project = TMEProject(name="Breast_Cancer_Study")

# Add image to project
image_id = "patient_001_slide_01"
project.add_image(
    image_id=image_id,
    image_path="data/patient_001/slide_01.tif",
    channels={
        'nuclei': 0,
        'collagen': 1,
        'cd8': 2
    }
)

# ============================================================
# STEP 2: Define tumor region
# ============================================================

# Create tumor region ROI (from manual annotation or segmentation)
tumor_roi = create_roi_from_polygon(tumor_boundary_points)

tumor_region = TumorRegion(
    object_id="tumor_1",
    parent_id=image_id,
    roi=tumor_roi,
    region_type="invasive_carcinoma"
)

project.add_object(tumor_region, parent_id=image_id)

# ============================================================
# STEP 3: Analyze fiber orientation at tumor boundary
# ============================================================

# Define boundary region (50 micron zone around tumor)
boundary_width = 50.0  # microns
boundary_roi = project._get_or_create_boundary_roi(
    tumor_region, boundary_width
)

# Set orientation analysis parameters
orientation_params = OrientationParams(
    mode=OrientationMode.CURVEALIGN,
    window_size=128,
    curvelet_levels=5,
    curvelet_angles=16,
    pixel_size=0.5,  # microns per pixel
    compute_coherency=True,
    compute_energy=True
)

# Analyze orientation in tumor boundary
boundary_orientation = project.analyze_region_orientation(
    image_id=image_id,
    region_id="tumor_1_boundary",
    params=orientation_params,
    region_type="tumor_boundary"
)

print(f"Tumor boundary mean orientation: {boundary_orientation.get_dominant_orientation():.2f}°")
print(f"Alignment score: {boundary_orientation.get_alignment_score():.3f}")

# ============================================================
# STEP 4: Extract individual fibers at tumor boundary
# ============================================================

extraction_params = ExtractionParams(
    mode=ExtractionMode.CTFIRE,
    pixel_size=0.5,
    min_fiber_length=10.0,  # microns
    ctfire_threshold=0.1,
    straightness_threshold=0.5
)

boundary_fibers = project.extract_region_fibers(
    image_id=image_id,
    region_id="tumor_1_boundary",
    params=extraction_params,
    region_type="tumor_boundary"
)

print(f"Extracted {len(boundary_fibers)} fibers from tumor boundary")

# ============================================================
# STEP 5: Analyze fiber alignment to tumor boundary
# ============================================================

alignment_results = project.analyze_fiber_alignment_to_tumor_boundary(
    tumor_region_id="tumor_1",
    boundary_width=50.0
)

print(f"\nFiber-Tumor Boundary Alignment Analysis:")
print(f"Total fibers: {alignment_results['total_fibers']}")
print(f"Boundary fibers: {alignment_results['boundary_fibers']}")
print(f"Core fibers: {alignment_results['core_fibers']}")
print(f"Mean alignment angle: {alignment_results['mean_alignment_angle']:.2f}°")
print(f"Parallel ratio: {alignment_results['parallel_ratio']:.3f}")
print(f"Perpendicular ratio: {alignment_results['perpendicular_ratio']:.3f}")
print(f"TACS type: {alignment_results['tacs_type']}")
print(f"TACS score: {alignment_results['tacs_score']:.3f}")

# ============================================================
# STEP 6: Compare orientation across regions
# ============================================================

# Also analyze tumor core and stroma
core_orientation = project.analyze_region_orientation(
    image_id=image_id,
    region_id="tumor_1_core",
    params=orientation_params,
    region_type="tumor_core"
)

stroma_orientation = project.analyze_region_orientation(
    image_id=image_id,
    region_id="stroma_1",
    params=orientation_params,
    region_type="stroma"
)

# Compare orientations
comparison = project.compare_orientation_across_regions([
    "tumor_1_boundary",
    "tumor_1_core",
    "stroma_1"
])

print(f"\nOrientation Comparison:")
for region_id, data in comparison.items():
    if isinstance(data, dict):
        print(f"{region_id}: {data['mean_orientation']:.2f}° (alignment: {data['alignment_score']:.3f})")

# ============================================================
# STEP 7: Query fibers by criteria
# ============================================================

# Get long, straight fibers in tumor boundary
long_straight_boundary_fibers = project.get_fibers_by_criteria(
    min_length=20.0,
    min_straightness=0.8,
    in_tumor_boundary=True
)

print(f"\nLong straight fibers in boundary: {len(long_straight_boundary_fibers)}")

# Get perpendicular fibers (TACS-3 candidates)
perpendicular_fibers = [
    f for f in alignment_results['fibers']['boundary']
    if f.angle_to_tumor_boundary is not None
    and abs(f.angle_to_tumor_boundary) > 60
]

print(f"Perpendicular fibers (TACS-3): {len(perpendicular_fibers)}")

# ============================================================
# STEP 8: Export results
# ============================================================

# Export fiber data
project.fiber_analyzer.export_results(
    output_dir=f"results/{image_id}/fibers/",
    formats=["csv", "excel", "geojson"],
    prefix="tumor_boundary_fibers"
)

# Export orientation maps
project.export_orientation_maps(
    output_dir=f"results/{image_id}/orientation/",
    formats=["csv", "json"]
)

# ============================================================
# STEP 9: Visualize in Napari (optional)
# ============================================================

from napari_curvealign.workflows.combined_cell_fiber_analysis import (
    CombinedCellFiberAnalysis
)

# Create Napari workflow
napari_workflow = CombinedCellFiberAnalysis(project)

# Visualize tumor boundary with fibers and orientation
napari_workflow.visualize_fiber_tumor_interaction(
    tumor_region_id="tumor_1",
    show_boundary=True,
    show_fibers=True,
    show_orientation_map=True,
    color_by_alignment=True
)