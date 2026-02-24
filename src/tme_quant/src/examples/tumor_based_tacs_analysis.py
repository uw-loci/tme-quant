"""
Example 2: Tumor-based TACS analysis - Complete workflow
"""

from tme_quant.core.project import TMEProject
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams,
    TumorDetectionParams,
    AnalysisMode,
    TumorDetectionMethod
)

# Create project
project = TMEProject(name="TACS_Study")

# Add image
project.add_image(
    image_id="patient_001",
    image_path="data/patient_001.tif",
    channels={'nuclei': 0, 'collagen': 1},
    pixel_size=(0.5, 0.5)
)

# Segment cells (using project integration)
from tme_quant.cell_analysis import CellAnalyzer
cell_analyzer = CellAnalyzer()
cell_result = cell_analyzer.segment_cells_2d(
    cell_image, seg_params, image_id="patient_001"
)

# Add cells to project
for cell_props in cell_result.cells:
    cell_obj = create_cell_object_from_segmentation(
        cell_props, parent_id="patient_001"
    )
    project.add_object(cell_obj)

# Get cells from project
cells = project.get_objects_by_type("cell")

# Detect tumor regions automatically
tme_analyzer = TMEAnalyzer(verbose=True)

tumor_params = TumorDetectionParams(
    method=TumorDetectionMethod.CLUSTERING,
    dbscan_eps=100.0,
    dbscan_min_samples=10,
    min_tumor_area=1000.0,
    smooth_boundary=True
)

tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)

print(f"Detected {len(tumor_regions)} tumor regions")

# Add tumor regions to project
for tumor in tumor_regions:
    project.add_object(tumor)

# Extract fibers
fiber_result = project.extract_region_fibers(
    image_id="patient_001",
    region_id=tumor_regions[0].object_id,
    params=fiber_extract_params
)

fibers = project.get_fibers_in_region(tumor_regions[0].object_id)

# Configure TACS analysis
tacs_params = TMEAnalysisParams(
    mode=AnalysisMode.TUMOR_BASED,
    tumor_boundary_distance=100.0,  # 100 µm boundary zone
    compute_tacs=True,
    compute_morphology=True,
    compute_spatial=True,
    compute_prognostic=True,
    generate_zones=True,
    invasive_margin_width=50.0,
    stroma_width=200.0
)

# Run TACS analysis
tacs_result = tme_analyzer.analyze(
    cells=cells,
    fibers=fibers,
    tumor_regions=tumor_regions,
    params=tacs_params,
    analysis_id="tacs_patient_001"
)

# Access TACS features
tacs = tacs_result.tacs_features

print("\n=== TACS Analysis Results ===")
print(f"Total boundary fibers: {tacs['total_boundary_fibers']}")
print(f"TACS-1 (Random): {tacs['tacs1_ratio']:.1%} ({tacs['tacs1_count']})")
print(f"TACS-2 (Parallel): {tacs['tacs2_ratio']:.1%} ({tacs['tacs2_count']})")
print(f"TACS-3 (Perpendicular): {tacs['tacs3_ratio']:.1%} ({tacs['tacs3_count']})")
print(f"Dominant TACS: {tacs['dominant_tacs_type']}")

# Access prognostic scores
prog = tacs_result.prognostic_scores

print("\n=== Prognostic Scores ===")
print(f"Collagen Prognostic Score: {prog['collagen_prognostic_score']:.3f}")
print(f"TACS-3 Prognostic: {prog['tacs3_prognostic']:.3f}")
print(f"TME Interaction Score: {prog['tme_interaction_score']:.3f}")
print(f"Invasive Potential: {prog['invasive_potential_score']:.3f}")
print(f"Overall TME Risk: {prog['overall_tme_risk_score']:.3f}")

# Export results
tme_analyzer.export_results(
    output_dir="output/tacs_analysis/",
    formats=["csv", "excel", "json"]
)

# Visualize in Napari
import napari

viewer = napari.Viewer()

# Add collagen image
viewer.add_image(collagen_image, name='Collagen', colormap='gray')

# Add tumor boundary
tumor_mask = tumor_regions[0].roi.get_mask(collagen_image.shape)
viewer.add_labels(tumor_mask, name='Tumor Boundary')

# Add TACS-classified fibers
tacs_colors = {
    'TACS-1': 'blue',
    'TACS-2': 'green',
    'TACS-3': 'red',
}

fiber_shapes = []
fiber_colors = []

for pair in tacs_result.interaction_pairs:
    if pair.target_type == "tumor_boundary":
        # Get fiber
        fiber = next(f for f in fibers if f.object_id == pair.source_id)
        
        fiber_shapes.append(fiber.centerline)
        fiber_colors.append(tacs_colors.get(pair.interaction_type, 'gray'))

viewer.add_shapes(
    fiber_shapes,
    shape_type='path',
    edge_color=fiber_colors,
    edge_width=2,
    name='TACS Fibers'
)

napari.run()