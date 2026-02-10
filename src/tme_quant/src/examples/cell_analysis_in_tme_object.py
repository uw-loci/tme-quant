# tme_quant/src/examples/cell_analysis_in_tme_object.py
"""
Example 3: Integrate cell analysis into TME project hierarchy
"""

from tme_quant.core.project import TMEProject
from tme_quant.cell_analysis.core.cell_analyzer import CellAnalyzer
from tme_quant.core.tme_models.cell_model import create_cell_object_from_segmentation

# Create TME project
project = TMEProject(name="Breast_Cancer_Study")

# Add image
project.add_image(
    image_id="patient_001",
    image_path="data/patient_001.tif",
    channels={'nuclei': 0, 'cd3': 1, 'cd8': 2, 'collagen': 3},
    pixel_size=(0.5, 0.5)
)

# Define tumor region (from manual annotation or auto-detection)
tumor_region_id = "tumor_1"
# ... add tumor region to project ...

# Segment cells in tumor region
analyzer = CellAnalyzer()
seg_result = analyzer.segment_cells_2d(
    tumor_region_image,
    seg_params,
    image_id=f"{image_id}_{tumor_region_id}"
)

# Convert segmented cells to CellObjects and add to hierarchy
for cell_props in seg_result.cells:
    cell_obj = create_cell_object_from_segmentation(
        cell_props,
        parent_id=tumor_region_id,
        region_type="tumor",
        pixel_size=0.5
    )
    
    # Add to project hierarchy
    project.add_object(cell_obj, parent_id=tumor_region_id)

# Classify cells
class_result = analyzer.classify_cells(seg_result, class_params, image)

# Update cell objects with classifications
cells_in_tumor = project.get_cells_in_region(tumor_region_id)
for cell in cells_in_tumor:
    cell_id = int(cell.object_id.split('_')[-1])
    if cell_id in class_result.cell_types:
        cell.cell_type = class_result.cell_types[cell_id]
        cell.cell_type_confidence = class_result.confidences[cell_id]

# Compute spatial relationships
project.compute_cell_neighbors(
    tumor_region_id,
    max_distance=50.0
)

# Query cells by type
tumor_cells = project.get_cells_by_criteria(
    cell_type=CellType.TUMOR,
    in_tumor_region=True
)
t_cells = project.get_cells_by_criteria(
    cell_type=CellType.T_CELL,
    in_tumor_region=True
)

print(f"Tumor cells: {len(tumor_cells)}")
print(f"T cells: {len(t_cells)}")
print(f"T cell to tumor ratio: {len(t_cells) / len(tumor_cells):.3f}")

# Export project
project.export_cell_analysis_results(
    output_dir="output/project/",
    formats=["csv", "excel"]
)