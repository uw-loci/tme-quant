"""
Complete integration example showing all modules working together.
"""

from tme_quant.core.project import TMEProject
from tme_quant.cell_analysis import CellAnalyzer
from tme_quant.fiber_analysis import FiberAnalyzer
from tme_quant.tme_analysis import TMEAnalyzer

# ============================================================
# STEP 1: Project Setup
# ============================================================

project = TMEProject(name="Complete_TME_Analysis")

project.add_image(
    image_id="sample_001",
    channels={'nuclei': 0, 'collagen': 1},
    pixel_size=(0.5, 0.5)
)

# ============================================================
# STEP 2: Cell Analysis (Using cell_analysis module)
# ============================================================

cell_analyzer = CellAnalyzer()

# Segment
seg_result = cell_analyzer.segment_cells_2d(
    cell_image, seg_params
)

# Classify
class_result = cell_analyzer.classify_cells(
    seg_result, class_params, image
)

# Quantify
quant_result = cell_analyzer.quantify_cells(
    seg_result, quant_params, image
)

# Add to project
for cell_props in seg_result.cells:
    cell_obj = create_cell_object_from_segmentation(
        cell_props, parent_id="sample_001"
    )
    # Add classification
    cell_id = cell_props.cell_id
    if cell_id in class_result.cell_types:
        cell_obj.cell_type = class_result.cell_types[cell_id]
    
    project.add_object(cell_obj)

# ============================================================
# STEP 3: Fiber Analysis (Using fiber_analysis module)
# ============================================================

fiber_analyzer = FiberAnalyzer()

# Extract
fiber_result = fiber_analyzer.extract_fibers_2d(
    fiber_image, extract_params
)

# Add to project
for fiber_props in fiber_result.fibers:
    fiber_obj = create_fiber_object_from_extraction(
        fiber_props, parent_id="sample_001"
    )
    project.add_object(fiber_obj)

# ============================================================
# STEP 4: TME Analysis (Using NEW tme_analysis module)
# ============================================================

tme_analyzer = TMEAnalyzer()

# Detect tumor regions
tumor_regions = tme_analyzer.detect_tumor_regions(
    project.get_objects_by_type("cell"),
    tumor_params
)

for tumor in tumor_regions:
    project.add_object(tumor)

# Run TACS analysis
tacs_result = tme_analyzer.analyze(
    cells=project.get_objects_by_type("cell"),
    fibers=project.get_objects_by_type("fiber"),
    tumor_regions=tumor_regions,
    params=tacs_params
)

# ============================================================
# STEP 5: Export Everything
# ============================================================

# Export project
project.save("output/project.tme")

# Export cell analysis
cell_analyzer.export_results("output/cells/")

# Export fiber analysis
fiber_analyzer.export_results("output/fibers/")

# Export TME analysis
tme_analyzer.export_results("output/tme/")

print("Complete TME analysis pipeline finished!")