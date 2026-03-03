'''Example 1: Simple Import'''
from tme_quant.tme_analysis import TMEAnalyzer

analyzer = TMEAnalyzer(verbose=True)
result = analyzer.analyze(cells, fibers, tumor_regions, params)

'''Example 2: Import Configuration'''
from tme_quant.tme_analysis.config import (
    TMEAnalysisParams,
    AnalysisMode,
    InteractionStrategy,
    TumorDetectionParams,
    TumorDetectionMethod
)

params = TMEAnalysisParams(
    mode=AnalysisMode.TUMOR_BASED,
    interaction_strategy=InteractionStrategy.RADIUS,
    tumor_boundary_distance=100.0
)

'''Example 3: Import Utilities'''
from tme_quant.tme_analysis.utils import (
    compute_pairwise_distances,
    compute_circular_statistics,
    compute_region_centroid
)

# Use utilities
distances = compute_pairwise_distances(cell_points, fiber_points)

'''Example 4: Import Export'''
from tme_quant.tme_analysis.io import export_tme_analysis_results

# Export results
export_paths = export_tme_analysis_results(
    result,
    output_dir="output/",
    formats=["csv", "excel", "json", "geojson"]
)

'''Example 5: Direct Component Access'''
from tme_quant.tme_analysis.core import (
    InteractionDetector,
    RegionManager,
    MeasurementEngine
)

# Use components directly for advanced workflows
detector = InteractionDetector()
pairs = detector.detect_fiber_tumor_interactions(fibers, tumors)

'''Example 6: Installation Verification'''
# Test imports
import tme_quant.tme_analysis
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import TMEAnalysisParams
from tme_quant.tme_analysis.io import TMEAnalysisExporter

# Check version
print(tme_quant.tme_analysis.__version__)  # Should print: 0.1.0

# Verify main class
analyzer = TMEAnalyzer()
print(type(analyzer))  # Should print: <class 'tme_quant.tme_analysis.core.tme_analyzer.TMEAnalyzer'>

'''Example 7: Complete Workflow Integration'''
from tme_quant.tme_analysis import TMEAnalyzer
from tme_quant.tme_analysis.config import TMEAnalysisParams, AnalysisMode
from tme_quant.tme_analysis.io import export_tme_analysis_results

# Initialize
analyzer = TMEAnalyzer(verbose=True)

# Configure
params = TMEAnalysisParams(
    mode=AnalysisMode.TUMOR_BASED,
    tumor_boundary_distance=100.0,
    compute_tacs=True,
    compute_prognostic=True
)

# Analyze
result = analyzer.analyze(cells, fibers, tumor_regions, params)

# Export (NEW - all formats)
export_paths = export_tme_analysis_results(
    result,
    output_dir="output/",
    formats=["csv", "excel", "json", "geojson"],
    prefix="patient_001"
)

# Output files created:
# - patient_001_interactions.csv
# - patient_001_tacs.csv
# - patient_001_prognostic.csv
# - patient_001_analysis.xlsx (8 sheets)
# - patient_001_results.json
# - patient_001_interactions.geojson