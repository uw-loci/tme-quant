"""
TMEQuant Platform - Complete Usage Examples

Demonstrates full workflow from image acquisition to TME analysis.
"""

# ============================================================
# EXAMPLE 1: Complete H&E-SHG TME Analysis Workflow
# ============================================================

def example_1_complete_he_shg_workflow():
    """
    Complete workflow: Registration → Cell Analysis → Fiber Analysis → TME Analysis
    
    This is the primary use case for TMEQuant.
    """
    
    import numpy as np
    from skimage import io
    
    # Import all modules
    from tme_quant.image_registration import RegistrationManager
    from tme_quant.image_registration.config import RegistrationParams, RegistrationMethod
    from tme_quant.cell_analysis import CellAnalyzer
    from tme_quant.cell_analysis.config import SegmentationParams, SegmentationMode
    from tme_quant.fiber_analysis import FiberAnalyzer
    from tme_quant.fiber_analysis.config import ExtractionParams, ExtractionMode
    from tme_quant.tme_analysis import TMEAnalyzer
    from tme_quant.tme_analysis.config import TMEAnalysisParams, AnalysisMode
    
    print("="*60)
    print("EXAMPLE 1: Complete H&E-SHG TME Analysis")
    print("="*60)
    
    # ============================================================
    # STEP 1: Load Images
    # ============================================================
    
    print("\n[1/6] Loading images...")
    
    he_image = io.imread("data/patient_001_HE.tif")          # H&E brightfield
    shg_image = io.imread("data/patient_001_SHG.tif")        # SHG collagen
    
    print(f"  ✓ H&E image: {he_image.shape}")
    print(f"  ✓ SHG image: {shg_image.shape}")
    
    # ============================================================
    # STEP 2: Register H&E to SHG
    # ============================================================
    
    print("\n[2/6] Registering H&E to SHG...")
    
    reg_params = RegistrationParams(
        method=RegistrationMethod.HE_SHG,  # Keikhosravi 2020 method
        transform_type=TransformType.AFFINE,
        use_multiresolution=True,
        pyramid_levels=3,
        num_iterations=200
    )
    
    registration_manager = RegistrationManager(verbose=True)
    reg_result = registration_manager.register(
        fixed_image=shg_image,      # Reference
        moving_image=he_image,       # To align
        params=reg_params
    )
    
    registered_he = reg_result.registered_image
    transform = reg_result.transform
    
    print(f"  ✓ Registration complete")
    print(f"  MI score: {reg_result.mutual_information:.4f}")
    
    # Save registered image
    io.imsave("output/patient_001_HE_registered.tif", registered_he)
    
    # Visualize registration quality
    from tme_quant.image_registration.visualization import create_checkerboard
    checker = create_checkerboard(shg_image, registered_he, num_squares=10)
    io.imsave("output/patient_001_registration_checker.png", checker)
    
    # ============================================================
    # STEP 3: Cell Segmentation
    # ============================================================
    
    print("\n[3/6] Segmenting cells...")
    
    seg_params = SegmentationParams(
        mode=SegmentationMode.STARDIST,
        image_modality=ImageModality.HE_BRIGHTFIELD,
        stardist_model="2D_versatile_he",
        pixel_size=0.5,              # 0.5 microns/pixel
        min_cell_size=20.0,
        max_cell_size=500.0
    )
    
    cell_analyzer = CellAnalyzer(verbose=True)
    seg_result = cell_analyzer.segment_cells_2d(
        registered_he,
        seg_params,
        image_id="patient_001"
    )
    
    print(f"  ✓ Segmented {seg_result.total_cell_count} cells")
    
    # Convert to CellObject instances
    from tme_quant.core.tme_models.cell_model import create_cell_object_from_segmentation
    cells = [
        create_cell_object_from_segmentation(cell_props, parent_id="patient_001", pixel_size=0.5)
        for cell_props in seg_result.cells
    ]
    
    # ============================================================
    # STEP 4: Fiber Extraction
    # ============================================================
    
    print("\n[4/6] Extracting collagen fibers...")
    
    extract_params = ExtractionParams(
        mode=ExtractionMode.CTFIRE,
        pixel_size=0.5,
        min_fiber_length=10.0,
        measure_length=True,
        measure_width=True,
        measure_straightness=True,
        measure_angle=True
    )
    
    fiber_analyzer = FiberAnalyzer(verbose=True)
    fiber_result = fiber_analyzer.extract_fibers_2d(
        shg_image,
        extract_params,
        image_id="patient_001"
    )
    
    print(f"  ✓ Extracted {len(fiber_result.fibers)} fibers")
    
    # Convert to FiberObject instances
    from tme_quant.core.tme_models.fiber_model import create_fiber_object_from_extraction
    fibers = [
        create_fiber_object_from_extraction(fiber_props, parent_id="patient_001", pixel_size=0.5)
        for fiber_props in fiber_result.fibers
    ]
    
    # ============================================================
    # STEP 5: TME Analysis with TACS Classification
    # ============================================================
    
    print("\n[5/6] Running TME analysis...")
    
    tme_analyzer = TMEAnalyzer(verbose=True)
    
    # 5a. Detect tumor regions
    from tme_quant.tme_analysis.config import TumorDetectionParams, TumorDetectionMethod
    
    tumor_params = TumorDetectionParams(
        method=TumorDetectionMethod.CLUSTERING,
        dbscan_eps=100.0,          # 100 micron radius
        dbscan_min_samples=10,
        min_tumor_area=1000.0,
        smooth_boundary=True
    )
    
    tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
    print(f"  ✓ Detected {len(tumor_regions)} tumor regions")
    
    # 5b. TACS analysis
    tacs_params = TMEAnalysisParams(
        mode=AnalysisMode.TUMOR_BASED,
        tumor_boundary_distance=100.0,   # 100 micron TACS zone
        
        # TACS classification thresholds
        tacs_angle_threshold_perpendicular=30.0,
        tacs_angle_threshold_parallel=60.0,
        tacs_straightness_threshold=0.7,
        
        # Enable all measurements
        compute_tacs=True,
        compute_morphology=True,
        compute_spatial=True,
        compute_prognostic=True,
        
        # Zone generation
        generate_zones=True,
        invasive_margin_width=50.0,
        stroma_width=200.0
    )
    
    tacs_result = tme_analyzer.analyze(
        cells=cells,
        fibers=fibers,
        tumor_regions=tumor_regions,
        params=tacs_params,
        analysis_id="tacs_patient_001"
    )
    
    # ============================================================
    # STEP 6: Results and Export
    # ============================================================
    
    print("\n[6/6] Exporting results...")
    
    # Display TACS summary
    print("\n" + "="*60)
    print(tacs_result.get_tacs_summary())
    print("="*60)
    
    # Display prognostic summary
    print("\n" + "="*60)
    print(tacs_result.get_prognostic_summary())
    print("="*60)
    
    # Export to multiple formats
    from tme_quant.tme_analysis.io import export_tme_analysis_results
    
    export_paths = export_tme_analysis_results(
        tacs_result,
        output_dir="output/",
        formats=["csv", "excel", "json", "geojson"],
        prefix="patient_001"
    )
    
    print("\n✓ Results exported:")
    for fmt, path in export_paths.items():
        print(f"  {fmt}: {path}")
    
    # Detailed metrics
    tacs = tacs_result.tacs_features
    prog = tacs_result.prognostic_scores
    
    print("\nDetailed TACS Metrics:")
    print(f"  Total boundary fibers: {tacs['total_boundary_fibers']}")
    print(f"  TACS-1 (random): {tacs['tacs1_ratio']:.1%} ({tacs['tacs1_count']} fibers)")
    print(f"  TACS-2 (parallel): {tacs['tacs2_ratio']:.1%} ({tacs['tacs2_count']} fibers)")
    print(f"  TACS-3 (perpendicular): {tacs['tacs3_ratio']:.1%} ({tacs['tacs3_count']} fibers)")
    print(f"  Dominant TACS: {tacs['dominant_tacs_type']}")
    print(f"  TACS heterogeneity: {tacs['tacs_heterogeneity']:.3f}")
    
    print("\nClinical Risk Assessment:")
    print(f"  Collagen Prognostic Score: {prog['collagen_prognostic_score']:.3f}")
    print(f"  TACS-3 Prognostic: {prog['tacs3_prognostic']:.3f}")
    print(f"  Invasive Potential: {prog['invasive_potential_score']:.3f}")
    print(f"  Overall TME Risk: {prog['overall_tme_risk_score']:.3f}")
    
    if prog['overall_tme_risk_score'] > 0.7:
        print("\n  ⚠️  HIGH RISK - Aggressive tumor phenotype")
    elif prog['overall_tme_risk_score'] > 0.4:
        print("\n  ⚡ MEDIUM RISK - Intermediate prognosis")
    else:
        print("\n  ✓ LOW RISK - Favorable prognosis")
    
    print("\n" + "="*60)
    print("✓ Complete workflow finished!")
    print("="*60)
    
    return tacs_result


# ============================================================
# EXAMPLE 2: Batch Processing Multiple Patients
# ============================================================

def example_2_batch_processing():
    """
    Batch process multiple patient samples.
    """
    
    from pathlib import Path
    import pandas as pd
    
    print("="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)
    
    # Initialize analyzers
    registration_manager = RegistrationManager(verbose=False)
    cell_analyzer = CellAnalyzer(verbose=False)
    fiber_analyzer = FiberAnalyzer(verbose=False)
    tme_analyzer = TMEAnalyzer(verbose=False)
    
    # Results storage
    all_results = []
    
    # Process each patient
    patient_dirs = Path("data/patients/").glob("patient_*")
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"\nProcessing {patient_id}...")
        
        try:
            # Load images
            he_image = io.imread(patient_dir / "HE.tif")
            shg_image = io.imread(patient_dir / "SHG.tif")
            
            # Register
            reg_result = registration_manager.register(shg_image, he_image, reg_params)
            registered_he = reg_result.registered_image
            
            # Segment cells
            seg_result = cell_analyzer.segment_cells_2d(registered_he, seg_params)
            cells = [create_cell_object_from_segmentation(c) for c in seg_result.cells]
            
            # Extract fibers
            fiber_result = fiber_analyzer.extract_fibers_2d(shg_image, extract_params)
            fibers = [create_fiber_object_from_extraction(f) for f in fiber_result.fibers]
            
            # Detect tumors
            tumor_regions = tme_analyzer.detect_tumor_regions(cells, tumor_params)
            
            # TME analysis
            tacs_result = tme_analyzer.analyze(cells, fibers, tumor_regions, tacs_params)
            
            # Extract key metrics
            metrics = {
                'patient_id': patient_id,
                'n_cells': len(cells),
                'n_fibers': len(fibers),
                'n_tumors': len(tumor_regions),
                'tacs1_ratio': tacs_result.tacs_features['tacs1_ratio'],
                'tacs2_ratio': tacs_result.tacs_features['tacs2_ratio'],
                'tacs3_ratio': tacs_result.tacs_features['tacs3_ratio'],
                'dominant_tacs': tacs_result.tacs_features['dominant_tacs_type'],
                'collagen_prognostic_score': tacs_result.prognostic_scores['collagen_prognostic_score'],
                'tme_risk_score': tacs_result.prognostic_scores['overall_tme_risk_score']
            }
            
            all_results.append(metrics)
            print(f"  ✓ {patient_id} complete")
            
        except Exception as e:
            print(f"  ✗ {patient_id} failed: {e}")
            continue
    
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Save batch results
    df.to_csv("output/batch_tacs_analysis.csv", index=False)
    
    print(f"\n✓ Batch processing complete: {len(all_results)} patients")
    print(f"\nSummary Statistics:")
    print(df[['tacs3_ratio', 'collagen_prognostic_score', 'tme_risk_score']].describe())
    
    # Identify high-risk patients
    high_risk = df[df['tme_risk_score'] > 0.7]
    print(f"\nHigh-risk patients (n={len(high_risk)}):")
    print(high_risk[['patient_id', 'tacs3_ratio', 'tme_risk_score']])
    
    return df


# ============================================================
# EXAMPLE 3: Napari Interactive Visualization
# ============================================================

def example_3_napari_visualization():
    """
    Interactive visualization of TME analysis in Napari.
    """
    
    import napari
    
    print("="*60)
    print("EXAMPLE 3: Napari Visualization")
    print("="*60)
    
    # Assume we have results from Example 1
    # Load images and results
    
    # Create Napari viewer
    viewer = napari.Viewer()
    
    # Add collagen SHG image
    viewer.add_image(
        shg_image,
        name='SHG Collagen',
        colormap='gray'
    )
    
    # Add registered H&E
    viewer.add_image(
        registered_he,
        name='H&E (Registered)',
        colormap='blue',
        opacity=0.5
    )
    
    # Add tumor boundaries
    for i, tumor in enumerate(tumor_regions):
        boundary_coords = np.array(tumor.roi.polygon.boundary.coords)
        
        viewer.add_shapes(
            [boundary_coords],
            shape_type='polygon',
            edge_color='yellow',
            edge_width=3,
            face_color='transparent',
            name=f'Tumor {i+1}'
        )
    
    # Add TACS-classified fibers with color coding
    tacs_colors = {
        'TACS-1': 'blue',     # Random, low risk
        'TACS-2': 'green',    # Parallel, medium risk
        'TACS-3': 'red',      # Perpendicular, HIGH RISK (invasive)
    }
    
    for tacs_type, color in tacs_colors.items():
        fiber_lines = []
        
        for pair in tacs_result.interaction_pairs:
            if pair.interaction_type == tacs_type:
                # Find corresponding fiber
                fiber = next(f for f in fibers if f.object_id == pair.source_id)
                fiber_lines.append(fiber.centerline)
        
        if fiber_lines:
            viewer.add_shapes(
                fiber_lines,
                shape_type='path',
                edge_color=color,
                edge_width=2,
                name=f'{tacs_type} Fibers'
            )
    
    # Add cell centroids
    cell_points = np.array([c.centroid for c in cells])
    viewer.add_points(
        cell_points,
        size=5,
        face_color='yellow',
        edge_color='white',
        name='Cells'
    )
    
    print("✓ Napari viewer ready")
    print("\nLegend:")
    print("  🔵 TACS-1 (Blue) - Random, curly fibers")
    print("  🟢 TACS-2 (Green) - Parallel to boundary")
    print("  🔴 TACS-3 (Red) - Perpendicular, invasive")
    print("\nClose the Napari window to continue...")
    
    napari.run()


# ============================================================
# EXAMPLE 4: Project-Based Workflow
# ============================================================

def example_4_project_workflow():
    """
    Use TMEProject for organized multi-image analysis.
    """
    
    from tme_quant.core.project import TMEProject
    
    print("="*60)
    print("EXAMPLE 4: Project-Based Workflow")
    print("="*60)
    
    # Create project
    project = TMEProject(name="TACS_Study_2024")
    
    # Add image
    project.add_image(
        image_id="patient_001",
        image_path="data/patient_001_HE.tif",
        channels={'nuclei': 0, 'collagen': 1},
        pixel_size=(0.5, 0.5)
    )
    
    # Run complete analysis and add to project
    # ... (run cell, fiber, TME analysis)
    
    # Add objects to project
    for cell in cells:
        project.add_object(cell)
    
    for fiber in fibers:
        project.add_object(fiber)
    
    for tumor in tumor_regions:
        project.add_object(tumor)
    
    # Save project
    project.save("output/tacs_study.tme")
    
    print("✓ Project saved")
    
    # Later: load project
    loaded_project = TMEProject.load("output/tacs_study.tme")
    
    # Access objects
    cells = loaded_project.get_objects_by_type("cell")
    fibers = loaded_project.get_objects_by_type("fiber")
    
    print(f"✓ Loaded project with {len(cells)} cells and {len(fibers)} fibers")
    
    return project


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TMEQuant Platform - Usage Examples")
    print("="*60)
    
    # Run Example 1: Complete workflow
    print("\n\nRunning Example 1...")
    result = example_1_complete_he_shg_workflow()
    
    print("\n\n✓ All examples available!")
    print("\nTo run other examples:")
    print("  - example_2_batch_processing()")
    print("  - example_3_napari_visualization()")
    print("  - example_4_project_workflow()")