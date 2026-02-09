"""
Example demonstrating comprehensive TME analysis with hierarchy
"""
import numpy as np
from tme_quant.core.base_models import (
    TMEObject, Geometry, TMEMetadata, Measurement, GeometryType
)
from tme_quant.core.tme_models import (
    TissueSample, Tumor, TumorRegion, TumorGrade,
    Cell, ImmuneCell, TumorCell, CellType, CellState, ImmuneCellSubtype,
    Fiber, CollagenFiber, FiberType, CollagenType,
    Vessel, VesselType,
    Stroma, StromaRegion, ECMComponent,
    TissueZone, TissueRegion,
    Interaction, InteractionNetwork, InteractionType
)
from tme_quant.core.hierarchy import TMEHierarchy

def create_sample_tme_hierarchy():
    """Create a comprehensive TME sample with hierarchy"""
    
    # 1. Create metadata
    metadata = TMEMetadata(
        sample_id="TME_001",
        patient_id="PAT_123",
        tissue_type="Breast carcinoma",
        stain_type="H&E + Picrosirius Red",
        magnification=20.0,
        pixel_size=(0.5, 0.5, 1.0),  # microns
        image_dimensions=(2048, 2048, 1, 3)
    )
    
    # 2. Create tumor regions
    tumor_region1 = TumorRegion(
        name="Tumor Core",
        geometry=Geometry(
            type=GeometryType.POLYGON,
            coordinates=np.array([
                [100, 100], [200, 100], [200, 200], [100, 200]
            ])
        ),
        grade=TumorGrade.G2,
        necrosis_percentage=10.0,
        invasion_front=True
    )
    
    tumor_region2 = TumorRegion(
        name="Tumor Periphery",
        geometry=Geometry(
            type=GeometryType.POLYGON,
            coordinates=np.array([
                [200, 100], [300, 100], [300, 200], [200, 200]
            ])
        ),
        grade=TumorGrade.G3,
        invasion_front=False
    )
    
    # 3. Create tumor object
    tumor = Tumor(
        name="Primary Tumor",
        regions=[tumor_region1, tumor_region2],
        dominant_grade=TumorGrade.G3,
        tumor_stroma_ratio=0.6
    )
    
    # 4. Create cells
    tumor_cell1 = TumorCell(
        name="Tumor Cell 001",
        geometry=Geometry(
            type=GeometryType.ELLIPSE,
            coordinates=np.array([150, 150, 10, 8])  # x, y, width, height
        ),
        cell_type=CellType.TUMOR,
        grade=TumorGrade.G2,
        proliferation_marker=0.8,
        stemness_score=0.3
    )
    
    immune_cell1 = ImmuneCell(
        name="T Cell 001",
        geometry=Geometry(
            type=GeometryType.ELLIPSE,
            coordinates=np.array([250, 150, 8, 8])
        ),
        cell_type=CellType.IMMUNE,
        immune_subtype=ImmuneCellSubtype.T_CYTOTOXIC,
        activation_state="activated",
        checkpoint_expression={"PD1": 0.7, "CTLA4": 0.3}
    )
    
    # 5. Create fibers
    collagen_fiber1 = CollagenFiber(
        name="Collagen Fiber 001",
        points=np.array([
            [120, 120, 0],
            [130, 125, 0],
            [140, 130, 0],
            [150, 135, 0]
        ]),
        fiber_type=FiberType.COLLAGEN,
        collagen_type=CollagenType.TYPE_I,
        thickness=2.0,
        crosslinking_density=0.8
    )
    
    collagen_fiber2 = CollagenFiber(
        name="Collagen Fiber 002",
        points=np.array([
            [250, 120, 0],
            [260, 130, 0],
            [270, 140, 0],
            [280, 150, 0]
        ]),
        fiber_type=FiberType.COLLAGEN,
        collagen_type=CollagenType.TYPE_III,
        thickness=1.5
    )
    
    # 6. Create vessel
    vessel1 = Vessel(
        name="Blood Vessel 001",
        geometry=Geometry(
            type=GeometryType.ELLIPSE,
            coordinates=np.array([400, 300, 30, 30])  # Larger ellipse
        ),
        vessel_type=VesselType.BLOOD,
        lumen_area=706.86,  # π * 15²
        wall_thickness=5.0,
        perfusion_status="perfused"
    )
    
    # 7. Create stroma region
    stroma_region1 = StromaRegion(
        name="Peritumoral Stroma",
        geometry=Geometry(
            type=GeometryType.POLYGON,
            coordinates=np.array([
                [300, 300], [400, 300], [400, 400], [300, 400]
            ])
        ),
        ecm_composition={
            ECMComponent.COLLAGEN: 0.6,
            ECMComponent.FIBRONECTIN: 0.2,
            ECMComponent.HYALURONAN: 0.1
        },
        cellularity=0.4,
        fibrosis_score=0.7
    )
    
    stroma = Stroma(
        name="Stromal Compartment",
        regions=[stroma_region1],
        fibroblast_density=0.3,
        stiffness=15.0  # kPa
    )
    
    # 8. Create tissue regions
    tumor_core_region = TissueRegion(
        name="Tumor Core Zone",
        geometry=tumor_region1.geometry,
        zone_type=TissueZone.TUMOR_CORE,
        tissue_type="epithelium"
    )
    
    stroma_zone = TissueRegion(
        name="Stromal Zone",
        geometry=stroma_region1.geometry,
        zone_type=TissueZone.PERITUMORAL_STROMA,
        tissue_type="connective"
    )
    
    # 9. Create tissue sample
    tissue_sample = TissueSample(
        name="Breast Carcinoma Sample 001",
        metadata=metadata,
        tumor=tumor,
        stroma=stroma,
        cells=[tumor_cell1, immune_cell1],
        fibers=[collagen_fiber1, collagen_fiber2],
        vessels=[vessel1],
        annotations=[tumor_core_region, stroma_zone],
        tissue_area=1000000.0,  # 1 mm²
        tumor_stroma_ratio=0.6
    )
    
    # 10. Create interactions
    interaction1 = Interaction(
        cell_id=tumor_cell1.id,
        fiber_id=collagen_fiber1.id,
        interaction_type=InteractionType.FIBER_ALIGNMENT,
        distance=5.0,
        alignment_angle=15.0,
        interaction_strength=0.8
    )
    
    interaction2 = Interaction(
        cell_id=immune_cell1.id,
        fiber_id=collagen_fiber2.id,
        interaction_type=InteractionType.FIBER_PROXIMITY,
        distance=8.0,
        alignment_angle=45.0,
        interaction_strength=0.5
    )
    
    interaction_network = InteractionNetwork(
        interactions=[interaction1, interaction2],
        cells={cell.id: cell for cell in [tumor_cell1, immune_cell1]},
        fibers={fiber.id: fiber for fiber in [collagen_fiber1, collagen_fiber2]}
    )
    
    return tissue_sample, interaction_network

def analyze_tme_hierarchy():
    """Analyze TME hierarchy and extract metrics"""
    
    # Create sample
    tissue_sample, interaction_network = create_sample_tme_hierarchy()
    
    # Create hierarchy manager
    hierarchy = TMEHierarchy(tissue_sample)
    
    # 1. Validate hierarchy
    issues = hierarchy.validate_hierarchy()
    if issues:
        print(f"Hierarchy issues: {issues}")
    else:
        print("Hierarchy is valid")
    
    # 2. Extract objects by type
    cells = hierarchy.get_objects_by_type(TMEType.CELL)
    fibers = hierarchy.get_objects_by_type(TMEType.FIBER)
    regions = hierarchy.get_objects_by_type(TMEType.REGION)
    
    print(f"\nFound {len(cells)} cells, {len(fibers)} fibers, {len(regions)} regions")
    
    # 3. Calculate TME metrics
    tme_metrics = tissue_sample.calculate_tme_metrics()
    print("\nTME Metrics:")
    for key, value in tme_metrics.items():
        print(f"  {key}: {value}")
    
    # 4. Export to QuPath format
    qupath_export = hierarchy.export_to_qupath()
    print(f"\nExported hierarchy with {len(qupath_export.get('children', []))} top-level objects")
    
    # 5. Analyze spatial relationships
    spatial_tree = hierarchy.get_spatial_hierarchy()
    print(f"\nSpatial tree depth: {_calculate_tree_depth(spatial_tree)}")
    
    # 6. Find specific objects
    tumor_cell = hierarchy.find_object("Tumor Cell 001")
    if tumor_cell:
        print(f"\nFound tumor cell: {tumor_cell.name}")
        morphometrics = tumor_cell.calculate_morphometrics()
        print(f"  Area: {morphometrics['total_area']:.2f} µm²")
        print(f"  Circularity: {morphometrics['circularity']:.3f}")
    
    # 7. Analyze interactions
    network_metrics = interaction_network.calculate_network_metrics()
    print(f"\nInteraction Network Metrics:")
    print(f"  Total interactions: {network_metrics['total_interactions']}")
    print(f"  Interaction density: {network_metrics['interaction_density']:.3f}")
    print(f"  Average distance: {network_metrics['average_distance']:.2f} µm")
    
    return tissue_sample, hierarchy, interaction_network

def _calculate_tree_depth(tree_node: dict) -> int:
    """Calculate depth of tree structure"""
    if not tree_node.get('children'):
        return 1
    return 1 + max(_calculate_tree_depth(child) for child in tree_node['children'])

def batch_hierarchy_processing(samples: List[TissueSample]):
    """Process multiple TME samples with hierarchy"""
    
    results = []
    
    for sample in samples:
        hierarchy = TMEHierarchy(sample)
        
        # Calculate comprehensive metrics
        sample_metrics = {
            'sample_id': sample.metadata.sample_id,
            'tissue_area': sample.tissue_area,
            'cell_count': len(sample.cells),
            'fiber_count': len(sample.fibers),
            'vessel_count': len(sample.vessels),
            'tumor_stroma_ratio': sample.tumor_stroma_ratio,
        }
        
        # Add hierarchy complexity metrics
        all_objects = hierarchy.root.get_descendants(include_self=True)
        sample_metrics['total_objects'] = len(all_objects)
        
        # Count by type
        type_counts = {}
        for obj_type in TMEType:
            count = len([obj for obj in all_objects if obj.type == obj_type])
            type_counts[obj_type.value] = count
        
        sample_metrics['type_counts'] = type_counts
        
        results.append(sample_metrics)
    
    return results

if __name__ == "__main__":
    # Run comprehensive analysis
    sample, hierarchy, interactions = analyze_tme_hierarchy()
    
    # Demonstrate hierarchy traversal
    print("\n=== Hierarchy Traversal Example ===")
    
    # Get all tumor cells
    tumor_cells = [obj for obj in hierarchy.root.get_descendants() 
                  if hasattr(obj, 'cell_type') and obj.cell_type == CellType.TUMOR]
    
    print(f"Found {len(tumor_cells)} tumor cells in hierarchy")
    
    # Calculate spatial statistics
    if tumor_cells:
        tumor_cell_areas = [cell.area for cell in tumor_cells]
        print(f"Tumor cell areas - Mean: {np.mean(tumor_cell_areas):.2f} µm², "
              f"Std: {np.std(tumor_cell_areas):.2f} µm²")
    
    # Export hierarchy for visualization
    import json
    with open("tme_hierarchy.json", "w") as f:
        json.dump(hierarchy.export_to_qupath(), f, indent=2)
    
    print("\nHierarchy exported to tme_hierarchy.json")