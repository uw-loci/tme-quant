"""
Bridge classes to connect different subsystems.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
from dataclasses import dataclass, field

# Import existing modules
try:
    from bioimage_core.image_processing import (
        ImageProcessor,
        Segmenter,
        FeatureExtractor
    )
    from bioimage_core.visualization import NapariVisualizer
    EXISTING_IMPORTS_AVAILABLE = True
except ImportError:
    EXISTING_IMPORTS_AVAILABLE = False
    # Mock classes
    class ImageProcessor: pass
    class Segmenter: pass
    class FeatureExtractor: pass
    class NapariVisualizer: pass

# Import hierarchy
from qupath_hierarchy import (
    QuPathLikeProject,
    PathObject,
    ImageObject,
    MeasurementCalculator,
    NapariBridge as HierarchyNapariBridge
)

# Import adapters
from .adapters import (
    ImageProcessingAdapter,
    SegmentationAdapter,
    MeasurementAdapter,
    ProjectAdapter
)


@dataclass
class BioImageBridge:
    """
    Bridge between bioimage processing and hierarchy model.
    """
    
    image_processor: Optional[ImageProcessor] = None
    segmenter: Optional[Segmenter] = None
    feature_extractor: Optional[FeatureExtractor] = None
    measurement_calculator: MeasurementCalculator = field(default_factory=MeasurementCalculator)
    
    # Adapters
    image_adapter: ImageProcessingAdapter = field(default_factory=ImageProcessingAdapter)
    segmentation_adapter: SegmentationAdapter = field(default_factory=SegmentationAdapter)
    measurement_adapter: MeasurementAdapter = field(default_factory=MeasurementAdapter)
    
    def __post_init__(self):
        """Initialize existing modules if available."""
        if EXISTING_IMPORTS_AVAILABLE:
            if self.image_processor is None:
                self.image_processor = ImageProcessor()
            if self.segmenter is None:
                self.segmenter = Segmenter()
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor()
    
    def process_image_with_hierarchy(
        self,
        image_data: np.ndarray,
        image_metadata: Dict[str, Any],
        analysis_pipeline: str = "default"
    ) -> QuPathLikeProject:
        """
        Complete image analysis pipeline with hierarchy integration.
        
        Args:
            image_data: Image data array
            image_metadata: Image metadata
            analysis_pipeline: Name of pipeline to use
            
        Returns:
            QuPathLikeProject with analysis results
        """
        from qupath_hierarchy import QuPathLikeProject, ImageEntry
        
        # Create project
        project = QuPathLikeProject(
            name=f"Analysis_{analysis_pipeline}"
        )
        
        # Add image
        image_entry = ImageEntry(
            name="Analyzed_Image",
            metadata=image_metadata
        )
        image_obj = project.add_image(image_entry)
        
        # Step 1: Image preprocessing using existing tools
        if self.image_processor:
            processed = self.image_processor.preprocess(
                image_data, pipeline=analysis_pipeline
            )
            
            # Store preprocessing results
            image_obj.metadata['preprocessing'] = {
                'pipeline': analysis_pipeline,
                'processor': str(type(self.image_processor))
            }
        else:
            processed = image_data
        
        # Step 2: Apply analysis pipeline
        if analysis_pipeline == "cell_analysis":
            project = self._cell_analysis_pipeline(processed, image_obj, project)
        elif analysis_pipeline == "tissue_analysis":
            project = self._tissue_analysis_pipeline(processed, image_obj, project)
        elif analysis_pipeline == "fiber_analysis":
            project = self._fiber_analysis_pipeline(processed, image_obj, project)
        else:
            project = self._default_analysis_pipeline(processed, image_obj, project)
        
        return project
    
    def _cell_analysis_pipeline(
        self,
        image_data: np.ndarray,
        image_obj: ImageObject,
        project: QuPathLikeProject
    ) -> QuPathLikeProject:
        """Cell analysis pipeline."""
        from qupath_hierarchy import TissueRegion, Cell, ObjectType
        
        # Create tissue region
        tissue = TissueRegion(name="Tissue")
        project.add_object(tissue, parent=image_obj)
        
        # Segment cells using existing segmenter
        if self.segmenter:
            cell_masks = self.segmenter.segment_cells(image_data)
            
            # Convert to hierarchy objects
            cells = self.segmentation_adapter.segmentation_to_objects(
                cell_masks, tissue, ObjectType.CELL, "Cell"
            )
            
            # Add to project
            for cell in cells:
                project.add_object(cell, parent=tissue)
                
                # Compute features using both systems
                self._compute_cross_system_features(cell, image_data)
        
        return project
    
    def _tissue_analysis_pipeline(
        self,
        image_data: np.ndarray,
        image_obj: ImageObject,
        project: QuPathLikeProject
    ) -> QuPathLikeProject:
        """Tissue analysis pipeline."""
        from qupath_hierarchy import TissueRegion, TumorRegion, ObjectType
        
        # Segment tissue regions
        if self.segmenter:
            tissue_masks = self.segmenter.segment_tissue(image_data)
            
            # Create tissue regions
            tissues = self.segmentation_adapter.segmentation_to_objects(
                tissue_masks, image_obj, ObjectType.TISSUE_REGION, "Tissue"
            )
            
            for tissue in tissues:
                project.add_object(tissue, parent=image_obj)
                
                # Detect tumor within tissue
                tumor_masks = self.segmenter.detect_tumor(
                    image_data, tissue.rois[0] if tissue.rois else None
                )
                
                tumors = self.segmentation_adapter.segmentation_to_objects(
                    tumor_masks, tissue, ObjectType.TUMOR_REGION, "Tumor"
                )
                
                for tumor in tumors:
                    project.add_object(tumor, parent=tissue)
        
        return project
    
    def _compute_cross_system_features(
        self,
        obj: PathObject,
        image_data: np.ndarray
    ) -> None:
        """Compute features using both existing and hierarchy systems."""
        # Existing feature extraction
        if self.feature_extractor and obj.rois:
            existing_features = self.feature_extractor.extract(
                image_data, obj.rois[0].coordinates
            )
            
            # Convert to hierarchy measurements
            self.measurement_adapter.existing_to_hierarchy_measurements(
                existing_features, obj, "existing"
            )
        
        # Hierarchy measurement calculation
        if obj.rois:
            hierarchy_features = self.measurement_calculator.compute_all(
                obj.rois[0], image_data
            )
            
            for name, value in hierarchy_features.items():
                obj.add_measurement(f"hierarchy_{name}", value)
    
    def update_hierarchy_from_existing_analysis(
        self,
        project: QuPathLikeProject,
        existing_results: Dict[str, Any]
    ) -> QuPathLikeProject:
        """
        Update hierarchy project with new analysis results.
        
        Args:
            project: Existing hierarchy project
            existing_results: New analysis results
            
        Returns:
            Updated project
        """
        # Implementation depends on existing results format
        # This is a template method
        for img_id, img_results in existing_results.items():
            if img_id in project.images:
                self._update_image_from_results(
                    project.images[img_id],
                    img_results
                )
        
        return project
    
    def _update_image_from_results(
        self,
        image_obj: ImageObject,
        results: Dict[str, Any]
    ) -> None:
        """Update image object with new results."""
        # Store results in metadata
        image_obj.metadata['last_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'results_summary': str(results.keys())
        }


@dataclass
class VisualizationBridge:
    """
    Bridge between hierarchy model and visualization systems.
    """
    
    napari_visualizer: Optional[Any] = None
    hierarchy_napari_bridge: Optional[HierarchyNapariBridge] = None
    
    def __post_init__(self):
        """Initialize visualization components."""
        if EXISTING_IMPORTS_AVAILABLE and self.napari_visualizer is None:
            self.napari_visualizer = NapariVisualizer()
        
        if self.hierarchy_napari_bridge is None:
            from qupath_hierarchy import NapariBridge
            self.hierarchy_napari_bridge = NapariBridge()
    
    def display_project_in_napari(
        self,
        project: QuPathLikeProject,
        viewer: Optional[Any] = None,
        active_image_id: Optional[str] = None
    ) -> None:
        """
        Display hierarchy project in Napari.
        
        Args:
            project: Hierarchy project to display
            viewer: Existing Napari viewer (optional)
            active_image_id: ID of image to display initially
        """
        if viewer is None:
            import napari
            viewer = napari.Viewer()
        
        # Use hierarchy's Napari bridge
        if self.hierarchy_napari_bridge:
            self.hierarchy_napari_bridge.display_project(
                project, viewer, active_image_id
            )
        
        # Add existing visualization overlays if available
        if self.napari_visualizer:
            self._add_existing_visualizations(project, viewer)
    
    def _add_existing_visualizations(
        self,
        project: QuPathLikeProject,
        viewer: Any
    ) -> None:
        """Add existing visualization overlays."""
        # This would integrate existing visualization tools
        # For example, adding heatmaps, graphs, etc.
        pass
    
    def export_visualizations(
        self,
        project: QuPathLikeProject,
        output_dir: Union[str, Path],
        formats: List[str] = ['png', 'html']
    ) -> Dict[str, List[Path]]:
        """
        Export project visualizations.
        
        Args:
            project: Hierarchy project
            output_dir: Output directory
            formats: List of formats to export
            
        Returns:
            Dictionary of exported files by format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        exported = {fmt: [] for fmt in formats}
        
        # Export using hierarchy's visualization tools
        from qupath_hierarchy import MatplotlibVisualizer
        
        viz = MatplotlibVisualizer()
        
        # Create summary plots
        if 'png' in formats:
            # Object type distribution
            fig = viz.plot_object_type_distribution(project)
            png_path = output_dir / "object_types.png"
            fig.savefig(png_path, dpi=150)
            exported['png'].append(png_path)
            
            # Measurement distributions
            fig = viz.plot_measurement_distributions(project)
            png_path = output_dir / "measurements.png"
            fig.savefig(png_path, dpi=150)
            exported['png'].append(png_path)
        
        # Export interactive HTML
        if 'html' in formats:
            try:
                from plotly import graph_objects as go
                
                # Create interactive plot
                fig = go.Figure()
                # Add traces based on project data
                
                html_path = output_dir / "interactive_plot.html"
                fig.write_html(str(html_path))
                exported['html'].append(html_path)
            except ImportError:
                pass
        
        return exported