"""
Cell analysis pipeline updated to use hierarchy model.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

# Import integration layer
from integration.facades import UnifiedAnalysisFacade
from integration.bridges import BioImageBridge

# Import existing bioimage core
from bioimage_core.image_processing.segmentation import segment_cells
from bioimage_core.image_processing.filters import preprocess_image


@dataclass
class CellAnalysisPipeline:
    """
    Cell analysis pipeline with hierarchy integration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.facade = UnifiedAnalysisFacade()
        self.bridge = BioImageBridge()
    
    def analyze_cells(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze cells in image using hierarchy model.
        
        Args:
            image_path: Path to image file
            output_dir: Optional output directory
            
        Returns:
            Analysis results
        """
        from bioimage_core.utils.file_io import load_image
        
        # Load image
        image_data, metadata = load_image(image_path)
        
        # Preprocess using existing tools
        processed = preprocess_image(
            image_data,
            method=self.config.get('preprocessing_method', 'default')
        )
        
        # Create project using facade
        project = self.facade.create_project(
            name=image_path.stem,
            description=f"Cell analysis of {image_path.name}",
            tags=['cell_analysis', 'automated']
        )
        
        # Add image
        from qupath_hierarchy import ImageEntry
        image_entry = ImageEntry(
            file_path=image_path,
            name=image_path.stem,
            metadata=metadata
        )
        image_obj = project.add_image(image_entry)
        
        # Segment cells using existing segmentation
        cell_masks = segment_cells(
            processed,
            method=self.config.get('segmentation_method', 'watershed')
        )
        
        # Convert to hierarchy objects using bridge
        from integration.adapters import SegmentationAdapter
        seg_adapter = SegmentationAdapter()
        
        cells = seg_adapter.segmentation_to_objects(
            cell_masks,
            image_obj,
            object_type=ObjectType.CELL,
            classification="Cell"
        )
        
        # Add cells to project
        for cell in cells:
            project.add_object(cell, parent=image_obj)
            
            # Compute measurements
            self._compute_cell_measurements(cell, processed)
        
        # Compute spatial relationships
        if self.config.get('compute_spatial', True):
            self._compute_spatial_relationships(cells)
        
        # Save results
        results = {
            'project': project,
            'num_cells': len(cells),
            'image_path': str(image_path)
        }
        
        if output_dir:
            self._save_results(project, output_dir, results)
        
        return results
    
    def _compute_cell_measurements(self, cell, image_data):
        """Compute comprehensive cell measurements."""
        from qupath_hierarchy import MeasurementCalculator
        
        calculator = MeasurementCalculator()
        
        if cell.rois:
            # Basic morphology
            morph_features = calculator.compute_morphology(cell.rois[0])
            for name, value in morph_features.items():
                cell.add_measurement(name, value)
            
            # Intensity features for each channel
            for channel_idx in range(image_data.shape[0]):
                intensity_features = calculator.compute_intensity(
                    cell.rois[0], image_data[channel_idx]
                )
                for name, value in intensity_features.items():
                    cell.add_measurement(
                        f"{name}_ch{channel_idx}",
                        value,
                        measurement_type="per_channel",
                        channel=channel_idx
                    )
    
    def _compute_spatial_relationships(self, cells):
        """Compute spatial relationships between cells."""
        from scipy.spatial import KDTree
        import numpy as np
        
        # Extract centroids
        centroids = []
        for cell in cells:
            if cell.rois:
                bbox = cell.rois[0].bounding_box()
                centroid = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                )
                centroids.append(centroid)
                cell.add_measurement('centroid_x', centroid[0])
                cell.add_measurement('centroid_y', centroid[1])
        
        if len(centroids) > 1:
            # Build KD-tree for nearest neighbors
            tree = KDTree(centroids)
            
            for i, cell in enumerate(cells):
                if i < len(centroids):
                    # Find nearest neighbors
                    distances, indices = tree.query(
                        centroids[i], 
                        k=min(5, len(centroids))
                    )
                    
                    # Skip self
                    distances = distances[1:]
                    indices = indices[1:]
                    
                    # Store distances to nearest neighbors
                    for j, (dist, idx) in enumerate(zip(distances, indices)):
                        cell.add_measurement(f'neighbor_{j+1}_distance', dist)
                        cell.add_measurement(
                            f'neighbor_{j+1}_id',
                            cells[idx].id[:8]
                        )
    
    def _save_results(self, project, output_dir, results):
        """Save analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save project
        project.save(output_dir / "analysis_project.qproj")
        
        # Export measurements
        df = project.export_measurements(format='dataframe')
        df.to_csv(output_dir / "measurements.csv")
        
        # Save summary
        import json
        summary = {
            'num_cells': results['num_cells'],
            'image_path': results['image_path'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def batch_process(
        self,
        image_paths: List[Path],
        output_base: Path,
        parallel: bool = False
    ) -> Dict[Path, Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            output_base: Base output directory
            parallel: Whether to process in parallel
            
        Returns:
            Dictionary of results by image path
        """
        results = {}
        
        if parallel:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = {}
                for img_path in image_paths:
                    output_dir = output_base / img_path.stem
                    future = executor.submit(
                        self.analyze_cells,
                        img_path,
                        output_dir
                    )
                    futures[future] = img_path
                
                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        results[img_path] = future.result()
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        results[img_path] = {'error': str(e)}
        else:
            # Sequential processing
            for img_path in image_paths:
                output_dir = output_base / img_path.stem
                try:
                    results[img_path] = self.analyze_cells(
                        img_path, output_dir
                    )
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    results[img_path] = {'error': str(e)}
        
        return results