# tme_quant/cell_analysis/core/cell_analyzer.py

"""
Main cell analyzer class coordinating segmentation, classification, and quantification.

This is the primary entry point for all cell analysis operations.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
from pathlib import Path
import time

from .segmentation_analyzer import CellSegmentationAnalyzer
from .classification_analyzer import CellClassificationAnalyzer
from .quantification_analyzer import CellQuantificationAnalyzer
from .results import CellAnalysisResult

from ..config.segmentation_params import SegmentationParams
from ..config.classification_params import ClassificationParams
from ..config.quantification_params import QuantificationParams
from ..io.exporters import CellAnalysisExporter


class CellAnalyzer:
    """
    Main class for cell analysis coordinating segmentation, classification, and quantification.
    
    Example:
        >>> analyzer = CellAnalyzer()
        >>> 
        >>> # Segmentation
        >>> seg_params = SegmentationParams(
        ...     mode=SegmentationMode.STARDIST,
        ...     image_modality=ImageModality.FLUORESCENCE
        ... )
        >>> seg_result = analyzer.segment_cells_2d(image, seg_params)
        >>> 
        >>> # Classification
        >>> class_params = ClassificationParams(
        ...     mode=ClassificationMode.MARKER,
        ...     cell_types=[CellType.TUMOR, CellType.IMMUNE]
        ... )
        >>> class_result = analyzer.classify_cells(seg_result, class_params)
        >>> 
        >>> # Quantification
        >>> quant_params = QuantificationParams(measure_area=True)
        >>> quant_result = analyzer.quantify_cells(seg_result, quant_params)
        >>> 
        >>> # Export
        >>> analyzer.export_results("output/", formats=["csv", "excel"])
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize cell analyzer.
        
        Args:
            verbose: Print progress messages
        """
        self.segmentation_analyzer = CellSegmentationAnalyzer(verbose=verbose)
        self.classification_analyzer = CellClassificationAnalyzer(verbose=verbose)
        self.quantification_analyzer = CellQuantificationAnalyzer(verbose=verbose)
        self.exporter = CellAnalysisExporter()
        
        self.results: Optional[CellAnalysisResult] = None
        self.verbose = verbose
    
    # ============================================================
    # SEGMENTATION METHODS
    # ============================================================
    
    def segment_cells_2d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
        image_id: str = "image_001"
    ) -> 'SegmentationResult':
        """
        Segment cells in 2D image.
        
        Args:
            image: 2D image (grayscale or multi-channel)
            params: Segmentation parameters
            image_id: Image identifier
            
        Returns:
            SegmentationResult with cell properties and masks
        """
        if self.verbose:
            print(f"Segmenting cells in {image_id} using {params.mode.value}")
        
        result = self.segmentation_analyzer.segment_2d(image, params)
        
        # Store in combined results
        if self.results is None:
            self.results = CellAnalysisResult(image_id=image_id)
        self.results.segmentation_result = result
        
        if self.verbose:
            print(f"Segmented {result.total_cell_count} cells")
        
        return result
    
    def segment_cells_3d(
        self,
        image: np.ndarray,
        params: SegmentationParams,
        image_id: str = "image_001"
    ) -> 'SegmentationResult':
        """
        Segment cells in 3D image.
        
        Args:
            image: 3D image
            params: Segmentation parameters
            image_id: Image identifier
            
        Returns:
            SegmentationResult with cell properties
        """
        if self.verbose:
            print(f"Segmenting cells in 3D image {image_id}")
        
        result = self.segmentation_analyzer.segment_3d(image, params)
        
        if self.results is None:
            self.results = CellAnalysisResult(image_id=image_id)
        self.results.segmentation_result = result
        
        if self.verbose:
            print(f"Segmented {result.total_cell_count} cells")
        
        return result
    
    # ============================================================
    # CLASSIFICATION METHODS
    # ============================================================
    
    def classify_cells(
        self,
        segmentation_result: 'SegmentationResult',
        params: ClassificationParams,
        image: Optional[np.ndarray] = None
    ) -> 'ClassificationResult':
        """
        Classify segmented cells.
        
        Args:
            segmentation_result: Result from segmentation
            params: Classification parameters
            image: Original image (required for some methods)
            
        Returns:
            ClassificationResult with cell type assignments
        """
        if self.verbose:
            print(f"Classifying {len(segmentation_result.cells)} cells")
        
        result = self.classification_analyzer.classify(
            segmentation_result, params, image
        )
        
        # Store in combined results
        if self.results:
            self.results.classification_result = result
        
        if self.verbose:
            print(f"Classification complete: {result.type_counts}")
        
        return result
    
    # ============================================================
    # QUANTIFICATION METHODS
    # ============================================================
    
    def quantify_cells(
        self,
        segmentation_result: 'SegmentationResult',
        params: QuantificationParams,
        image: Optional[np.ndarray] = None
    ) -> 'QuantificationResult':
        """
        Quantify cell features.
        
        Args:
            segmentation_result: Result from segmentation
            params: Quantification parameters
            image: Original image (for intensity measurements)
            
        Returns:
            QuantificationResult with measurements
        """
        if self.verbose:
            print(f"Quantifying {len(segmentation_result.cells)} cells")
        
        result = self.quantification_analyzer.quantify(
            segmentation_result, params, image
        )
        
        # Store in combined results
        if self.results:
            self.results.quantification_result = result
        
        if self.verbose:
            print(f"Quantification complete")
        
        return result
    
    # ============================================================
    # COMBINED PIPELINE
    # ============================================================
    
    def analyze_full_pipeline_2d(
        self,
        image: np.ndarray,
        segmentation_params: SegmentationParams,
        classification_params: Optional[ClassificationParams] = None,
        quantification_params: Optional[QuantificationParams] = None,
        image_id: str = "image_001"
    ) -> CellAnalysisResult:
        """
        Run full cell analysis pipeline: segment -> classify -> quantify.
        
        Args:
            image: 2D image
            segmentation_params: Segmentation parameters
            classification_params: Classification parameters (optional)
            quantification_params: Quantification parameters (optional)
            image_id: Image identifier
            
        Returns:
            CellAnalysisResult with all results
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Running full cell analysis pipeline on {image_id}")
        
        # Step 1: Segmentation
        seg_result = self.segment_cells_2d(image, segmentation_params, image_id)
        
        # Step 2: Classification (if requested)
        if classification_params:
            class_result = self.classify_cells(seg_result, classification_params, image)
        
        # Step 3: Quantification (if requested)
        if quantification_params:
            quant_result = self.quantify_cells(seg_result, quantification_params, image)
        
        # Compute combined measurements
        self._compute_combined_measurements()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"Full pipeline completed in {total_time:.2f}s")
        
        return self.results
    
    def analyze_full_pipeline_3d(
        self,
        image: np.ndarray,
        segmentation_params: SegmentationParams,
        classification_params: Optional[ClassificationParams] = None,
        quantification_params: Optional[QuantificationParams] = None,
        image_id: str = "image_001"
    ) -> CellAnalysisResult:
        """Run full pipeline on 3D image."""
        start_time = time.time()
        
        # Step 1: Segmentation
        seg_result = self.segment_cells_3d(image, segmentation_params, image_id)
        
        # Step 2: Classification (if requested)
        if classification_params:
            class_result = self.classify_cells(seg_result, classification_params, image)
        
        # Step 3: Quantification (if requested)
        if quantification_params:
            quant_result = self.quantify_cells(seg_result, quantification_params, image)
        
        self._compute_combined_measurements()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"3D pipeline completed in {total_time:.2f}s")
        
        return self.results
    
    def _compute_combined_measurements(self):
        """Compute combined measurements from all analysis results."""
        if self.results is None:
            return
        
        measurements = {}
        
        # Add segmentation measurements
        if self.results.segmentation_result:
            seg = self.results.segmentation_result
            measurements['segmentation'] = {
                'cell_count': seg.total_cell_count,
                'mean_area': seg.mean_cell_area,
                'mean_circularity': seg.mean_circularity,
                'mode': seg.mode.value
            }
        
        # Add classification measurements
        if self.results.classification_result:
            class_res = self.results.classification_result
            measurements['classification'] = {
                'mode': class_res.mode.value,
                'type_counts': {k.value: v for k, v in class_res.type_counts.items()},
                'type_ratios': {k.value: v for k, v in class_res.type_ratios.items()}
            }
        
        # Add quantification measurements
        if self.results.quantification_result:
            quant = self.results.quantification_result
            measurements['quantification'] = quant.population_stats
        
        self.results.measurements = measurements
    
    # ============================================================
    # EXPORT METHODS
    # ============================================================
    
    def export_results(
        self,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"],
        prefix: str = "cell_analysis"
    ) -> Dict[str, str]:
        """
        Export analysis results to various formats.
        
        Args:
            output_dir: Output directory path
            formats: List of formats ("csv", "excel", "json", "geojson")
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping format to file path
        """
        if self.results is None:
            raise ValueError("No results to export. Run analysis first.")
        
        export_paths = self.exporter.export(
            self.results,
            output_dir=output_dir,
            formats=formats,
            prefix=prefix
        )
        
        self.results.export_paths = export_paths
        
        if self.verbose:
            print(f"Results exported to {output_dir}")
            for fmt, path in export_paths.items():
                print(f"  {fmt}: {path}")
        
        return export_paths
    
    def get_results(self) -> Optional[CellAnalysisResult]:
        """Get the current analysis results."""
        return self.results
    
    def clear_results(self):
        """Clear stored results."""
        self.results = None