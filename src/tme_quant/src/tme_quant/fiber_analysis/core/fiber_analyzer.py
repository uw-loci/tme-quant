"""Main fiber analyzer class coordinating orientation and extraction analysis."""

from typing import Optional, Union, List
import numpy as np
from pathlib import Path

from .orientation_analyzer import FiberOrientationAnalyzer
from .extraction_analyzer import FiberExtractionAnalyzer
from .results import FiberAnalysisResult
from ..config.orientation_params import OrientationParams
from ..config.extraction_params import ExtractionParams
from ..io.exporters import FiberAnalysisExporter


class FiberAnalyzer:
    """
    Main class for fiber analysis coordinating orientation and extraction.
    
    Example:
        >>> analyzer = FiberAnalyzer()
        >>> 
        >>> # Run orientation analysis
        >>> orientation_params = OrientationParams(
        ...     mode=OrientationMode.CURVEALIGN,
        ...     window_size=128
        ... )
        >>> orientation_result = analyzer.analyze_orientation_2d(
        ...     image, orientation_params
        ... )
        >>> 
        >>> # Run extraction analysis
        >>> extraction_params = ExtractionParams(
        ...     mode=ExtractionMode.CTFIRE,
        ...     min_fiber_length=10.0
        ... )
        >>> extraction_result = analyzer.extract_fibers_2d(
        ...     image, extraction_params
        ... )
        >>> 
        >>> # Export results
        >>> analyzer.export_results(
        ...     "results/",
        ...     formats=["csv", "json", "geojson"]
        ... )
    """
    
    def __init__(self):
        self.orientation_analyzer = FiberOrientationAnalyzer()
        self.extraction_analyzer = FiberExtractionAnalyzer()
        self.exporter = FiberAnalysisExporter()
        self.results: Optional[FiberAnalysisResult] = None
    
    # Orientation analysis methods
    def analyze_orientation_2d(
        self,
        image: np.ndarray,
        params: OrientationParams,
        image_id: str = "image_001"
    ) -> OrientationResult:
        """
        Analyze fiber orientation in 2D image.
        
        Args:
            image: 2D grayscale image
            params: Orientation analysis parameters
            image_id: Image identifier
            
        Returns:
            OrientationResult with orientation maps and statistics
        """
        result = self.orientation_analyzer.analyze_2d(image, params)
        
        # Store in combined results
        if self.results is None:
            self.results = FiberAnalysisResult(image_id=image_id)
        self.results.orientation_result = result
        
        return result
    
    def analyze_orientation_3d(
        self,
        image: np.ndarray,
        params: OrientationParams,
        image_id: str = "image_001"
    ) -> OrientationResult:
        """
        Analyze fiber orientation in 3D image.
        
        Args:
            image: 3D grayscale image
            params: Orientation analysis parameters
            image_id: Image identifier
            
        Returns:
            OrientationResult with orientation maps and statistics
        """
        result = self.orientation_analyzer.analyze_3d(image, params)
        
        # Store in combined results
        if self.results is None:
            self.results = FiberAnalysisResult(image_id=image_id)
        self.results.orientation_result = result
        
        return result
    
    # Extraction analysis methods
    def extract_fibers_2d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
        image_id: str = "image_001"
    ) -> ExtractionResult:
        """
        Extract individual fibers from 2D image.
        
        Args:
            image: 2D grayscale image
            params: Extraction parameters
            image_id: Image identifier
            
        Returns:
            ExtractionResult with fiber properties
        """
        result = self.extraction_analyzer.extract_2d(image, params)
        
        # Store in combined results
        if self.results is None:
            self.results = FiberAnalysisResult(image_id=image_id)
        self.results.extraction_result = result
        
        return result
    
    def extract_fibers_3d(
        self,
        image: np.ndarray,
        params: ExtractionParams,
        image_id: str = "image_001"
    ) -> ExtractionResult:
        """
        Extract individual fibers from 3D image.
        
        Args:
            image: 3D grayscale image
            params: Extraction parameters
            image_id: Image identifier
            
        Returns:
            ExtractionResult with fiber properties
        """
        result = self.extraction_analyzer.extract_3d(image, params)
        
        # Store in combined results
        if self.results is None:
            self.results = FiberAnalysisResult(image_id=image_id)
        self.results.extraction_result = result
        
        return result
    
    # Combined analysis
    def analyze_full_pipeline_2d(
        self,
        image: np.ndarray,
        orientation_params: OrientationParams,
        extraction_params: ExtractionParams,
        image_id: str = "image_001"
    ) -> FiberAnalysisResult:
        """
        Run full fiber analysis pipeline (orientation + extraction) on 2D image.
        
        Args:
            image: 2D grayscale image
            orientation_params: Orientation analysis parameters
            extraction_params: Extraction parameters
            image_id: Image identifier
            
        Returns:
            FiberAnalysisResult with all results
        """
        # Run orientation analysis
        orientation_result = self.analyze_orientation_2d(
            image, orientation_params, image_id
        )
        
        # Run extraction analysis
        extraction_result = self.extract_fibers_2d(
            image, extraction_params, image_id
        )
        
        # Compute combined measurements
        self._compute_combined_measurements()
        
        return self.results
    
    def analyze_full_pipeline_3d(
        self,
        image: np.ndarray,
        orientation_params: OrientationParams,
        extraction_params: ExtractionParams,
        image_id: str = "image_001"
    ) -> FiberAnalysisResult:
        """
        Run full fiber analysis pipeline (orientation + extraction) on 3D image.
        """
        # Run orientation analysis
        orientation_result = self.analyze_orientation_3d(
            image, orientation_params, image_id
        )
        
        # Run extraction analysis
        extraction_result = self.extract_fibers_3d(
            image, extraction_params, image_id
        )
        
        # Compute combined measurements
        self._compute_combined_measurements()
        
        return self.results
    
    def _compute_combined_measurements(self):
        """Compute combined measurements from orientation and extraction results."""
        if self.results is None:
            return
        
        measurements = {}
        
        # Add orientation measurements
        if self.results.orientation_result:
            orient = self.results.orientation_result
            measurements['orientation'] = {
                'mean_orientation': orient.mean_orientation,
                'alignment_score': orient.alignment_score,
                'mode': orient.mode.value
            }
        
        # Add extraction measurements
        if self.results.extraction_result:
            extract = self.results.extraction_result
            measurements['extraction'] = {
                'fiber_count': extract.total_fiber_count,
                'mean_length': extract.mean_fiber_length,
                'mean_width': extract.mean_fiber_width,
                'mean_straightness': extract.mean_straightness,
                'mode': extract.mode.value
            }
        
        self.results.measurements = measurements
    
    # Export methods
    def export_results(
        self,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"],
        prefix: str = "fiber_analysis"
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
        return export_paths
    
    def get_results(self) -> Optional[FiberAnalysisResult]:
        """Get the current analysis results."""
        return self.results
    
    def clear_results(self):
        """Clear stored results."""
        self.results = None