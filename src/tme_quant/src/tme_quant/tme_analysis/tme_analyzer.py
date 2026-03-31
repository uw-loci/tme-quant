"""
Main TME analyzer class coordinating all analysis operations.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union
import time
from pathlib import Path

from .config import (
    TMEAnalysisParams,
    TumorDetectionParams,
    AnalysisMode,
    TMEAnalysisResult
)
from .interaction_detector import InteractionDetector
from .region_manager import RegionManager
from .measurement_engine import MeasurementEngine

from ..core.tme_objects.cell_objects import CellObject
from ..core.tme_objects.fiber_objects import FiberObject
from ..core.tme_objects.tumor_objects import TumorRegion


class TMEAnalyzer:
    """
    Main class for TME analysis.
    
    Supports 4 analysis modes:
        1. Cell-based: Analyze fibers/cells around each cell
        2. Tumor-based: Analyze fibers/cells near tumor boundary (TACS)
        3. Fiber-based: Analyze cells/fibers around each fiber
        4. ROI-based: Analyze all components in custom regions
    
    Example:
        >>> analyzer = TMEAnalyzer()
        >>> 
        >>> # Tumor-based TACS analysis
        >>> params = TMEAnalysisParams(
        ...     mode=AnalysisMode.TUMOR_BASED,
        ...     tumor_boundary_distance=100.0,
        ...     compute_tacs=True
        ... )
        >>> 
        >>> result = analyzer.analyze(
        ...     cells=cell_objects,
        ...     fibers=fiber_objects,
        ...     tumor_regions=tumor_regions,
        ...     params=params
        ... )
        >>> 
        >>> print(f"TACS-3 score: {result.tacs_features['tacs3_score']:.3f}")
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize TME analyzer.
        
        Args:
            verbose: Print progress messages
        """
        self.interaction_detector = InteractionDetector(verbose=verbose)
        self.region_manager = RegionManager(verbose=verbose)
        self.measurement_engine = MeasurementEngine(verbose=verbose)
        
        self.verbose = verbose
        self.results: Optional[TMEAnalysisResult] = None
    
    # ============================================================
    # MAIN ANALYSIS METHODS
    # ============================================================
    
    def analyze(
        self,
        cells: Optional[List[CellObject]] = None,
        fibers: Optional[List[FiberObject]] = None,
        tumor_regions: Optional[List[TumorRegion]] = None,
        custom_rois: Optional[List[Any]] = None,
        params: Optional[TMEAnalysisParams] = None,
        analysis_id: str = "tme_analysis_001"
    ) -> TMEAnalysisResult:
        """
        Run TME analysis based on specified mode.
        
        Args:
            cells: List of CellObject instances
            fibers: List of FiberObject instances
            tumor_regions: List of TumorRegion instances
            custom_rois: List of custom ROI objects
            params: Analysis parameters
            analysis_id: Unique identifier for this analysis
            
        Returns:
            TMEAnalysisResult with all measurements
        """
        if params is None:
            params = TMEAnalysisParams()

        cells = cells or []
        fibers = fibers or []
        tumor_regions = tumor_regions or []
        custom_rois = custom_rois or []

        if params.mode == AnalysisMode.CELL_BASED and not cells:
            raise ValueError("Cell-based analysis requires non-empty 'cells'")

        if params.mode == AnalysisMode.TUMOR_BASED:
            if not fibers:
                raise ValueError("Tumor-based analysis requires non-empty 'fibers'")
            if not tumor_regions:
                raise ValueError("Tumor-based analysis requires non-empty 'tumor_regions'")

        if params.mode == AnalysisMode.FIBER_BASED and not fibers:
            raise ValueError("Fiber-based analysis requires non-empty 'fibers'")

        if params.mode == AnalysisMode.ROI_BASED and not custom_rois:
            raise ValueError("ROI-based analysis requires non-empty 'custom_rois'")
        
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting TME analysis (mode: {params.mode.value})")
        
        # Route to appropriate analysis method
        if params.mode == AnalysisMode.CELL_BASED:
            result = self._analyze_cell_based(cells, fibers, params, analysis_id)
        
        elif params.mode == AnalysisMode.TUMOR_BASED:
            result = self._analyze_tumor_based(
                cells, fibers, tumor_regions, params, analysis_id
            )
        
        elif params.mode == AnalysisMode.FIBER_BASED:
            result = self._analyze_fiber_based(cells, fibers, params, analysis_id)
        
        elif params.mode == AnalysisMode.ROI_BASED:
            result = self._analyze_roi_based(
                cells, fibers, custom_rois, params, analysis_id
            )
        
        else:
            raise ValueError(f"Unknown analysis mode: {params.mode}")
        
        # Add processing metadata
        result.processing_time = time.time() - start_time
        result.parameters = params.to_dict()
        
        # Store result
        self.results = result
        
        if self.verbose:
            print(f"Analysis complete in {result.processing_time:.2f}s")
            print(f"Found {len(result.interaction_pairs)} interactions")
        
        return result
    
    # ============================================================
    # MODE-SPECIFIC ANALYSIS
    # ============================================================
    
    def _analyze_cell_based(
        self,
        cells: List[CellObject],
        fibers: Optional[List[FiberObject]],
        params: TMEAnalysisParams,
        analysis_id: str
    ) -> TMEAnalysisResult:
        """
        Cell-based analysis: For each cell, find interacting fibers/cells.
        
        Analysis workflow:
            1. For each cell:
                - Find fibers within cell_fiber_distance
                - Find other cells within cell_cell_distance
            2. Compute measurements for each interaction
            3. Aggregate statistics
        """
        if self.verbose:
            print(f"Cell-based analysis: {len(cells)} cells")
        
        result = TMEAnalysisResult(
            analysis_id=analysis_id,
            mode=AnalysisMode.CELL_BASED
        )
        
        # Find cell-fiber interactions
        if fibers:
            cell_fiber_pairs = self.interaction_detector.detect_cell_fiber_interactions(
                cells=cells,
                fibers=fibers,
                max_distance=params.cell_fiber_distance,
                strategy=params.interaction_strategy,
                k=params.k_neighbors
            )
            result.interaction_pairs.extend(cell_fiber_pairs)
        
        # Find cell-cell interactions
        cell_cell_pairs = self.interaction_detector.detect_cell_cell_interactions(
            cells=cells,
            max_distance=params.cell_cell_distance,
            strategy=params.interaction_strategy
        )
        result.interaction_pairs.extend(cell_cell_pairs)
        
        # Compute measurements
        if params.compute_morphology:
            result.morphological_features = (
                self.measurement_engine.compute_morphological_features(
                    cells=cells,
                    fibers=fibers,
                    interaction_pairs=result.interaction_pairs
                )
            )
        
        if params.compute_spatial:
            result.spatial_features = (
                self.measurement_engine.compute_spatial_features(
                    cells=cells,
                    fibers=fibers,
                    interaction_pairs=result.interaction_pairs
                )
            )
        
        if params.compute_density:
            result.density_features = (
                self.measurement_engine.compute_density_features(
                    cells=cells,
                    fibers=fibers
                )
            )
        
        # Summary statistics
        result.summary = self._compute_summary(result, params)
        
        return result
    
    def _analyze_tumor_based(
        self,
        cells: Optional[List[CellObject]],
        fibers: List[FiberObject],
        tumor_regions: List[TumorRegion],
        params: TMEAnalysisParams,
        analysis_id: str
    ) -> TMEAnalysisResult:
        """
        Tumor-based analysis: TACS and tumor boundary interactions.
        
        Analysis workflow:
            1. For each tumor region:
                - Identify boundary zone (within tumor_boundary_distance)
                - Find fibers in boundary zone
                - Find cells in boundary zone
            2. Compute TACS features for boundary fibers
            3. Analyze cell distribution near boundary
            4. Compute prognostic features
        """
        if self.verbose:
            print(f"Tumor-based analysis: {len(tumor_regions)} tumor regions")
        
        result = TMEAnalysisResult(
            analysis_id=analysis_id,
            mode=AnalysisMode.TUMOR_BASED,
            tumor_regions=[
                str(getattr(t, 'object_id', getattr(t, 'id', 'tumor')))
                for t in tumor_regions
            ]
        )
        
        # Generate zones if requested
        if params.generate_zones:
            result.zones = self.region_manager.generate_tumor_zones(
                tumor_regions=tumor_regions,
                invasive_margin_width=params.invasive_margin_width,
                stroma_width=params.stroma_width
            )
        
        # Find fiber-tumor boundary interactions
        fiber_tumor_pairs = self.interaction_detector.detect_fiber_tumor_interactions(
            fibers=fibers,
            tumor_regions=tumor_regions,
            boundary_distance=params.tumor_boundary_distance
        )
        result.interaction_pairs.extend(fiber_tumor_pairs)
        
        # Find cell-tumor boundary interactions
        if cells:
            cell_tumor_pairs = self.interaction_detector.detect_cell_tumor_interactions(
                cells=cells,
                tumor_regions=tumor_regions,
                boundary_distance=params.tumor_boundary_distance
            )
            result.interaction_pairs.extend(cell_tumor_pairs)
        
        # Compute TACS features (primary measurement for tumor-based)
        if params.compute_tacs:
            result.tacs_features = (
                self.measurement_engine.compute_tacs_features(
                    interaction_pairs=result.interaction_pairs,
                    angle_threshold_perp=params.tacs_angle_threshold_perpendicular,
                    angle_threshold_para=params.tacs_angle_threshold_parallel,
                    straightness_threshold=params.tacs_straightness_threshold
                )
            )
        
        # Compute prognostic features
        if params.compute_prognostic:
            result.prognostic_scores = (
                self.measurement_engine.compute_prognostic_scores(
                    tacs_features=result.tacs_features,
                    spatial_features=result.spatial_features,
                    interaction_pairs=result.interaction_pairs
                )
            )
        
        # Summary
        result.summary = self._compute_summary(result, params)
        
        return result
    
    def _analyze_fiber_based(
        self,
        cells: Optional[List[CellObject]],
        fibers: List[FiberObject],
        params: TMEAnalysisParams,
        analysis_id: str
    ) -> TMEAnalysisResult:
        """
        Fiber-based analysis: For each fiber, find interacting cells/fibers.
        
        Analysis workflow:
            1. For each fiber:
                - Find nearest cell(s)
                - Find parallel/perpendicular fibers within fiber_fiber_distance
            2. Compute fiber alignment metrics
            3. Analyze fiber-cell guidance
        """
        if self.verbose:
            print(f"Fiber-based analysis: {len(fibers)} fibers")
        
        result = TMEAnalysisResult(
            analysis_id=analysis_id,
            mode=AnalysisMode.FIBER_BASED
        )
        
        # Find fiber-cell interactions
        if cells:
            fiber_cell_pairs = self.interaction_detector.detect_fiber_cell_interactions(
                fibers=fibers,
                cells=cells,
                max_distance=params.cell_fiber_distance,
                strategy=params.interaction_strategy
            )
            result.interaction_pairs.extend(fiber_cell_pairs)
        
        # Find fiber-fiber interactions
        fiber_fiber_pairs = self.interaction_detector.detect_fiber_fiber_interactions(
            fibers=fibers,
            max_distance=params.fiber_fiber_distance,
            strategy=params.interaction_strategy
        )
        result.interaction_pairs.extend(fiber_fiber_pairs)
        
        # Compute orientation features (fiber alignment)
        if params.compute_orientation:
            result.orientation_features = (
                self.measurement_engine.compute_orientation_features(
                    interaction_pairs=result.interaction_pairs
                )
            )
        
        # Summary
        result.summary = self._compute_summary(result, params)
        
        return result
    
    def _analyze_roi_based(
        self,
        cells: Optional[List[CellObject]],
        fibers: Optional[List[FiberObject]],
        custom_rois: List[Any],
        params: TMEAnalysisParams,
        analysis_id: str
    ) -> TMEAnalysisResult:
        """
        ROI-based analysis: Analyze all components within custom regions.
        """
        if self.verbose:
            print(f"ROI-based analysis: {len(custom_rois)} ROIs")
        
        result = TMEAnalysisResult(
            analysis_id=analysis_id,
            mode=AnalysisMode.ROI_BASED
        )
        
        # For each ROI, find components inside
        for roi in custom_rois:
            # Filter components by ROI
            cells_in_roi = self.region_manager.filter_cells_by_roi(cells, roi)
            fibers_in_roi = self.region_manager.filter_fibers_by_roi(fibers, roi)
            
            # Analyze interactions within this ROI
            # (Similar to cell-based or fiber-based depending on focus)
            
            if cells_in_roi and fibers_in_roi:
                pairs = self.interaction_detector.detect_cell_fiber_interactions(
                    cells=cells_in_roi,
                    fibers=fibers_in_roi,
                    max_distance=params.interaction_distance,
                    strategy=params.interaction_strategy
                )
                result.interaction_pairs.extend(pairs)
        
        # Compute measurements
        # ... (similar to other modes)
        
        result.summary = self._compute_summary(result, params)
        
        return result
    
    # ============================================================
    # TUMOR REGION GENERATION
    # ============================================================
    
    def detect_tumor_regions(
        self,
        cells: List[CellObject],
        params: Optional[TumorDetectionParams] = None
    ) -> List[TumorRegion]:
        """
        Automatically detect tumor regions from cell data.
        
        Args:
            cells: List of CellObject instances
            params: Tumor detection parameters
            
        Returns:
            List of detected TumorRegion objects
        """
        if params is None:
            params = TumorDetectionParams()
        
        if self.verbose:
            print(f"Detecting tumor regions using {params.method.value}")
        
        tumor_regions = self.region_manager.detect_tumor_regions(cells, params)
        
        if self.verbose:
            print(f"Detected {len(tumor_regions)} tumor regions")
        
        return tumor_regions
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _compute_summary(
        self,
        result: TMEAnalysisResult,
        params: TMEAnalysisParams
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {
            'n_interactions': len(result.interaction_pairs),
            'analysis_mode': params.mode.value,
        }
        
        # Add mode-specific summaries
        if params.mode == AnalysisMode.TUMOR_BASED and result.tacs_features:
            summary['dominant_tacs'] = result.tacs_features.get('dominant_tacs_type')
            summary['tacs3_ratio'] = result.tacs_features.get('tacs3_ratio', 0.0)
        
        return summary
    
    def get_results(self) -> Optional[TMEAnalysisResult]:
        """Get the current analysis results."""
        return self.results
    
    def export_results(
        self,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"]
    ) -> Dict[str, str]:
        """
        Export analysis results.
        
        Args:
            output_dir: Output directory
            formats: Export formats
            
        Returns:
            Dictionary mapping format to file path
        """
        from .io import TMEAnalysisExporter

        if self.results is None:
            raise ValueError("No analysis results available. Run analyze() before export.")
        
        exporter = TMEAnalysisExporter()
        return exporter.export(
            self.results,
            output_dir=output_dir,
            formats=formats
        )