"""Export fiber analysis results to various formats."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from ..core.results import FiberAnalysisResult, FiberProperties


class FiberAnalysisExporter:
    """Export fiber analysis results to CSV, Excel, JSON, and GeoJSON."""
    
    def export(
        self,
        result: FiberAnalysisResult,
        output_dir: Path,
        formats: List[str],
        prefix: str = "fiber_analysis"
    ) -> Dict[str, str]:
        """
        Export results to multiple formats.
        
        Args:
            result: FiberAnalysisResult to export
            output_dir: Output directory
            formats: List of formats ("csv", "excel", "json", "geojson")
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping format to filepath
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        if "csv" in formats:
            export_paths["csv"] = self._export_csv(result, output_dir, prefix)
        
        if "excel" in formats:
            export_paths["excel"] = self._export_excel(result, output_dir, prefix)
        
        if "json" in formats:
            export_paths["json"] = self._export_json(result, output_dir, prefix)
        
        if "geojson" in formats:
            export_paths["geojson"] = self._export_geojson(result, output_dir, prefix)
        
        return export_paths
    
    def _export_csv(
        self,
        result: FiberAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> str:
        """Export to CSV format."""
        # Export fiber properties
        if result.extraction_result and result.extraction_result.fibers:
            fiber_data = self._fibers_to_dataframe(
                result.extraction_result.fibers
            )
            csv_path = output_dir / f"{prefix}_fibers.csv"
            fiber_data.to_csv(csv_path, index=False)
        
        # Export summary statistics
        if result.measurements:
            summary_df = pd.DataFrame([result.measurements])
            summary_path = output_dir / f"{prefix}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
        
        return str(csv_path)
    
    def _export_excel(
        self,
        result: FiberAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> str:
        """Export to Excel format with multiple sheets."""
        excel_path = output_dir / f"{prefix}_results.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Fiber properties sheet
            if result.extraction_result and result.extraction_result.fibers:
                fiber_data = self._fibers_to_dataframe(
                    result.extraction_result.fibers
                )
                fiber_data.to_excel(writer, sheet_name='Fibers', index=False)
            
            # Summary statistics sheet
            if result.measurements:
                summary_df = pd.DataFrame([result.measurements])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Orientation statistics sheet
            if result.orientation_result:
                orient_stats = {
                    'mean_orientation': result.orientation_result.mean_orientation,
                    'alignment_score': result.orientation_result.alignment_score,
                    'mode': result.orientation_result.mode.value
                }
                orient_df = pd.DataFrame([orient_stats])
                orient_df.to_excel(writer, sheet_name='Orientation', index=False)
        
        return str(excel_path)
    
    def _export_json(
        self,
        result: FiberAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> str:
        """Export to JSON format."""
        json_path = output_dir / f"{prefix}_results.json"
        
        export_data = {
            'image_id': result.image_id,
            'measurements': result.measurements
        }
        
        # Add fiber data
        if result.extraction_result and result.extraction_result.fibers:
            export_data['fibers'] = [
                self._fiber_to_dict(f) for f in result.extraction_result.fibers
            ]
        
        # Add orientation data
        if result.orientation_result:
            export_data['orientation'] = {
                'mean_orientation': float(result.orientation_result.mean_orientation),
                'alignment_score': float(result.orientation_result.alignment_score),
                'mode': result.orientation_result.mode.value
            }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(json_path)
    
    def _export_geojson(
        self,
        result: FiberAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> str:
        """Export to GeoJSON format (for spatial visualization)."""
        geojson_path = output_dir / f"{prefix}_fibers.geojson"
        
        if not (result.extraction_result and result.extraction_result.fibers):
            return None
        
        features = []
        for fiber in result.extraction_result.fibers:
            # Convert centerline to GeoJSON LineString
            coordinates = fiber.centerline.tolist()
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coordinates
                },
                'properties': {
                    'fiber_id': fiber.fiber_id,
                    'length': fiber.length,
                    'width': fiber.width,
                    'straightness': fiber.straightness,
                    'angle': fiber.angle,
                    'curvature': fiber.curvature
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return str(geojson_path)
    
    def _fibers_to_dataframe(self, fibers: List[FiberProperties]) -> pd.DataFrame:
        """Convert list of fibers to pandas DataFrame."""
        data = []
        for fiber in fibers:
            data.append({
                'fiber_id': fiber.fiber_id,
                'length': fiber.length,
                'width': fiber.width,
                'straightness': fiber.straightness,
                'angle': fiber.angle,
                'curvature': fiber.curvature,
                'aspect_ratio': fiber.aspect_ratio
            })
        return pd.DataFrame(data)
    
    def _fiber_to_dict(self, fiber: FiberProperties) -> Dict[str, Any]:
        """Convert FiberProperties to dictionary."""
        return {
            'fiber_id': fiber.fiber_id,
            'length': fiber.length,
            'width': fiber.width,
            'straightness': fiber.straightness,
            'angle': fiber.angle,
            'curvature': fiber.curvature,
            'centerline': fiber.centerline.tolist()
        }