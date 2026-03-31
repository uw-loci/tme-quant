"""
Export utilities for TME analysis results.
"""

from __future__ import annotations


# ========================================================
# EXPORTERS
# ========================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any
import warnings

from .config import TMEAnalysisResult


class TMEAnalysisExporter:
    """
    Export TME analysis results to multiple formats.
    
    Formats:
        - CSV: Tabular interaction data, TACS metrics, prognostic scores
        - Excel: Multi-sheet workbook (interactions, TACS, prognostic, summary)
        - JSON: Complete results with metadata
        - GeoJSON: Spatial data for visualization
    
    Follows the same export pattern as FiberAnalysisExporter and CellAnalysisExporter.
    """
    
    def __init__(self):
        """Initialize exporter."""
        pass
    
    def export(
        self,
        result: TMEAnalysisResult,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"],
        prefix: str = "tme_analysis"
    ) -> Dict[str, str]:
        """
        Export analysis results to specified formats.
        
        Args:
            result: TMEAnalysisResult to export
            output_dir: Output directory
            formats: List of formats ("csv", "excel", "json", "geojson")
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        for fmt in formats:
            if fmt == "csv":
                path = self._export_csv(result, output_dir, prefix)
                export_paths["csv"] = str(path)
            
            elif fmt == "excel":
                path = self._export_excel(result, output_dir, prefix)
                export_paths["excel"] = str(path)
            
            elif fmt == "json":
                path = self._export_json(result, output_dir, prefix)
                export_paths["json"] = str(path)
            
            elif fmt == "geojson":
                path = self._export_geojson(result, output_dir, prefix)
                export_paths["geojson"] = str(path)
            
            else:
                warnings.warn(f"Unknown format: {fmt}")
        
        return export_paths
    
    # ============================================================
    # CSV EXPORT
    # ============================================================
    
    def _export_csv(
        self,
        result: TMEAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """
        Export to CSV format.
        
        Creates 3 CSV files:
            - {prefix}_interactions.csv - Interaction pairs
            - {prefix}_tacs.csv - TACS features (if available)
            - {prefix}_prognostic.csv - Prognostic scores (if available)
        """
        # Export interaction pairs
        if result.interaction_pairs:
            interactions_path = output_dir / f"{prefix}_interactions.csv"
            self._export_interactions_csv(result.interaction_pairs, interactions_path)
        
        # Export TACS features
        if result.tacs_features:
            tacs_path = output_dir / f"{prefix}_tacs.csv"
            self._export_tacs_csv(result.tacs_features, tacs_path)
        
        # Export prognostic scores
        if result.prognostic_scores:
            prog_path = output_dir / f"{prefix}_prognostic.csv"
            self._export_prognostic_csv(result.prognostic_scores, prog_path)
        
        # Return main interactions file
        return output_dir / f"{prefix}_interactions.csv"
    
    def _export_interactions_csv(
        self,
        interaction_pairs: List,
        output_path: Path
    ):
        """Export interaction pairs to CSV."""
        rows = []
        
        for pair in interaction_pairs:
            row = {
                'source_id': pair.source_id,
                'target_id': pair.target_id,
                'source_type': pair.source_type,
                'target_type': pair.target_type,
                'distance': pair.distance,
                'contact': pair.contact,
                'relative_angle': pair.relative_angle,
                'angle_to_boundary_normal': pair.angle_to_boundary_normal,
                'angle_to_boundary_tangent': pair.angle_to_boundary_tangent,
                'interaction_type': pair.interaction_type,
            }
            
            # Add interaction point if available
            if pair.interaction_point:
                row['interaction_x'] = pair.interaction_point[0]
                row['interaction_y'] = pair.interaction_point[1]
            
            # Add boundary point if available
            if pair.nearest_boundary_point:
                row['boundary_x'] = pair.nearest_boundary_point[0]
                row['boundary_y'] = pair.nearest_boundary_point[1]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def _export_tacs_csv(
        self,
        tacs_features: Dict[str, Any],
        output_path: Path
    ):
        """Export TACS features to CSV."""
        # Create a single-row DataFrame with all TACS metrics
        df = pd.DataFrame([tacs_features])
        df.to_csv(output_path, index=False)
    
    def _export_prognostic_csv(
        self,
        prognostic_scores: Dict[str, float],
        output_path: Path
    ):
        """Export prognostic scores to CSV."""
        # Create DataFrame with score names and values
        df = pd.DataFrame([
            {'score_name': k, 'score_value': v}
            for k, v in prognostic_scores.items()
        ])
        df.to_csv(output_path, index=False)
    
    # ============================================================
    # EXCEL EXPORT
    # ============================================================
    
    def _export_excel(
        self,
        result: TMEAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """
        Export to Excel format (multi-sheet workbook).
        
        Sheets:
            1. Interactions - Interaction pairs
            2. TACS Features - TACS metrics
            3. Prognostic Scores - Clinical scores
            4. Morphological - Morphological features
            5. Spatial - Spatial features
            6. Summary - Overall summary
        """
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl required for Excel export. Install with: pip install openpyxl")
        
        output_path = output_dir / f"{prefix}_analysis.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Interaction pairs
            if result.interaction_pairs:
                interactions_data = []
                for pair in result.interaction_pairs:
                    row = {
                        'source_id': pair.source_id,
                        'target_id': pair.target_id,
                        'source_type': pair.source_type,
                        'target_type': pair.target_type,
                        'distance': pair.distance,
                        'contact': pair.contact,
                        'relative_angle': pair.relative_angle,
                        'angle_to_normal': pair.angle_to_boundary_normal,
                        'interaction_type': pair.interaction_type,
                    }
                    interactions_data.append(row)
                
                df_interactions = pd.DataFrame(interactions_data)
                df_interactions.to_excel(writer, sheet_name='Interactions', index=False)
            
            # Sheet 2: TACS Features
            if result.tacs_features:
                df_tacs = pd.DataFrame([result.tacs_features])
                df_tacs.to_excel(writer, sheet_name='TACS Features', index=False)
            
            # Sheet 3: Prognostic Scores
            if result.prognostic_scores:
                prog_data = [
                    {'Score Name': k, 'Value': v}
                    for k, v in result.prognostic_scores.items()
                ]
                df_prog = pd.DataFrame(prog_data)
                df_prog.to_excel(writer, sheet_name='Prognostic Scores', index=False)
            
            # Sheet 4: Morphological Features
            if result.morphological_features:
                df_morph = pd.DataFrame([result.morphological_features])
                df_morph.to_excel(writer, sheet_name='Morphological', index=False)
            
            # Sheet 5: Spatial Features
            if result.spatial_features:
                # Filter out dict values (like interaction_type_distribution)
                spatial_simple = {
                    k: v for k, v in result.spatial_features.items()
                    if not isinstance(v, dict)
                }
                df_spatial = pd.DataFrame([spatial_simple])
                df_spatial.to_excel(writer, sheet_name='Spatial', index=False)
            
            # Sheet 6: Orientation Features
            if result.orientation_features:
                df_orient = pd.DataFrame([result.orientation_features])
                df_orient.to_excel(writer, sheet_name='Orientation', index=False)
            
            # Sheet 7: Density Features
            if result.density_features:
                df_density = pd.DataFrame([result.density_features])
                df_density.to_excel(writer, sheet_name='Density', index=False)
            
            # Sheet 8: Summary
            summary_data = {
                'Metric': ['Analysis ID', 'Mode', 'N Interactions', 'Processing Time (s)'],
                'Value': [
                    result.analysis_id,
                    result.mode.value,
                    len(result.interaction_pairs),
                    result.processing_time if result.processing_time else 0
                ]
            }
            
            # Add TACS summary if available
            if result.tacs_features:
                summary_data['Metric'].extend([
                    'Dominant TACS',
                    'TACS-1 Ratio',
                    'TACS-2 Ratio',
                    'TACS-3 Ratio',
                    'Mean TACS Score'
                ])
                summary_data['Value'].extend([
                    result.tacs_features.get('dominant_tacs_type', 'N/A'),
                    f"{result.tacs_features.get('tacs1_ratio', 0):.1%}",
                    f"{result.tacs_features.get('tacs2_ratio', 0):.1%}",
                    f"{result.tacs_features.get('tacs3_ratio', 0):.1%}",
                    f"{result.tacs_features.get('mean_tacs_score', 0):.3f}"
                ])
            
            # Add prognostic summary if available
            if result.prognostic_scores:
                summary_data['Metric'].append('Overall TME Risk')
                summary_data['Value'].append(
                    f"{result.prognostic_scores.get('overall_tme_risk_score', 0):.3f}"
                )
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        return output_path
    
    # ============================================================
    # JSON EXPORT
    # ============================================================
    
    def _export_json(
        self,
        result: TMEAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """Export to JSON format (complete results)."""
        # Convert result to dictionary
        output_data = {
            'analysis_id': result.analysis_id,
            'mode': result.mode.value,
            'processing_time': result.processing_time,
            'parameters': result.parameters,
            'n_interactions': len(result.interaction_pairs),
            'tumor_regions': result.tumor_regions,
        }
        
        # Add interaction pairs
        if result.interaction_pairs:
            output_data['interaction_pairs'] = [
                pair.to_dict() for pair in result.interaction_pairs
            ]
        
        # Add all feature sets
        if result.tacs_features:
            output_data['tacs_features'] = result.tacs_features
        
        if result.morphological_features:
            output_data['morphological_features'] = result.morphological_features
        
        if result.spatial_features:
            # Convert to serializable format
            spatial_clean = {}
            for k, v in result.spatial_features.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    spatial_clean[k] = v
                elif isinstance(v, dict):
                    spatial_clean[k] = v
                elif isinstance(v, np.integer):
                    spatial_clean[k] = int(v)
                elif isinstance(v, np.floating):
                    spatial_clean[k] = float(v)
            output_data['spatial_features'] = spatial_clean
        
        if result.orientation_features:
            output_data['orientation_features'] = result.orientation_features
        
        if result.density_features:
            output_data['density_features'] = result.density_features
        
        if result.prognostic_scores:
            output_data['prognostic_scores'] = result.prognostic_scores
        
        if result.summary:
            output_data['summary'] = result.summary
        
        # Save to JSON
        output_path = output_dir / f"{prefix}_results.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_path
    
    # ============================================================
    # GEOJSON EXPORT
    # ============================================================
    
    def _export_geojson(
        self,
        result: TMEAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """
        Export to GeoJSON format for spatial visualization.
        
        Creates features for:
            - Interaction pairs (as lines connecting components)
            - TACS-classified fibers (color-coded)
        """
        features = []
        
        # Create features for interaction pairs
        for pair in result.interaction_pairs:
            # Skip if no spatial information
            if not pair.interaction_point:
                continue
            
            # Create point feature for interaction location
            properties = {
                'source_id': pair.source_id,
                'target_id': pair.target_id,
                'source_type': pair.source_type,
                'target_type': pair.target_type,
                'distance': float(pair.distance),
                'interaction_type': pair.interaction_type,
            }
            
            # Add TACS classification for fiber-tumor interactions
            if pair.interaction_type and 'TACS' in pair.interaction_type:
                properties['tacs_type'] = pair.interaction_type
                properties['angle_to_normal'] = pair.angle_to_boundary_normal
                
                # Color coding for visualization
                if pair.interaction_type == 'TACS-1':
                    properties['color'] = 'blue'
                    properties['risk'] = 'low'
                elif pair.interaction_type == 'TACS-2':
                    properties['color'] = 'green'
                    properties['risk'] = 'medium'
                elif pair.interaction_type == 'TACS-3':
                    properties['color'] = 'red'
                    properties['risk'] = 'high'
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [
                        float(pair.interaction_point[0]),
                        float(pair.interaction_point[1])
                    ]
                },
                'properties': properties
            }
            
            features.append(feature)
            
            # If boundary point available, add line from interaction to boundary
            if pair.nearest_boundary_point:
                line_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [
                            [float(pair.interaction_point[0]), float(pair.interaction_point[1])],
                            [float(pair.nearest_boundary_point[0]), float(pair.nearest_boundary_point[1])]
                        ]
                    },
                    'properties': {
                        'type': 'interaction_line',
                        'source_id': pair.source_id,
                        'interaction_type': pair.interaction_type
                    }
                }
                features.append(line_feature)
        
        # Create FeatureCollection
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'properties': {
                'analysis_id': result.analysis_id,
                'mode': result.mode.value,
                'n_interactions': len(result.interaction_pairs),
            }
        }
        
        # Add TACS summary to metadata
        if result.tacs_features:
            geojson['properties']['tacs_summary'] = {
                'tacs1_ratio': result.tacs_features.get('tacs1_ratio', 0),
                'tacs2_ratio': result.tacs_features.get('tacs2_ratio', 0),
                'tacs3_ratio': result.tacs_features.get('tacs3_ratio', 0),
                'dominant_tacs': result.tacs_features.get('dominant_tacs_type'),
            }
        
        # Save to file
        output_path = output_dir / f"{prefix}_interactions.geojson"
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return output_path
    
    # ============================================================
    # SPECIALIZED EXPORTS
    # ============================================================
    
    def export_tacs_summary(
        self,
        result: TMEAnalysisResult,
        output_path: Union[str, Path]
    ):
        """
        Export TACS summary report as text file.
        
        Creates a human-readable summary of TACS analysis results.
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TACS Analysis Summary Report\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Analysis ID: {result.analysis_id}\n")
            f.write(f"Analysis Mode: {result.mode.value}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n\n")
            
            if result.tacs_features:
                f.write(result.get_tacs_summary() + "\n\n")
            
            if result.prognostic_scores:
                f.write(result.get_prognostic_summary() + "\n\n")
            
            if result.summary:
                f.write("Additional Metrics:\n")
                f.write("-"*60 + "\n")
                for key, value in result.summary.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
    
    def export_interaction_summary(
        self,
        result: TMEAnalysisResult,
        output_path: Union[str, Path]
    ):
        """
        Export interaction summary by type.
        
        Groups interactions by type and provides counts/statistics.
        """
        output_path = Path(output_path)
        
        # Group interactions by type
        from collections import Counter
        
        interaction_types = Counter([
            f"{p.source_type}-{p.target_type}"
            for p in result.interaction_pairs
        ])
        
        with open(output_path, 'w') as f:
            f.write("Interaction Summary\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Interactions: {len(result.interaction_pairs)}\n\n")
            
            f.write("By Interaction Type:\n")
            f.write("-"*60 + "\n")
            for itype, count in interaction_types.most_common():
                f.write(f"  {itype}: {count}\n")
            
            # TACS breakdown if available
            if any('tumor_boundary' in p.target_type for p in result.interaction_pairs):
                f.write("\nTACS Classification:\n")
                f.write("-"*60 + "\n")
                
                tacs_counts = Counter([
                    p.interaction_type for p in result.interaction_pairs
                    if p.interaction_type and 'TACS' in p.interaction_type
                ])
                
                for tacs_type, count in tacs_counts.items():
                    f.write(f"  {tacs_type}: {count}\n")


# Export convenience function
def export_tme_analysis_results(
    result: TMEAnalysisResult,
    output_dir: Union[str, Path],
    formats: List[str] = ["csv", "excel", "json"],
    prefix: str = "tme_analysis"
) -> Dict[str, str]:
    """
    Convenience function to export TME analysis results.
    
    Args:
        result: TMEAnalysisResult to export
        output_dir: Output directory
        formats: Export formats
        prefix: Filename prefix
        
    Returns:
        Dictionary mapping format to file path
    """
    exporter = TMEAnalysisExporter()
    return exporter.export(result, output_dir, formats, prefix)