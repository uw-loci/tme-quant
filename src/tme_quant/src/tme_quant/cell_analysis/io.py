"""
Export utilities and model loaders for cell analysis.
"""

from __future__ import annotations


# ========================================================
# EXPORTERS
# ========================================================

# tme_quant/cell_analysis/io/exporters.py
"""
Export cell analysis results to various formats.

Supports CSV, Excel, JSON, and GeoJSON exports.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any
import warnings

from ..core.tme_models.cell_model import CellAnalysisResult


class CellAnalysisExporter:
    """
    Export cell analysis results to multiple formats.
    
    Formats:
        - CSV: Tabular cell data
        - Excel: Multi-sheet workbook (cells, classification, quantification, summary)
        - JSON: Complete results with metadata
        - GeoJSON: Spatial data for visualization
    """
    
    def __init__(self):
        """Initialize exporter."""
        pass
    
    def export(
        self,
        result: CellAnalysisResult,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"],
        prefix: str = "cell_analysis"
    ) -> Dict[str, str]:
        """
        Export analysis results to specified formats.
        
        Args:
            result: CellAnalysisResult to export
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
    
    def _export_csv(
        self,
        result: CellAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """Export to CSV format."""
        if not result.segmentation_result or not result.segmentation_result.cells:
            raise ValueError("No cells to export")
        
        # Prepare data
        rows = []
        for cell in result.segmentation_result.cells:
            row = {
                'cell_id': cell.cell_id,
                'centroid_x': cell.centroid[0],
                'centroid_y': cell.centroid[1],
                'area': cell.area,
                'perimeter': cell.perimeter,
                'circularity': cell.circularity,
                'eccentricity': cell.eccentricity,
                'solidity': cell.solidity,
                'extent': cell.extent,
                'major_axis_length': cell.major_axis_length,
                'minor_axis_length': cell.minor_axis_length,
                'orientation': cell.orientation,
            }
            
            # Add classification if available
            if result.classification_result:
                cell_type = result.classification_result.cell_types.get(cell.cell_id)
                confidence = result.classification_result.confidences.get(cell.cell_id)
                row['cell_type'] = cell_type.value if cell_type else None
                row['cell_type_confidence'] = confidence
            
            # Add quantification if available
            if result.quantification_result:
                measurements = result.quantification_result.measurements.get(cell.cell_id, {})
                row.update(measurements)
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_path = output_dir / f"{prefix}_cells.csv"
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _export_excel(
        self,
        result: CellAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """Export to Excel format (multi-sheet)."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("openpyxl required for Excel export. Install with: pip install openpyxl")
        
        output_path = output_dir / f"{prefix}_analysis.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Cell data
            if result.segmentation_result and result.segmentation_result.cells:
                cells_data = []
                for cell in result.segmentation_result.cells:
                    row = {
                        'cell_id': cell.cell_id,
                        'centroid_x': cell.centroid[0],
                        'centroid_y': cell.centroid[1],
                        'area': cell.area,
                        'perimeter': cell.perimeter,
                        'circularity': cell.circularity,
                        'eccentricity': cell.eccentricity,
                        'solidity': cell.solidity,
                        'major_axis': cell.major_axis_length,
                        'minor_axis': cell.minor_axis_length,
                        'orientation': cell.orientation,
                    }
                    
                    # Add classification
                    if result.classification_result:
                        cell_type = result.classification_result.cell_types.get(cell.cell_id)
                        row['cell_type'] = cell_type.value if cell_type else None
                        row['confidence'] = result.classification_result.confidences.get(cell.cell_id)
                    
                    cells_data.append(row)
                
                df_cells = pd.DataFrame(cells_data)
                df_cells.to_excel(writer, sheet_name='Cells', index=False)
            
            # Sheet 2: Classification summary
            if result.classification_result:
                class_data = {
                    'Cell Type': [ct.value for ct in result.classification_result.type_counts.keys()],
                    'Count': list(result.classification_result.type_counts.values()),
                    'Ratio': list(result.classification_result.type_ratios.values())
                }
                df_class = pd.DataFrame(class_data)
                df_class.to_excel(writer, sheet_name='Classification', index=False)
            
            # Sheet 3: Quantification summary
            if result.quantification_result and result.quantification_result.population_stats:
                pop_data = [
                    {'Metric': k, 'Value': v}
                    for k, v in result.quantification_result.population_stats.items()
                ]
                df_quant = pd.DataFrame(pop_data)
                df_quant.to_excel(writer, sheet_name='Population Stats', index=False)
            
            # Sheet 4: Spatial statistics
            if result.quantification_result and result.quantification_result.spatial_stats:
                spatial_data = [
                    {'Metric': k, 'Value': v}
                    for k, v in result.quantification_result.spatial_stats.items()
                ]
                df_spatial = pd.DataFrame(spatial_data)
                df_spatial.to_excel(writer, sheet_name='Spatial Stats', index=False)
            
            # Sheet 5: Overall summary
            summary_data = {
                'Metric': ['Image ID', 'Total Cells'],
                'Value': [
                    result.image_id,
                    result.segmentation_result.total_cell_count if result.segmentation_result else 0
                ]
            }
            
            if result.segmentation_result:
                summary_data['Metric'].extend([
                    'Mean Cell Area',
                    'Mean Circularity'
                ])
                summary_data['Value'].extend([
                    result.segmentation_result.mean_cell_area,
                    result.segmentation_result.mean_circularity
                ])
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        return output_path
    
    def _export_json(
        self,
        result: CellAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """Export to JSON format."""
        # Convert result to dictionary
        output_data = result.to_dict()
        
        # Save to JSON
        output_path = output_dir / f"{prefix}_results.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_path
    
    def _export_geojson(
        self,
        result: CellAnalysisResult,
        output_dir: Path,
        prefix: str
    ) -> Path:
        """Export to GeoJSON format for spatial visualization."""
        if not result.segmentation_result or not result.segmentation_result.cells:
            raise ValueError("No cells to export")
        
        features = []
        
        for cell in result.segmentation_result.cells:
            # Create polygon from boundary
            if len(cell.boundary) > 2:
                coordinates = [cell.boundary.tolist()]
            else:
                # Approximate as circle if no boundary
                radius = np.sqrt(cell.area / np.pi)
                theta = np.linspace(0, 2*np.pi, 32)
                x = cell.centroid[0] + radius * np.cos(theta)
                y = cell.centroid[1] + radius * np.sin(theta)
                coordinates = [np.column_stack([x, y]).tolist()]
            
            # Properties
            properties = {
                'cell_id': int(cell.cell_id),
                'area': float(cell.area),
                'circularity': float(cell.circularity),
            }
            
            # Add classification
            if result.classification_result:
                cell_type = result.classification_result.cell_types.get(cell.cell_id)
                properties['cell_type'] = cell_type.value if cell_type else None
                properties['confidence'] = result.classification_result.confidences.get(cell.cell_id)
            
            # Create feature
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': coordinates
                },
                'properties': properties
            }
            
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        # Save to file
        output_path = output_dir / f"{prefix}_cells.geojson"
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return output_path
# ========================================================
# MODEL LOADERS
# ========================================================

## Model Loaders

### Location: `tme_quant/cell_analysis/io/model_loaders.py`

"""
Load pre-trained models for StarDist and Cellpose.

Handles model downloading, caching, and validation.
"""

from pathlib import Path
from typing import Optional, Union
import warnings


class ModelLoader:
    """
    Load and manage pre-trained segmentation models.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model loader.
        
        Args:
            cache_dir: Directory for caching models (default: ~/.tme_quant/models)
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.tme_quant' / 'models'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_stardist_model(
        self,
        model_name: str = "2D_versatile_fluo",
        dimension: str = "2D"
    ):
        """
        Load StarDist model.
        
        Args:
            model_name: Model name (e.g., "2D_versatile_fluo", "2D_versatile_he")
            dimension: "2D" or "3D"
            
        Returns:
            Loaded StarDist model
        """
        try:
            if dimension == "2D":
                from stardist.models import StarDist2D
                model = StarDist2D.from_pretrained(model_name)
            else:
                from stardist.models import StarDist3D
                model = StarDist3D.from_pretrained(model_name)
            
            return model
        
        except ImportError:
            raise ImportError(
                "StarDist not installed. Install with: pip install stardist"
            )
        except Exception as e:
            warnings.warn(f"Failed to load StarDist model {model_name}: {e}")
            return None
    
    def load_cellpose_model(
        self,
        model_type: str = "nuclei",
        gpu: bool = False
    ):
        """
        Load Cellpose model.
        
        Args:
            model_type: Model type ("nuclei", "cyto", "cyto2")
            gpu: Use GPU if available
            
        Returns:
            Loaded Cellpose model
        """
        try:
            from cellpose import models
            
            # Check GPU availability
            if gpu:
                import torch
                gpu = torch.cuda.is_available()
                if not gpu:
                    warnings.warn("GPU requested but not available, using CPU")
            
            model = models.Cellpose(gpu=gpu, model_type=model_type)
            return model
        
        except ImportError:
            raise ImportError(
                "Cellpose not installed. Install with: pip install cellpose"
            )
        except Exception as e:
            warnings.warn(f"Failed to load Cellpose model {model_type}: {e}")
            return None
    
    def list_available_models(self) -> dict:
        """
        List available pre-trained models.
        
        Returns:
            Dictionary of available models by framework
        """
        models = {
            'stardist_2d': [
                '2D_versatile_fluo',
                '2D_versatile_he',
                '2D_paper_dsb2018'
            ],
            'stardist_3d': [
                '3D_demo'
            ],
            'cellpose': [
                'nuclei',
                'cyto',
                'cyto2'
            ]
        }
        
        return models