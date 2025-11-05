"""
ROI Manager module for CurveAlign Napari plugin.

Provides full ROI management functionality matching MATLAB CurveAlign ROI Manager:
- ROI creation: Rectangle, Freehand, Ellipse, Polygon
- ROI management: Save, Load, Delete, Rename, Combine
- ROI analysis integration with CurveAlign
- ROI table display with analysis results
- Export/Import: Text (CSV), Mask (TIFF)
"""

import os
import json
import struct
import zipfile
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import napari
    from napari.layers import Shapes
    from napari.types import LayerDataTuple
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False

try:
    from skimage import io
    from skimage.measure import label, regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import curvealign_py as curvealign
    from curvealign_py.types import Boundary
    HAS_CURVEALIGN = True
except ImportError:
    HAS_CURVEALIGN = False

try:
    import roifile
    HAS_ROIFILE = True
except ImportError:
    HAS_ROIFILE = False


class ROIShape(Enum):
    """ROI shape types."""
    RECTANGLE = "Rectangle"
    FREEHAND = "Freehand"
    ELLIPSE = "Ellipse"
    POLYGON = "Polygon"


class ROIAnalysisMethod(Enum):
    """ROI analysis methods."""
    CURVELETS = "Curvelets"
    CTFIRE = "CT-FIRE"
    POST_ANALYSIS = "Post-Analysis"  # Analysis on previously analyzed whole image


@dataclass
class ROI:
    """Represents a single ROI with metadata."""
    id: int
    name: str
    shape: ROIShape
    coordinates: np.ndarray  # Shape-specific coordinates
    center: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    analysis_result: Optional[Dict] = None
    analysis_method: Optional[ROIAnalysisMethod] = None
    boundary_mode: Optional[str] = None
    crop_mode: bool = True  # True: cropped ROI, False: mask-based
    metadata: Dict = field(default_factory=dict)
    
    def to_boundary(self, image_shape: Tuple[int, int]) -> Boundary:
        """Convert ROI to CurveAlign Boundary object."""
        if not HAS_CURVEALIGN:
            raise ImportError("curvealign_py is required")
        
        # Create mask from ROI
        mask = self.to_mask(image_shape)
        
        # Get boundary coordinates from mask
        from skimage import measure
        contours = measure.find_contours(mask, 0.5)
        if len(contours) > 0:
            # Use largest contour
            contour = max(contours, key=len)
            # Convert to (row, col) format
            boundary_coords = np.array([[int(r), int(c)] for r, c in contour])
        else:
            boundary_coords = np.array([])
        
        return Boundary(coords=boundary_coords)
    
    def to_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert ROI to binary mask."""
        mask = np.zeros(image_shape, dtype=bool)
        
        if self.shape == ROIShape.RECTANGLE:
            # Rectangle coordinates: [[x1, y1], [x2, y2]]
            x1, y1 = int(self.coordinates[0, 0]), int(self.coordinates[0, 1])
            x2, y2 = int(self.coordinates[1, 0]), int(self.coordinates[1, 1])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            mask[y1:y2, x1:x2] = True
        elif self.shape == ROIShape.ELLIPSE:
            # Ellipse coordinates: two corners of bounding box
            from skimage.draw import ellipse
            x1, y1 = self.coordinates[0]
            x2, y2 = self.coordinates[1]
            cx, cy = self.center
            
            # skimage.draw.ellipse can choke on 0 radii. Clamp to ≥1
            r_row = max(1, int(abs(y2 - cy)))
            r_col = max(1, int(abs(x2 - cx)))
            
            rr, cc = ellipse(int(cy), int(cx), r_row, r_col, shape=image_shape)
            mask[rr, cc] = True
        elif self.shape in (ROIShape.FREEHAND, ROIShape.POLYGON):
            # Polygon coordinates: array of [x, y] points
            from skimage.draw import polygon
            coords = self.coordinates.astype(int)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=image_shape)
            mask[rr, cc] = True
        
        return mask


class ROIManager:
    """
    ROI Manager for CurveAlign analysis.
    
    Manages ROIs, their analysis results, and provides integration with
    napari Shapes layer and CurveAlign API.
    """
    
    def __init__(self, viewer: Optional['napari.viewer.Viewer'] = None):
        """
        Initialize ROI Manager.
        
        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            Napari viewer instance
        """
        self.viewer = viewer
        self.rois: List[ROI] = []
        self.roi_counter = 0
        self.shapes_layer: Optional[Shapes] = None
        self.current_image_shape: Optional[Tuple[int, int]] = None
        
    def set_viewer(self, viewer: 'napari.viewer.Viewer'):
        """Set the napari viewer."""
        self.viewer = viewer
        
    def set_image_shape(self, shape: Tuple[int, int]):
        """Set current image shape for ROI operations."""
        self.current_image_shape = shape
        
    def create_shapes_layer(self) -> Shapes:
        """Create or get napari Shapes layer for ROIs."""
        if self.viewer is None:
            raise ValueError("Viewer must be set before creating shapes layer")
        
        # Look for existing shapes layer
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes) and layer.name == "ROIs":
                self.shapes_layer = layer
                return layer
        
        # Create new shapes layer
        self.shapes_layer = self.viewer.add_shapes(
            name="ROIs",
            shape_type="rectangle",
            edge_color="cyan",
            face_color="transparent",
            edge_width=2
        )
        return self.shapes_layer
    
    def add_roi(
        self,
        coordinates: np.ndarray,
        shape: ROIShape,
        name: Optional[str] = None
    ) -> ROI:
        """
        Add a new ROI.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Shape-specific coordinates
        shape : ROIShape
            Type of ROI shape
        name : str, optional
            ROI name (auto-generated if not provided)
        
        Returns
        -------
        ROI
            Created ROI object
        """
        # Ensure coordinates are float array for consistency
        coordinates = np.asarray(coordinates, dtype=float)
        
        if name is None:
            name = f"ROI_{self.roi_counter + 1}"
        
        # Calculate center and area
        if shape == ROIShape.RECTANGLE:
            center = (
                (coordinates[0, 0] + coordinates[1, 0]) / 2,
                (coordinates[0, 1] + coordinates[1, 1]) / 2
            )
            area = abs((coordinates[1, 0] - coordinates[0, 0]) * 
                       (coordinates[1, 1] - coordinates[0, 1]))
        else:
            center = (np.mean(coordinates[:, 0]), np.mean(coordinates[:, 1]))
            if self.current_image_shape:
                temp_roi = ROI(0, name, shape, coordinates)
                mask = temp_roi.to_mask(self.current_image_shape)
                area = np.sum(mask)
            else:
                area = 0.0
        
        roi = ROI(
            id=self.roi_counter,
            name=name,
            shape=shape,
            coordinates=coordinates,
            center=center,
            area=area
        )
        
        self.rois.append(roi)
        self.roi_counter += 1
        
        # Update napari shapes layer
        if self.shapes_layer is not None:
            self._update_shapes_layer()
        
        return roi
    
    def delete_roi(self, roi_id: int) -> bool:
        """Delete ROI by ID."""
        for i, roi in enumerate(self.rois):
            if roi.id == roi_id:
                self.rois.pop(i)
                if self.shapes_layer is not None:
                    self._update_shapes_layer()
                return True
        return False
    
    def rename_roi(self, roi_id: int, new_name: str) -> bool:
        """Rename ROI by ID."""
        for roi in self.rois:
            if roi.id == roi_id:
                roi.name = new_name
                return True
        return False
    
    def combine_rois(self, roi_ids: List[int], name: Optional[str] = None) -> Optional[ROI]:
        """
        Combine multiple ROIs into one.
        
        Parameters
        ----------
        roi_ids : List[int]
            IDs of ROIs to combine
        name : str, optional
            Name for combined ROI
        
        Returns
        -------
        ROI, optional
            Combined ROI or None if failed
        """
        if len(roi_ids) < 2:
            return None
        
        if self.current_image_shape is None:
            return None
        
        # Create combined mask
        combined_mask = np.zeros(self.current_image_shape, dtype=bool)
        for roi_id in roi_ids:
            roi = self.get_roi(roi_id)
            if roi:
                mask = roi.to_mask(self.current_image_shape)
                combined_mask |= mask
        
        # Convert mask to polygon
        from skimage import measure
        contours = measure.find_contours(combined_mask.astype(float), 0.5)
        if len(contours) == 0:
            return None
        
        # Use largest contour
        contour = max(contours, key=len)
        coordinates = np.asarray([[c, r] for r, c in contour], dtype=float)
        
        if name is None:
            name = f"Combined_{roi_ids[0]}"
        
        combined_roi = self.add_roi(coordinates, ROIShape.POLYGON, name)
        
        # Delete original ROIs
        for roi_id in sorted(roi_ids, reverse=True):
            self.delete_roi(roi_id)
        
        return combined_roi
    
    def get_roi(self, roi_id: int) -> Optional[ROI]:
        """Get ROI by ID."""
        for roi in self.rois:
            if roi.id == roi_id:
                return roi
        return None
    
    def analyze_roi(
        self,
        roi_id: int,
        image: np.ndarray,
        method: ROIAnalysisMethod = ROIAnalysisMethod.CURVELETS,
        options: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Analyze ROI using CurveAlign.
        
        Parameters
        ----------
        roi_id : int
            ROI ID to analyze
        image : np.ndarray
            Image data
        method : ROIAnalysisMethod
            Analysis method to use
        options : dict, optional
            CurveAlign options
        
        Returns
        -------
        dict, optional
            Analysis results
        """
        if not HAS_CURVEALIGN:
            return None
        
        roi = self.get_roi(roi_id)
        if roi is None:
            return None
        
        # Convert ROI to boundary
        boundary = roi.to_boundary(image.shape[:2])
        
        # Prepare image
        if roi.crop_mode:
            # Crop image to ROI
            mask = roi.to_mask(image.shape[:2])
            bbox = self._get_bbox(mask)
            cropped_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            # Adjust boundary coordinates
            boundary.coords = boundary.coords - np.array([bbox[0], bbox[1]])
        else:
            cropped_image = image
        
        # Run analysis
        if options is None:
            options = {}
        
        ca_options = curvealign.CurveAlignOptions(**options)
        
        if method == ROIAnalysisMethod.CURVELETS:
            result = curvealign.analyze_image(
                cropped_image,
                boundary=boundary if not roi.crop_mode else None,
                mode="curvelets",
                options=ca_options
            )
        elif method == ROIAnalysisMethod.CTFIRE:
            result = curvealign.analyze_image(
                cropped_image,
                boundary=boundary if not roi.crop_mode else None,
                mode="ctfire",
                options=ca_options
            )
        else:
            # Post-analysis - would need previously computed features
            return None
        
        # Store results
        roi.analysis_result = {
            "n_curvelets": len(result.curvelets),
            "mean_angle": result.stats.get("mean_angle", 0.0),
            "alignment": result.stats.get("alignment", 0.0),
            "density": result.stats.get("density", 0.0),
            "stats": result.stats,
            "features": result.features
        }
        roi.analysis_method = method
        
        return roi.analysis_result
    
    def get_analysis_table(self) -> pd.DataFrame:
        """
        Get ROI analysis results as DataFrame.
        
        Returns DataFrame matching MATLAB ROI Manager output table format:
        Columns: No., Image Label, ROI label, Orientation, Alignment, FeatNum,
                 Methods, Boundary, CROP, POST, Shape, Xc, Yc, Z
        """
        rows = []
        for roi in self.rois:
            result = roi.analysis_result or {}
            rows.append({
                "No.": roi.id,
                "Image Label": roi.metadata.get("image_label", ""),
                "ROI label": roi.name,
                "Orientation": f"{result.get('mean_angle', 0.0):.1f}°",
                "Alignment": f"{result.get('alignment', 0.0):.3f}",
                "FeatNum": result.get("n_curvelets", 0),
                "Methods": roi.analysis_method.value if roi.analysis_method else "",
                "Boundary": roi.boundary_mode or "",
                "CROP": "Yes" if roi.crop_mode else "No",
                "POST": "",
                "Shape": roi.shape.value,
                "Xc": roi.center[0],
                "Yc": roi.center[1],
                "Z": roi.metadata.get("z_slice", 0)
            })
        
        return pd.DataFrame(rows)
    
    def save_rois_json(self, file_path: str, roi_ids: Optional[List[int]] = None):
        """
        Save ROIs to JSON file (primary format).
        
        This is the recommended format for full data preservation.
        """
        if roi_ids is None:
            roi_ids = [roi.id for roi in self.rois]
        
        rois_data = []
        for roi_id in roi_ids:
            roi = self.get_roi(roi_id)
            if roi:
                roi_dict = {
                    "id": roi.id,
                    "name": roi.name,
                    "shape": roi.shape.value,
                    "coordinates": roi.coordinates.tolist(),
                    "center": list(roi.center),
                    "area": float(roi.area),
                    "analysis_result": roi.analysis_result,
                    "analysis_method": roi.analysis_method.value if roi.analysis_method else None,
                    "boundary_mode": roi.boundary_mode,
                    "crop_mode": roi.crop_mode,
                    "metadata": roi.metadata
                }
                rois_data.append(roi_dict)
        
        output_data = {
            "version": "1.0",
            "image_shape": list(self.current_image_shape) if self.current_image_shape else None,
            "rois": rois_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def load_rois_json(self, file_path: str) -> List[ROI]:
        """Load ROIs from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Set image shape if available
        if data.get("image_shape"):
            self.current_image_shape = tuple(data["image_shape"])
        
        loaded_rois = []
        for roi_data in data.get("rois", []):
            coords = np.asarray(roi_data["coordinates"], dtype=float)
            shape = ROIShape(roi_data["shape"])
            
            roi = self.add_roi(coords, shape, roi_data["name"])
            roi.center = tuple(roi_data["center"])
            roi.area = roi_data["area"]
            roi.analysis_result = roi_data.get("analysis_result")
            if roi_data.get("analysis_method"):
                roi.analysis_method = ROIAnalysisMethod(roi_data["analysis_method"])
            roi.boundary_mode = roi_data.get("boundary_mode")
            roi.crop_mode = roi_data.get("crop_mode", True)
            roi.metadata = roi_data.get("metadata", {})
            loaded_rois.append(roi)
        
        return loaded_rois
    
    def save_rois_fiji(self, file_path: str, roi_ids: Optional[List[int]] = None):
        """
        Save ROIs to Fiji/ImageJ ROI format (.roi or .zip).
        
        Supports both single .roi file and RoiSet.zip format.
        """
        if not HAS_ROIFILE:
            # Fallback: create simple text-based ROI format
            self._save_rois_fiji_fallback(file_path, roi_ids)
            return
        
        if roi_ids is None:
            roi_ids = [roi.id for roi in self.rois]
        
        # If multiple ROIs, save as ZIP
        if len(roi_ids) > 1 or file_path.endswith('.zip'):
            self._save_rois_fiji_zip(file_path, roi_ids)
        else:
            # Single ROI
            roi = self.get_roi(roi_ids[0])
            if roi:
                self._save_roi_fiji_single(file_path, roi)
    
    def _save_rois_fiji_zip(self, file_path: str, roi_ids: List[int]):
        """Save multiple ROIs as Fiji RoiSet.zip."""
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Track used filenames to handle duplicates
            used_names = set()
            
            # Save each ROI as individual .roi file
            for i, roi_id in enumerate(roi_ids):
                roi = self.get_roi(roi_id)
                if roi:
                    # Ensure unique filename
                    base_name = roi.name
                    roi_filename = f"{base_name}.roi"
                    counter = 1
                    while roi_filename in used_names:
                        roi_filename = f"{base_name}_{counter}.roi"
                        counter += 1
                    used_names.add(roi_filename)
                    
                    roi_file = os.path.join(temp_dir, roi_filename)
                    self._save_roi_fiji_single(roi_file, roi)
            
            # Create ZIP file
            if not file_path.endswith('.zip'):
                file_path = file_path + '.zip'
            
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for filename in os.listdir(temp_dir):
                    roi_file = os.path.join(temp_dir, filename)
                    zipf.write(roi_file, filename)
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    def _save_roi_fiji_single(self, file_path: str, roi: ROI):
        """Save single ROI in Fiji .roi format."""
        if HAS_ROIFILE:
            # Use roifile library if available
            try:
                import roifile as rf
                
                # Convert ROI to ImageJ format
                # Create using frompoints, then set the roitype attribute
                if roi.shape == ROIShape.RECTANGLE:
                    # Rectangle: two corners [[x1, y1], [x2, y2]]
                    x1, y1 = roi.coordinates[0]
                    x2, y2 = roi.coordinates[1]
                    points = np.array([[x1, y1], [x2, y2]])
                    fiji_roi = rf.ImagejRoi.frompoints(points, name=roi.name)
                    fiji_roi.roitype = rf.ROI_TYPE.RECT
                    
                elif roi.shape == ROIShape.ELLIPSE:
                    # Ellipse: bbox corners [[x1, y1], [x2, y2]]
                    x1, y1 = roi.coordinates[0]
                    x2, y2 = roi.coordinates[1]
                    points = np.array([[x1, y1], [x2, y2]])
                    fiji_roi = rf.ImagejRoi.frompoints(points, name=roi.name)
                    fiji_roi.roitype = rf.ROI_TYPE.OVAL
                    
                elif roi.shape == ROIShape.POLYGON:
                    # Polygon: array of points
                    fiji_roi = rf.ImagejRoi.frompoints(roi.coordinates, name=roi.name)
                    fiji_roi.roitype = rf.ROI_TYPE.POLYGON
                    
                elif roi.shape == ROIShape.FREEHAND:
                    # Freehand: array of points
                    fiji_roi = rf.ImagejRoi.frompoints(roi.coordinates, name=roi.name)
                    fiji_roi.roitype = rf.ROI_TYPE.FREEHAND
                    
                else:
                    return
                
                fiji_roi.tofile(file_path)
            except Exception as e:
                print(f"roifile save failed: {e}, using fallback")
                self._save_rois_fiji_fallback(file_path, [roi.id])
        else:
            self._save_rois_fiji_fallback(file_path, [roi.id])
    
    
    def _save_rois_fiji_fallback(self, file_path: str, roi_ids: List[int]):
        """Fallback: Save as simple text format that Fiji can import."""
        rows = []
        for roi_id in roi_ids:
            roi = self.get_roi(roi_id)
            if roi:
                # Save coordinates in format Fiji can import
                coords_str = ";".join([f"{x},{y}" for x, y in roi.coordinates])
                rows.append({
                    "Name": roi.name,
                    "Type": roi.shape.value,
                    "Coordinates": coords_str
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path.replace('.roi', '.txt').replace('.zip', '.txt'), 
                  index=False, sep='\t')
    
    def load_rois_fiji(self, file_path: str) -> List[ROI]:
        """Load ROIs from Fiji/ImageJ ROI format (.roi or .zip)."""
        if not HAS_ROIFILE:
            # Try fallback text format
            return self._load_rois_fiji_fallback(file_path)
        
        loaded_rois = []
        
        try:
            import roifile as rf
            
            if file_path.endswith('.zip'):
                # Load from ZIP file
                with zipfile.ZipFile(file_path, 'r') as zipf:
                    for filename in zipf.namelist():
                        if filename.endswith('.roi'):
                            # Read bytes and use frombytes
                            roi_bytes = zipf.read(filename)
                            fiji_roi = rf.ImagejRoi.frombytes(roi_bytes)
                            roi = self._convert_fiji_roi_to_roi(fiji_roi)
                            if roi:
                                loaded_rois.append(roi)
            else:
                # Load single ROI file
                fiji_roi = rf.ImagejRoi.fromfile(file_path)
                roi = self._convert_fiji_roi_to_roi(fiji_roi)
                if roi:
                    loaded_rois.append(roi)
        except Exception as e:
            print(f"Failed to load Fiji ROI: {e}")
            # Try fallback
            loaded_rois = self._load_rois_fiji_fallback(file_path)
        
        return loaded_rois
    
    def _convert_fiji_roi_to_roi(self, fiji_roi) -> Optional[ROI]:
        """Convert Fiji/ImageJ ROI to our ROI format."""
        try:
            import roifile as rf
            
            # Determine shape type
            roi_type = fiji_roi.roitype
            
            # Handle rectangles and ovals specially - use bbox
            if roi_type == rf.ROI_TYPE.RECT:
                # For rectangle, read bbox explicitly
                left = fiji_roi.left if hasattr(fiji_roi, 'left') else 0
                top = fiji_roi.top if hasattr(fiji_roi, 'top') else 0
                width = fiji_roi.width if hasattr(fiji_roi, 'width') else 0
                height = fiji_roi.height if hasattr(fiji_roi, 'height') else 0
                
                # Store as two corners
                coords = np.asarray([
                    [left, top],
                    [left + width, top + height]
                ], dtype=float)
                shape = ROIShape.RECTANGLE
                
            elif roi_type == rf.ROI_TYPE.OVAL:
                # For oval/ellipse, read bbox explicitly
                left = fiji_roi.left if hasattr(fiji_roi, 'left') else 0
                top = fiji_roi.top if hasattr(fiji_roi, 'top') else 0
                width = fiji_roi.width if hasattr(fiji_roi, 'width') else 0
                height = fiji_roi.height if hasattr(fiji_roi, 'height') else 0
                
                # Store as two corners of bounding box
                coords = np.asarray([
                    [left, top],
                    [left + width, top + height]
                ], dtype=float)
                shape = ROIShape.ELLIPSE
                
            elif roi_type == rf.ROI_TYPE.POLYGON:
                # For polygon, use x and y arrays
                coords = fiji_roi.coordinates()
                if coords is None or len(coords) == 0:
                    return None
                coords = np.asarray(coords, dtype=float)
                shape = ROIShape.POLYGON
                
            elif roi_type == rf.ROI_TYPE.FREEHAND or roi_type == rf.ROI_TYPE.FREEROI:
                # For freehand, use x and y arrays
                coords = fiji_roi.coordinates()
                if coords is None or len(coords) == 0:
                    return None
                coords = np.asarray(coords, dtype=float)
                shape = ROIShape.FREEHAND
                
            else:
                # Default: try to get coordinates and treat as polygon
                coords = fiji_roi.coordinates()
                if coords is None or len(coords) == 0:
                    return None
                coords = np.asarray(coords, dtype=float)
                shape = ROIShape.POLYGON
            
            # Get name
            name = fiji_roi.name if hasattr(fiji_roi, 'name') else None
            
            # Add ROI
            roi = self.add_roi(coords, shape, name)
            return roi
        except Exception as e:
            print(f"Failed to convert Fiji ROI: {e}")
            return None
    
    def _load_rois_fiji_fallback(self, file_path: str) -> List[ROI]:
        """Fallback: Load from text format."""
        txt_file = file_path.replace('.roi', '.txt').replace('.zip', '.txt')
        if not os.path.exists(txt_file):
            return []
        
        df = pd.read_csv(txt_file, sep='\t')
        loaded_rois = []
        
        for _, row in df.iterrows():
            # Parse coordinates
            coord_pairs = row["Coordinates"].split(";")
            coords = []
            for pair in coord_pairs:
                x, y = map(float, pair.split(","))
                coords.append([x, y])
            
            coords = np.asarray(coords, dtype=float)
            shape = ROIShape(row["Type"])
            
            roi = self.add_roi(coords, shape, row["Name"])
            loaded_rois.append(roi)
        
        return loaded_rois
    
    def save_rois_csv(self, file_path: str, roi_ids: Optional[List[int]] = None):
        """Save ROIs to CSV file (simple format)."""
        if roi_ids is None:
            roi_ids = [roi.id for roi in self.rois]
        
        rows = []
        for roi_id in roi_ids:
            roi = self.get_roi(roi_id)
            if roi:
                rows.append({
                    "ID": roi.id,
                    "Name": roi.name,
                    "Shape": roi.shape.value,
                    "Center_X": roi.center[0],
                    "Center_Y": roi.center[1],
                    "Area": roi.area,
                    "Coordinates": str(roi.coordinates.tolist())
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
    
    def load_rois_csv(self, file_path: str) -> List[ROI]:
        """Load ROIs from CSV file."""
        df = pd.read_csv(file_path)
        loaded_rois = []
        
        for _, row in df.iterrows():
            # Parse coordinates from string
            import ast
            coords = np.asarray(ast.literal_eval(row["Coordinates"]), dtype=float)
            shape = ROIShape(row["Shape"])
            
            roi = self.add_roi(coords, shape, row["Name"])
            roi.center = (row["Center_X"], row["Center_Y"])
            roi.area = row["Area"]
            loaded_rois.append(roi)
        
        return loaded_rois
    
    def save_rois(self, file_path: str, roi_ids: Optional[List[int]] = None, format: str = 'auto'):
        """
        Save ROIs to file in specified format.
        
        Parameters
        ----------
        file_path : str
            Output file path
        roi_ids : List[int], optional
            IDs of ROIs to save (default: all)
        format : str
            Format to use: 'json', 'fiji', 'csv', 'mask', or 'auto' (detect from extension)
        """
        if format == 'auto':
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext in ['.roi', '.zip']:
                format = 'fiji'
            elif ext == '.csv':
                format = 'csv'
            elif ext in ['.tif', '.tiff']:
                format = 'mask'
            else:
                format = 'json'  # Default
        
        if format == 'json':
            self.save_rois_json(file_path, roi_ids)
        elif format == 'fiji':
            self.save_rois_fiji(file_path, roi_ids)
        elif format == 'csv':
            self.save_rois_csv(file_path, roi_ids)
        elif format == 'mask':
            # Save all ROIs as separate mask files
            for roi_id in (roi_ids or [r.id for r in self.rois]):
                roi = self.get_roi(roi_id)
                if roi and self.current_image_shape:
                    base = os.path.splitext(file_path)[0]
                    mask_file = f"{base}_{roi.name}.tif"
                    self.save_roi_mask(mask_file, roi_id, self.current_image_shape)
    
    def load_rois(self, file_path: str, format: str = 'auto') -> List[ROI]:
        """
        Load ROIs from file.
        
        Parameters
        ----------
        file_path : str
            Input file path
        format : str
            Format: 'json', 'fiji', 'csv', 'mask', or 'auto'
        
        Returns
        -------
        List[ROI]
            Loaded ROIs
        """
        if format == 'auto':
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.json':
                format = 'json'
            elif ext in ['.roi', '.zip']:
                format = 'fiji'
            elif ext == '.csv':
                format = 'csv'
            elif ext in ['.tif', '.tiff']:
                format = 'mask'
            else:
                format = 'json'  # Default
        
        if format == 'json':
            return self.load_rois_json(file_path)
        elif format == 'fiji':
            return self.load_rois_fiji(file_path)
        elif format == 'csv':
            return self.load_rois_csv(file_path)
        elif format == 'mask':
            roi = self.load_roi_from_mask(file_path)
            return [roi] if roi else []
        
        return []
    
    def save_roi_mask(self, file_path: str, roi_id: int, image_shape: Tuple[int, int]):
        """Save ROI as binary mask image."""
        roi = self.get_roi(roi_id)
        if roi:
            mask = roi.to_mask(image_shape)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Use available I/O library
            if HAS_SKIMAGE:
                io.imsave(file_path, mask_uint8)
            else:
                # Fallback to imageio or PIL
                try:
                    import imageio.v3 as iio
                    iio.imwrite(file_path, mask_uint8)
                except ImportError:
                    try:
                        from PIL import Image
                        Image.fromarray(mask_uint8).save(file_path)
                    except ImportError:
                        print("No image I/O library available (skimage, imageio, or PIL required)")
    
    def load_roi_from_mask(self, file_path: str, name: Optional[str] = None) -> Optional[ROI]:
        """Load ROI from binary mask image."""
        if not HAS_SKIMAGE:
            return None
        
        mask = io.imread(file_path)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = mask > 127  # Threshold
        
        # Get boundary from mask
        from skimage import measure
        contours = measure.find_contours(mask.astype(float), 0.5)
        if len(contours) == 0:
            return None
        
        # Use largest contour
        contour = max(contours, key=len)
        coordinates = np.asarray([[c, r] for r, c in contour], dtype=float)

        if name is None:
            name = f"ROI_from_mask_{self.roi_counter + 1}"
        
        return self.add_roi(coordinates, ROIShape.POLYGON, name)
    
    def _update_shapes_layer(self):
        """Update napari shapes layer with current ROIs."""
        if self.shapes_layer is None:
            return
        
        # Clear existing shapes
        self.shapes_layer.data = []
        
        # Add all ROIs using proper API
        for roi in self.rois:
            if roi.shape == ROIShape.RECTANGLE:
                # Use add_rectangles for rectangles
                self.shapes_layer.add_rectangles(
                    [roi.coordinates],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.ELLIPSE:
                # Use add_ellipses for ellipses
                self.shapes_layer.add_ellipses(
                    [roi.coordinates],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.POLYGON:
                # Use add_polygons for filled polygons
                self.shapes_layer.add_polygons(
                    [roi.coordinates],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.FREEHAND:
                # Use add_paths for unfilled freehand
                self.shapes_layer.add_paths(
                    [roi.coordinates],
                    edge_color="cyan",
                    edge_width=2
                )
    
    def _get_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from mask (y1, x1, y2, x2)."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, mask.shape[0], mask.shape[1])
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return (y1, x1, y2 + 1, x2 + 1)

