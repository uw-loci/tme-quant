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
from typing import List, Dict, Optional, Tuple, Union, Iterable, Sequence, Any
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
    from skimage.color import rgb2gray
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, binary_erosion, disk
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    binary_dilation = None
    binary_erosion = None
    disk = None

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
    annotation_type: str = "custom_annotation"
    source_object_ids: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

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
            contour = max(contours, key=len)
            # Convert to (row, col) format
            boundary_coords = np.array([[int(r), int(c)] for r, c in contour])
            return Boundary(kind="polygon", data=boundary_coords)

        # Fallback to mask boundary definition
        return Boundary(kind="mask", data=mask.astype(np.uint8))

    def to_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert ROI to binary mask."""
        mask = np.zeros(image_shape, dtype=bool)

        if self.shape == ROIShape.RECTANGLE:
            x1, y1 = int(self.coordinates[0, 0]), int(self.coordinates[0, 1])
            x2, y2 = int(self.coordinates[1, 0]), int(self.coordinates[1, 1])
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            mask[y1:y2, x1:x2] = True
        elif self.shape == ROIShape.ELLIPSE:
            from skimage.draw import ellipse
            x1, y1 = self.coordinates[0]
            x2, y2 = self.coordinates[1]
            cx, cy = self.center
            r_row = max(1, int(abs(y2 - cy)))
            r_col = max(1, int(abs(x2 - cx)))
            rr, cc = ellipse(int(cy), int(cx), r_row, r_col, shape=image_shape)
            mask[rr, cc] = True
        elif self.shape in (ROIShape.FREEHAND, ROIShape.POLYGON):
            from skimage.draw import polygon
            coords = self.coordinates.astype(int)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=image_shape)
            mask[rr, cc] = True

        return mask

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim >= 3:
            if image.shape[-1] in (3, 4):
                img = image[..., :3]
                if HAS_SKIMAGE:
                    return rgb2gray(img)
                return np.mean(img, axis=-1)
            return image[0]
        return image


@dataclass
class AnnotationObject:
    """Represents a detected object (cell/fiber) that can become an annotation."""
    id: int
    name: str
    kind: str  # 'cell' or 'fiber'
    boundary_rc: np.ndarray  # stored as (row, col) for napari compatibility
    centroid_rc: Tuple[float, float]
    orientation: Optional[float] = None
    area: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def centroid_xy(self) -> Tuple[float, float]:
        """Return centroid as (x, y)."""
        return (self.centroid_rc[1], self.centroid_rc[0])

    def to_roi_coordinates(self) -> np.ndarray:
        """Convert stored (row, col) boundary to ROI (x, y) coordinates."""
        if self.boundary_rc.size == 0:
            return np.empty((0, 2))
        return np.column_stack((self.boundary_rc[:, 1], self.boundary_rc[:, 0]))


class ROIManager:
    """
    ROI Manager for CurveAlign analysis.
    
    Manages ROIs, their analysis results, and provides integration with
    napari Shapes layer and CurveAlign API.
    """
    
    def __init__(
        self,
        viewer: Optional['napari.viewer.Viewer'] = None,
        overlay_callback: Optional[callable] = None,
    ):
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
        self.active_image_label: Optional[str] = None
        self.objects: Dict[str, List[AnnotationObject]] = {"cell": [], "fiber": []}
        self.object_lookup: Dict[int, AnnotationObject] = {}
        self._object_id_counter = 0
        self.object_layer: Optional[Shapes] = None
        self._active_object_filter: Sequence[str] = ("cell", "fiber")
        self.detection_distance: int = 25
        self.overlay_callback = overlay_callback
        
    def set_viewer(self, viewer: 'napari.viewer.Viewer'):
        """Set the napari viewer."""
        self.viewer = viewer
        self.shapes_layer = None
        self.object_layer = None
        
    def set_image_shape(self, shape: Tuple[int, int]):
        """Set current image shape for ROI operations."""
        self.current_image_shape = shape

    def set_active_image(self, label: Optional[str], shape: Optional[Tuple[int, int]] = None):
        """
        Set the active image label and shape so ROIs can be scoped per image.
        """
        self.active_image_label = label
        if shape is not None:
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

    @staticmethod
    def _rc_to_xy(coords: np.ndarray) -> np.ndarray:
        """Convert (row, col) coordinates to (x, y) order."""
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be N x 2 array")
        return np.column_stack((coords[:, 1], coords[:, 0]))

    @staticmethod
    def _xy_to_rc(coords: np.ndarray) -> np.ndarray:
        """Convert (x, y) coordinates to (row, col) order."""
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be N x 2 array")
        return np.column_stack((coords[:, 1], coords[:, 0]))

    def _shape_data_to_roi(self, data: np.ndarray, shape_type: str) -> Tuple[np.ndarray, ROIShape]:
        """Convert napari shape data into ROI coordinates/shape."""
        coords_rc = np.asarray(data, dtype=float)
        if coords_rc.ndim != 2 or coords_rc.shape[1] != 2:
            raise ValueError("Shape data must be N x 2 array")

        if shape_type == "rectangle":
            rows = coords_rc[:, 0]
            cols = coords_rc[:, 1]
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            coords_xy = np.array([[x1, y1], [x2, y2]], dtype=float)
            return coords_xy, ROIShape.RECTANGLE
        if shape_type == "ellipse":
            rows = coords_rc[:, 0]
            cols = coords_rc[:, 1]
            y1, y2 = rows.min(), rows.max()
            x1, x2 = cols.min(), cols.max()
            coords_xy = np.array([[x1, y1], [x2, y2]], dtype=float)
            return coords_xy, ROIShape.ELLIPSE
        if shape_type == "polygon":
            return self._rc_to_xy(coords_rc), ROIShape.POLYGON
        if shape_type == "path":
            return self._rc_to_xy(coords_rc), ROIShape.FREEHAND

        raise ValueError(f"Unsupported shape_type '{shape_type}'")

    def add_rois_from_shapes(
        self,
        indices: Optional[Iterable[int]] = None,
        *,
        annotation_type: str = "custom_annotation"
    ) -> List[ROI]:
        """Convert selected napari shapes into managed ROIs."""
        if self.shapes_layer is None:
            self.create_shapes_layer()

        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            return []

        if indices is None:
            if self.shapes_layer.selected_data:
                indices = list(self.shapes_layer.selected_data)
            else:
                indices = [len(self.shapes_layer.data) - 1]
        else:
            indices = list(indices)

        new_rois: List[ROI] = []
        for idx in indices:
            if idx < 0 or idx >= len(self.shapes_layer.data):
                continue
            try:
                coords, roi_shape = self._shape_data_to_roi(
                    self.shapes_layer.data[idx],
                    self.shapes_layer.shape_type[idx]
                )
            except ValueError as exc:
                print(f"Skipping shape {idx}: {exc}")
                continue
            roi = self.add_roi(coords, roi_shape, annotation_type=annotation_type)
            new_rois.append(roi)

        return new_rois
    
    def add_roi(
        self,
        coordinates: np.ndarray,
        shape: ROIShape,
        name: Optional[str] = None,
        *,
        annotation_type: str = "custom_annotation",
        metadata: Optional[Dict] = None
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
        
        roi_metadata = metadata.copy() if metadata else {}
        roi_metadata.setdefault("annotation_type", annotation_type)
        roi_metadata.setdefault("source", roi_metadata.get("source", "manual"))
        if self.active_image_label:
            roi_metadata.setdefault("image_label", self.active_image_label)

        roi = ROI(
            id=self.roi_counter,
            name=name,
            shape=shape,
            coordinates=coordinates,
            center=center,
            area=area,
            metadata=roi_metadata,
            annotation_type=annotation_type
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
        
        combined_roi = self.add_roi(
            coordinates,
            ROIShape.POLYGON,
            name,
            annotation_type="combined",
            metadata={"component_roi_ids": roi_ids}
        )
        combined_sources: List[int] = []
        for rid in roi_ids:
            existing = self.get_roi(rid)
            if existing and existing.source_object_ids:
                combined_sources.extend(existing.source_object_ids)
        if combined_sources:
            combined_roi.source_object_ids = combined_sources
        
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

    def get_all_roi_ids(self) -> List[int]:
        return [roi.id for roi in self.rois]

    def get_roi_summary(self, roi_id: int) -> Dict:
        """Return metadata summary for UI."""
        roi = self.get_roi(roi_id)
        if roi is None:
            return {}
        summary = {
            "id": roi.id,
            "name": roi.name,
            "annotation_type": roi.annotation_type,
            "source": roi.metadata.get("source", "manual"),
            "area": float(roi.area),
            "center": tuple(roi.center),
            "has_analysis": roi.analysis_result is not None,
            "analysis_method": roi.analysis_method.value if roi.analysis_method else "",
            "n_curvelets": roi.analysis_result.get("n_curvelets", 0) if roi.analysis_result else 0,
        }
        if roi.analysis_result:
            summary["alignment"] = roi.analysis_result.get("alignment")
            summary["mean_angle"] = roi.analysis_result.get("mean_angle")
        return summary

    def highlight_roi(self, roi_id: int):
        """Highlight ROI inside napari."""
        if self.shapes_layer is None:
            return
        selection = set()
        visible_rois = self.get_rois_for_active_image()
        for idx, roi in enumerate(visible_rois):
            if roi.id == roi_id:
                selection.add(idx)
                break
        self.shapes_layer.selected_data = selection

    def _next_object_id(self) -> int:
        """Return the next unique object identifier."""
        self._object_id_counter += 1
        return self._object_id_counter

    def _register_object(self, obj: AnnotationObject):
        """Register an object and update lookup tables."""
        if obj.kind not in self.objects:
            self.objects[obj.kind] = []
        self.objects[obj.kind].append(obj)
        self.object_lookup[obj.id] = obj

    def clear_objects(self, kinds: Optional[Iterable[str]] = None):
        """Clear tracked objects (cells/fibers)."""
        if kinds is None:
            kinds = list(self.objects.keys())
        kinds = list(kinds)
        for kind in kinds:
            for obj in self.objects.get(kind, []):
                self.object_lookup.pop(obj.id, None)
            self.objects[kind] = []
        self._update_object_layer()

    def _iter_objects(self, kinds: Optional[Sequence[str]] = None) -> Iterable[AnnotationObject]:
        """Iterate over objects filtered by kind."""
        if kinds is None:
            kinds = self._active_object_filter or ("cell", "fiber")
        for kind in kinds:
            for obj in self.objects.get(kind, []):
                yield obj

    def set_object_display_filter(self, kinds: Optional[Sequence[str]]):
        """Set which object types should be visible."""
        if kinds is None or len(kinds) == 0:
            self._active_object_filter = tuple()
        else:
            self._active_object_filter = tuple(kinds)
        self._update_object_layer()

    def _ensure_object_layer(self) -> Optional[Shapes]:
        """Create or retrieve the napari layer used for object visualization."""
        if self.viewer is None:
            return None

        if self.object_layer and self.object_layer in self.viewer.layers:
            return self.object_layer

        for layer in self.viewer.layers:
            if isinstance(layer, Shapes) and layer.name == "Objects":
                self.object_layer = layer
                self.object_layer.interactive = False
                self.object_layer.visible = False
                return self.object_layer

        self.object_layer = self.viewer.add_shapes(
            name="Objects",
            shape_type="path",
            edge_color="yellow",
            face_color="transparent",
            edge_width=1
        )
        self.object_layer.interactive = False
        self.object_layer.visible = False
        return self.object_layer

    def _update_object_layer(self, highlight_ids: Optional[Iterable[int]] = None):
        """Refresh the object overlay layer."""
        layer = self._ensure_object_layer()
        if layer is None:
            return

        kinds = self._active_object_filter or ("cell", "fiber")
        displayed_objects = list(self._iter_objects(kinds))
        if not displayed_objects:
            layer.data = []
            layer.visible = False
            layer.selected_data = set()
            return

        data = [obj.boundary_rc for obj in displayed_objects]
        edge_color = ['#ffd200' if obj.kind == 'cell' else '#ff4fe1' for obj in displayed_objects]
        layer.data = data
        layer.edge_color = edge_color
        layer.face_color = ['transparent'] * len(displayed_objects)
        layer.mode = 'pan_zoom'
        layer.visible = True

        if highlight_ids:
            highlight_ids = set(highlight_ids)
            selected = {idx for idx, obj in enumerate(displayed_objects) if obj.id in highlight_ids}
            layer.selected_data = selected
        else:
            layer.selected_data = set()

    def highlight_objects(self, object_ids: Iterable[int]):
        """Highlight specific objects in the viewer."""
        self._update_object_layer(highlight_ids=object_ids)

    def get_objects(self, kinds: Optional[Sequence[str]] = None) -> List[AnnotationObject]:
        """Return objects filtered by type."""
        return list(self._iter_objects(kinds))

    def get_rois_for_active_image(self) -> List[ROI]:
        """Return ROIs scoped to the active image label (or all if none set)."""
        if not self.active_image_label:
            return list(self.rois)
        return [
            roi for roi in self.rois
            if roi.metadata.get("image_label") == self.active_image_label
            or "image_label" not in roi.metadata
        ]

    def get_object(self, object_id: int) -> Optional[AnnotationObject]:
        """Return object by identifier."""
        return self.object_lookup.get(object_id)

    def register_cell_objects(
        self,
        roi_data_list: List[Dict],
        *,
        replace_existing: bool = True
    ):
        """Register cell objects generated from segmentation data."""
        if replace_existing:
            self.clear_objects(["cell"])

        for entry in roi_data_list:
            coords = np.asarray(entry.get("coordinates", []), dtype=float)
            if coords.size == 0:
                continue
            # masks_to_roi_data stores coordinates as (row, col)
            boundary_rc = coords
            centroid = entry.get("centroid", (0.0, 0.0))
            obj = AnnotationObject(
                id=self._next_object_id(),
                name=entry.get("name", f"cell_{len(self.objects['cell']) + 1}"),
                kind="cell",
                boundary_rc=boundary_rc,
                centroid_rc=(centroid[0], centroid[1]),
                area=float(entry.get("area", 0.0)),
                metadata={"bbox": entry.get("bbox")}
            )
            self._register_object(obj)

        self._update_object_layer()

    def register_fiber_objects(
        self,
        features: Union[pd.DataFrame, List[Dict]],
        *,
        replace_existing: bool = False,
        default_length: float = 5.0
    ):
        """Register fiber objects from CurveAlign features."""
        if features is None:
            return
        if replace_existing:
            self.clear_objects(["fiber"])

        if isinstance(features, pd.DataFrame):
            records = features.to_dict(orient="records")
        else:
            records = features

        for entry in records:
            if not isinstance(entry, dict):
                if hasattr(entry, "_asdict"):
                    entry = entry._asdict()
                else:
                    continue
            center = self._extract_center(entry)
            if center is None:
                continue
            angle = self._extract_orientation(entry)
            if angle is None:
                continue

            boundary_rc = self._fiber_boundary_from_feature(
                center,
                angle,
                length=entry.get("fiber_length", default_length)
            )
            obj = AnnotationObject(
                id=self._next_object_id(),
                name=entry.get("name", f"fiber_{len(self.objects['fiber']) + 1}"),
                kind="fiber",
                boundary_rc=boundary_rc,
                centroid_rc=(center[1], center[0]),
                orientation=angle,
                area=float(entry.get("area", 0.0)),
                metadata={"source": entry}
            )
            self._register_object(obj)

        self._update_object_layer()

    @staticmethod
    def _fiber_boundary_from_feature(center_xy: Tuple[float, float], angle_deg: float, length: float = 5.0) -> np.ndarray:
        """Create a short line segment representing a fiber."""
        angle_rad = np.deg2rad(angle_deg)
        dx = np.cos(angle_rad) * length * 0.5
        dy = np.sin(angle_rad) * length * 0.5
        x, y = center_xy
        # Return as (row, col)
        return np.array(
            [
                [y - dy, x - dx],
                [y + dy, x + dx],
            ],
            dtype=float,
        )

    @staticmethod
    def _extract_center(entry: Dict) -> Optional[Tuple[float, float]]:
        """Extract center coordinate (x, y) from a feature entry."""
        for keys in (("x", "y"), ("col", "row"), ("center_x", "center_y"), ("xc", "yc")):
            if keys[0] in entry and keys[1] in entry:
                return (float(entry[keys[0]]), float(entry[keys[1]]))
        if "center" in entry and len(entry["center"]) == 2:
            return (float(entry["center"][0]), float(entry["center"][1]))
        return None

    @staticmethod
    def _extract_orientation(entry: Dict) -> Optional[float]:
        """Extract orientation in degrees from a feature entry."""
        for key in ("orientation", "angle", "theta"):
            if key in entry:
                return float(entry[key])
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
        
        # Convert ROI to boundary and prepare masks
        boundary = roi.to_boundary(image.shape[:2])
        boundary_data = boundary.data
        base_mask = roi.to_mask(image.shape[:2])
        bbox = (0, 0, image.shape[0], image.shape[1])
        
        # Prepare image
        if roi.crop_mode:
            mask = base_mask
            bbox = self._get_bbox(mask)
            cropped_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if boundary.kind == "polygon" and isinstance(boundary_data, np.ndarray):
                adjusted = boundary_data - np.array([bbox[0], bbox[1]])
                boundary = Boundary(kind="polygon", data=adjusted, spacing_xy=boundary.spacing_xy)
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

        # Attempt to expose curvelet features as fiber objects for annotation workflow
        try:
            if hasattr(result, "features") and result.features is not None:
                self.register_fiber_objects(result.features, replace_existing=False)
        except Exception as exc:
            print(f"Fiber registration skipped: {exc}")
        
        if self.overlay_callback and self.current_image_shape is not None:
            try:
                full_mask = roi.to_mask(self.current_image_shape)
                overlay_payload = {
                    "roi_id": roi.id,
                    "method": method.value,
                    "mask": full_mask,
                    "bbox": bbox if roi.crop_mode else (0, 0, *self.current_image_shape),
                    "result": roi.analysis_result,
                }
                self.overlay_callback(overlay_payload)
            except Exception as exc:
                print(f"Overlay callback failed: {exc}")
        return roi.analysis_result

    def measure_roi(
        self,
        roi_id: int,
        image: np.ndarray,
        histogram_bins: int = 32
    ) -> Optional[Dict[str, Any]]:
        if not HAS_SKIMAGE:
            raise ImportError("scikit-image is required for ROI measurements")
        roi = self.get_roi(roi_id)
        if roi is None:
            return None

        gray = self._ensure_grayscale(image.astype(np.float32))
        gray_norm = gray
        if gray_norm.max() > 1.0:
            gray_norm = gray_norm / np.max(gray_norm)

        mask = roi.to_mask(gray_norm.shape)
        if not np.any(mask):
            return None

        labeled = mask.astype(np.uint8)
        props = regionprops(labeled)
        if not props:
            return None
        prop = props[0]

        values = gray_norm[mask]
        hist, bin_edges = np.histogram(values, bins=histogram_bins, range=(0.0, 1.0))

        metrics = {
            "roi_id": roi_id,
            "area_px": float(prop.area),
            "perimeter_px": float(prop.perimeter),
            "centroid": [float(prop.centroid[1]), float(prop.centroid[0])],
            "bbox": [float(v) for v in prop.bbox],
            "eccentricity": float(prop.eccentricity),
            "orientation_deg": float(np.degrees(prop.orientation)),
            "mean_intensity": float(np.mean(values)),
            "median_intensity": float(np.median(values)),
            "std_intensity": float(np.std(values)),
            "histogram": {
                "bins": bin_edges.tolist(),
                "counts": hist.tolist(),
            },
        }

        roi.metrics = metrics
        return metrics

    def detect_objects_in_roi(
        self,
        roi_id: int,
        object_types: Optional[Sequence[str]] = None,
        distance: Optional[int] = None,
        include_interior: bool = True,
        include_boundary_ring: bool = False,
        boundary_width: int = 5
    ) -> Dict[str, List[AnnotationObject]]:
        """
        Detect registered objects contained within a given ROI.
        
        Parameters
        ----------
        roi_id : int
            Target ROI identifier.
        object_types : Sequence[str], optional
            Object kinds to inspect (defaults to current filter).
        distance : int, optional
            Additional dilation distance (pixels) around ROI boundary.
        """
        roi = self.get_roi(roi_id)
        if roi is None:
            raise ValueError(f"ROI {roi_id} not found")
        if self.current_image_shape is None:
            raise ValueError("Image shape is not set; call set_image_shape first")
        
        mask = roi.to_mask(self.current_image_shape)
        dilation_pixels = distance if distance is not None else self.detection_distance
        if dilation_pixels and dilation_pixels > 0 and HAS_SKIMAGE and binary_dilation is not None:
            mask = binary_dilation(mask, disk(int(dilation_pixels)))
        boundary_mask = None
        if include_boundary_ring and HAS_SKIMAGE and binary_erosion is not None:
            inner = binary_erosion(mask, disk(max(1, boundary_width // 2)))
            outer = binary_dilation(mask, disk(max(1, boundary_width)))
            boundary_mask = np.logical_and(outer, np.logical_not(inner))
        
        kinds = object_types or self._active_object_filter or ("cell", "fiber")
        result: Dict[str, List[AnnotationObject]] = {kind: [] for kind in kinds}
        
        for kind in kinds:
            for obj in self.objects.get(kind, []):
                row = int(round(obj.centroid_rc[0]))
                col = int(round(obj.centroid_rc[1]))
                if row < 0 or col < 0 or row >= mask.shape[0] or col >= mask.shape[1]:
                    continue
                inside = mask[row, col]
                if not include_interior and inside and boundary_mask is None:
                    continue
                if boundary_mask is not None:
                    if not boundary_mask[row, col]:
                        continue
                result[kind].append(obj)
        
        highlight_ids = [obj.id for objs in result.values() for obj in objs]
        if highlight_ids:
            self.highlight_objects(highlight_ids)
        else:
            self.highlight_objects([])
        return result

    def add_annotation_from_object(
        self,
        object_id: int,
        *,
        annotation_type: Optional[str] = None
    ) -> Optional[ROI]:
        """Create an annotation ROI from an existing object."""
        obj = self.get_object(object_id)
        if obj is None:
            return None

        coords = obj.to_roi_coordinates()
        if coords.size == 0:
            return None

        roi = self.add_roi(
            coords,
            ROIShape.POLYGON,
            name=f"{obj.kind}_{object_id}",
            annotation_type=annotation_type or f"{obj.kind}_computed",
            metadata={"source_object_ids": [object_id], "source_kind": obj.kind}
        )
        roi.source_object_ids = [object_id]
        return roi

    def compute_roi_metrics(
        self,
        roi_ids: Sequence[int],
        intensity_image: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if self.current_image_shape is None:
            raise ValueError("Image shape is not set; call set_image_shape first")

        metrics: List[Dict[str, Any]] = []
        if intensity_image is not None and intensity_image.ndim > 2:
            if intensity_image.shape[:2] != self.current_image_shape:
                intensity_image = np.mean(intensity_image, axis=-1)

        for roi_id in roi_ids:
            roi = self.get_roi(roi_id)
            if not roi:
                continue
            mask = roi.to_mask(self.current_image_shape)
            if not np.any(mask):
                continue
            label_image = mask.astype(np.uint8)
            props = regionprops(
                label_image,
                intensity_image=intensity_image if intensity_image is not None else None,
            )
            if not props:
                continue
            prop = props[0]
            stat = {
                "id": roi.id,
                "name": roi.name,
                "type": roi.annotation_type,
                "source": roi.metadata.get("source", ""),
                "area_px": float(prop.area),
                "perimeter_px": float(prop.perimeter),
                "major_axis_px": float(getattr(prop, "major_axis_length", 0.0) or 0.0),
                "minor_axis_px": float(getattr(prop, "minor_axis_length", 0.0) or 0.0),
                "eccentricity": float(getattr(prop, "eccentricity", 0.0) or 0.0),
                "solidity": float(getattr(prop, "solidity", 0.0) or 0.0),
                "orientation_deg": float(
                    np.degrees(getattr(prop, "orientation", 0.0) or 0.0)
                ),
                "centroid_row": float(prop.centroid[0]),
                "centroid_col": float(prop.centroid[1]),
                "bbox_min_row": int(prop.bbox[0]),
                "bbox_min_col": int(prop.bbox[1]),
                "bbox_max_row": int(prop.bbox[2]),
                "bbox_max_col": int(prop.bbox[3]),
            }
            if intensity_image is not None:
                intensities = intensity_image[mask]
                if intensities.size:
                    stat["intensity_mean"] = float(np.mean(intensities))
                    stat["intensity_std"] = float(np.std(intensities))
                    stat["intensity_min"] = float(np.min(intensities))
                    stat["intensity_max"] = float(np.max(intensities))
            metrics.append(stat)
        return metrics

    def get_metrics_dataframe(
        self,
        roi_ids: Optional[Sequence[int]] = None,
        intensity_image: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if roi_ids is None:
            roi_ids = self.get_all_roi_ids()
        data = self.compute_roi_metrics(roi_ids, intensity_image=intensity_image)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    
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
                "Z": roi.metadata.get("z_slice", 0),
                "Annotation": roi.annotation_type
            })
        
        return pd.DataFrame(rows)

    def get_metrics(self, roi_id: int) -> Optional[Dict[str, Any]]:
        roi = self.get_roi(roi_id)
        if roi is None:
            return None
        return roi.metrics if roi.metrics else None
    
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
                    "metadata": roi.metadata,
                    "annotation_type": roi.annotation_type,
                    "source_object_ids": roi.source_object_ids
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
            annotation_type = roi_data.get("annotation_type") or roi_data.get("metadata", {}).get("annotation_type", "custom_annotation")
            roi = self.add_roi(
                coords,
                shape,
                roi_data["name"],
                annotation_type=annotation_type,
                metadata=roi_data.get("metadata")
            )
            if self.active_image_label and "image_label" not in roi.metadata:
                roi.metadata["image_label"] = self.active_image_label
            roi.center = tuple(roi_data["center"])
            roi.area = roi_data["area"]
            roi.analysis_result = roi_data.get("analysis_result")
            if roi_data.get("analysis_method"):
                roi.analysis_method = ROIAnalysisMethod(roi_data["analysis_method"])
            roi.boundary_mode = roi_data.get("boundary_mode")
            roi.crop_mode = roi_data.get("crop_mode", True)
            roi.metadata = roi_data.get("metadata", {}) or {}
            roi.annotation_type = annotation_type
            roi.metadata.setdefault("annotation_type", annotation_type)
            roi.source_object_ids = roi_data.get("source_object_ids", roi.metadata.get("source_object_ids", []))
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
        
        # Add ROIs for active image using proper API
        for roi in self.get_rois_for_active_image():
            if roi.shape == ROIShape.RECTANGLE:
                # Use add_rectangles for rectangles
                coords_rc = self._xy_to_rc(roi.coordinates)
                self.shapes_layer.add_rectangles(
                    [coords_rc],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.ELLIPSE:
                # Use add_ellipses for ellipses
                coords_rc = self._xy_to_rc(roi.coordinates)
                self.shapes_layer.add_ellipses(
                    [coords_rc],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.POLYGON:
                # Use add_polygons for filled polygons
                coords_rc = self._xy_to_rc(roi.coordinates)
                self.shapes_layer.add_polygons(
                    [coords_rc],
                    edge_color="cyan",
                    face_color="transparent"
                )
            elif roi.shape == ROIShape.FREEHAND:
                # Use add_paths for unfilled freehand
                coords_rc = self._xy_to_rc(roi.coordinates)
                self.shapes_layer.add_paths(
                    [coords_rc],
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

