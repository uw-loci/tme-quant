"""
Adapters to convert between existing formats and hierarchy models.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import existing bioimage core
try:
    from bioimage_core.image_processing.segmentation import SegmentationResult
    from bioimage_core.image_processing.filters import FilterResult
    from bioimage_core.visualization.plot_utils import PlotData
    EXISTING_IMPORTS_AVAILABLE = True
except ImportError:
    EXISTING_IMPORTS_AVAILABLE = False
    # Define mock classes for development
    class SegmentationResult: pass
    class FilterResult: pass
    class PlotData: pass

# Import hierarchy module
from qupath_hierarchy import (
    QuPathLikeProject,
    PathObject,
    ImageObject,
    Cell,
    TissueRegion,
    TumorRegion,
    ROI,
    ImageEntry,
    ObjectType,
    RoiType,
)


@dataclass
class ImageProcessingAdapter:
    """
    Adapter between existing image processing and hierarchy.
    """
    
    def filter_to_hierarchy(
        self,
        filter_result: FilterResult,
        image_object: ImageObject,
        filter_name: str
    ) -> Dict[str, Any]:
        """
        Convert filter results to hierarchy measurements.
        
        Args:
            filter_result: Existing filter result
            image_object: Target image object
            filter_name: Name of the filter applied
            
        Returns:
            Dictionary of filter measurements
        """
        measurements = {}
        
        if hasattr(filter_result, 'get_measurements'):
            existing_measurements = filter_result.get_measurements()
            
            for obj_id, obj_measurements in existing_measurements.items():
                # Find corresponding hierarchy object
                hierarchy_obj = self._find_object_by_metadata(
                    image_object, {'existing_id': obj_id}
                )
                
                if hierarchy_obj:
                    for key, value in obj_measurements.items():
                        hierarchy_obj.add_measurement(
                            f"{filter_name}_{key}", value
                        )
                    
                    measurements[hierarchy_obj.id] = obj_measurements
        
        return measurements
    
    def _find_object_by_metadata(
        self,
        parent: PathObject,
        metadata_filter: Dict[str, Any]
    ) -> Optional[PathObject]:
        """Find object by metadata."""
        for obj in parent.get_children(recursive=True):
            if all(
                obj.metadata.get(key) == value
                for key, value in metadata_filter.items()
            ):
                return obj
        return None


@dataclass
class SegmentationAdapter:
    """
    Adapter between existing segmentation and hierarchy objects.
    """
    
    def segmentation_to_objects(
        self,
        segmentation_result: SegmentationResult,
        parent_object: Optional[PathObject] = None,
        object_type: ObjectType = ObjectType.CELL,
        classification: Optional[str] = None
    ) -> List[PathObject]:
        """
        Convert segmentation result to hierarchy objects.
        
        Args:
            segmentation_result: Existing segmentation result
            parent_object: Parent object in hierarchy
            object_type: Type of objects to create
            classification: Optional classification label
            
        Returns:
            List of created hierarchy objects
        """
        objects = []
        
        if not EXISTING_IMPORTS_AVAILABLE:
            return objects
        
        # Get masks and properties from segmentation
        masks = getattr(segmentation_result, 'masks', [])
        properties = getattr(segmentation_result, 'properties', [])
        
        for i, (mask, props) in enumerate(zip(masks, properties)):
            # Create ROI from mask
            roi = self._mask_to_roi(mask, f"object_{i}")
            
            # Create appropriate object type
            obj = self._create_object_by_type(
                object_type,
                name=f"{object_type.value}_{i:03d}",
                classification=classification
            )
            
            # Add ROI
            obj.rois.append(roi)
            
            # Set parent
            if parent_object:
                obj.parent = parent_object
            
            # Add measurements from segmentation properties
            for prop_name, prop_value in props.items():
                if isinstance(prop_value, (int, float, str)):
                    obj.add_measurement(prop_name, prop_value)
            
            # Store segmentation metadata
            obj.metadata['segmentation_source'] = 'existing_pipeline'
            obj.metadata['segmentation_id'] = i
            
            objects.append(obj)
        
        return objects
    
    def _mask_to_roi(self, mask: np.ndarray, name: str) -> ROI:
        """Convert binary mask to ROI."""
        # This is a simplified implementation
        # In practice, you'd convert mask to polygon
        from qupath_hierarchy import RoiType
        
        return ROI(
            type=RoiType.MASK,
            coordinates=mask,
            name=name
        )
    
    def _create_object_by_type(
        self,
        object_type: ObjectType,
        name: str,
        classification: Optional[str] = None
    ) -> PathObject:
        """Create object of specified type."""
        from qupath_hierarchy import (
            Cell, TissueRegion, TumorRegion, Nucleus, Fiber
        )
        
        type_map = {
            ObjectType.CELL: Cell,
            ObjectType.TISSUE_REGION: TissueRegion,
            ObjectType.TUMOR_REGION: TumorRegion,
            ObjectType.NUCLEUS: Nucleus,
            ObjectType.FIBER: Fiber,
        }
        
        obj_class = type_map.get(object_type, PathObject)
        return obj_class(name=name, classification=classification)


@dataclass
class MeasurementAdapter:
    """
    Adapter for measurement conversions.
    """
    
    def existing_to_hierarchy_measurements(
        self,
        existing_measurements: Dict[str, Any],
        target_object: PathObject,
        measurement_prefix: str = "existing"
    ) -> None:
        """
        Convert existing measurements to hierarchy format.
        
        Args:
            existing_measurements: Existing measurement dictionary
            target_object: Target hierarchy object
            measurement_prefix: Prefix for measurement names
        """
        for key, value in existing_measurements.items():
            if isinstance(value, dict):
                # Nested measurements
                for subkey, subvalue in value.items():
                    target_object.add_measurement(
                        f"{measurement_prefix}_{key}_{subkey}",
                        subvalue
                    )
            else:
                # Simple measurements
                target_object.add_measurement(
                    f"{measurement_prefix}_{key}",
                    value
                )
    
    def hierarchy_to_existing_format(
        self,
        hierarchy_object: PathObject,
        flatten: bool = True
    ) -> Dict[str, Any]:
        """
        Convert hierarchy measurements to existing format.
        
        Args:
            hierarchy_object: Source hierarchy object
            flatten: Whether to flatten nested measurements
            
        Returns:
            Measurements in existing format
        """
        if flatten:
            # Flatten all measurements
            measurements = {}
            for mtype, mdict in hierarchy_object.measurements.items():
                for key, value in mdict.items():
                    measurements[f"{mtype}_{key}"] = value
            return measurements
        else:
            # Keep structure
            return hierarchy_object.measurements.copy()


@dataclass
class ProjectAdapter:
    """
    Adapter for project-level conversions.
    """
    
    def create_project_from_existing_analysis(
        self,
        analysis_results: Dict[str, Any],
        image_paths: List[Union[str, Path]],
        project_name: Optional[str] = None
    ) -> QuPathLikeProject:
        """
        Create hierarchy project from existing analysis results.
        
        Args:
            analysis_results: Existing analysis results
            image_paths: List of image paths
            project_name: Name for the new project
            
        Returns:
            QuPathLikeProject
        """
        from qupath_hierarchy import QuPathLikeProject, ImageEntry
        
        project_name = project_name or f"Converted_{len(image_paths)}_images"
        project = QuPathLikeProject(name=project_name)
        
        for img_idx, img_path in enumerate(image_paths):
            # Add image to project
            img_path = Path(img_path)
            image_entry = ImageEntry(
                file_path=img_path,
                name=img_path.stem,
                metadata={'original_index': img_idx}
            )
            image_obj = project.add_image(image_entry)
            
            # Convert segmentations if available
            seg_key = f"segmentation_{img_idx}"
            if seg_key in analysis_results:
                seg_result = analysis_results[seg_key]
                
                # Use segmentation adapter
                seg_adapter = SegmentationAdapter()
                objects = seg_adapter.segmentation_to_objects(
                    seg_result, image_obj
                )
                
                for obj in objects:
                    project.add_object(obj, parent=image_obj)
            
            # Convert measurements if available
            meas_key = f"measurements_{img_idx}"
            if meas_key in analysis_results:
                meas_adapter = MeasurementAdapter()
                for obj in image_obj.get_children():
                    meas_adapter.existing_to_hierarchy_measurements(
                        analysis_results[meas_key].get(obj.id, {}),
                        obj
                    )
        
        # Add analysis metadata
        project.metadata['conversion_source'] = 'existing_analysis'
        project.metadata['original_analysis'] = {
            k: str(type(v)) for k, v in analysis_results.items()
        }
        
        return project
    
    def export_project_to_existing_format(
        self,
        project: QuPathLikeProject,
        format: str = 'legacy'
    ) -> Dict[str, Any]:
        """
        Export hierarchy project to existing format.
        
        Args:
            project: Hierarchy project
            format: Target format ('legacy', 'csv', 'json')
            
        Returns:
            Data in existing format
        """
        output = {
            'project_name': project.name,
            'images': [],
            'objects': [],
            'measurements': {}
        }
        
        # Export images
        for img_id, img_obj in project.images.items():
            output['images'].append({
                'id': img_id,
                'name': img_obj.name,
                'path': str(img_obj.image_entry.file_path) if img_obj.image_entry.file_path else None,
                'metadata': img_obj.metadata
            })
        
        # Export objects with hierarchy
        for obj_id, obj in project.objects.items():
            obj_data = {
                'id': obj_id,
                'name': obj.name,
                'type': obj.object_type.value,
                'classification': obj.classification,
                'parent_id': obj.parent.id if obj.parent else None,
                'children_ids': [c.id for c in obj.get_children()],
                'rois': [roi.to_dict() for roi in obj.rois],
                'measurements': obj.measurements,
                'metadata': obj.metadata
            }
            output['objects'].append(obj_data)
        
        # Export measurements in flat format
        meas_adapter = MeasurementAdapter()
        for obj in project.objects.values():
            output['measurements'][obj.id] = meas_adapter.hierarchy_to_existing_format(
                obj, flatten=True
            )
        
        return output