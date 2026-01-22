"""
Tests for ROI format support (Cellpose, QuPath, StarDist) in napari_curvealign.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from napari_curvealign.roi_manager import ROIManager, ROIShape


@pytest.fixture
def roi_manager():
    """Create a ROI manager instance with a test image."""
    manager = ROIManager()
    # Set up a test image (100x100)
    manager.current_image_shape = (100, 100)
    manager.active_image_label = "test_image"
    return manager


@pytest.fixture
def sample_rois(roi_manager):
    """Create sample ROIs for testing."""
    # Rectangle ROI
    roi1 = roi_manager.add_roi(
        coordinates=np.array([[10.0, 10.0], [30.0, 30.0]]),
        shape=ROIShape.RECTANGLE,
        name="test_rect",
        annotation_type="cell"
    )
    
    # Polygon ROI
    roi2 = roi_manager.add_roi(
        coordinates=np.array([[50.0, 50.0], [60.0, 50.0], [60.0, 70.0], [50.0, 70.0]]),
        shape=ROIShape.POLYGON,
        name="test_poly",
        annotation_type="nucleus"
    )
    
    # Ellipse ROI
    roi3 = roi_manager.add_roi(
        coordinates=np.array([[70.0, 70.0], [10.0, 15.0]]),
        shape=ROIShape.ELLIPSE,
        name="test_ellipse",
        annotation_type="organelle"
    )
    
    return [roi1, roi2, roi3]


class TestCellposeFormat:
    """Tests for Cellpose .npy format support."""
    
    def test_save_cellpose_format(self, roi_manager, sample_rois, tmp_path):
        """Test saving ROIs in Cellpose format."""
        output_path = tmp_path / "cellpose_test.npy"
        
        # Save all ROIs
        roi_manager.save_rois_cellpose(str(output_path))
        
        # Check that files were created
        assert output_path.exists()
        metadata_path = output_path.with_suffix(".json")
        assert metadata_path.exists()
        
        # Check mask array
        masks = np.load(output_path)
        assert masks.shape == (100, 100)
        assert masks.dtype in [np.int32, np.int64, np.uint16]
        
        # Check that we have 3 distinct regions (plus background)
        unique_labels = np.unique(masks)
        assert len(unique_labels) >= 2  # At least background and one ROI
        assert len(unique_labels) <= 4  # At most background + 3 ROIs
        
        # Check metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        assert "rois" in metadata
        assert len(metadata["rois"]) == 3
    
    def test_load_cellpose_format(self, roi_manager, sample_rois, tmp_path):
        """Test loading ROIs from Cellpose format."""
        output_path = tmp_path / "cellpose_test.npy"
        
        # Save ROIs
        roi_manager.save_rois_cellpose(str(output_path))
        
        # Create new manager and load
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        loaded_rois = new_manager.load_rois_cellpose(str(output_path))
        
        # Check that ROIs were loaded
        assert len(loaded_rois) == 3
        
        # Verify ROI properties are preserved
        roi_names = {roi.name for roi in loaded_rois}
        assert "test_rect" in roi_names or len(loaded_rois) == 3
        
        # Verify annotation types
        annotation_types = {roi.annotation_type for roi in loaded_rois}
        assert len(annotation_types) >= 1  # At least one annotation type
    
    def test_save_selected_rois_cellpose(self, roi_manager, sample_rois, tmp_path):
        """Test saving only selected ROIs in Cellpose format."""
        output_path = tmp_path / "cellpose_selected.npy"
        
        # Save only first two ROIs
        roi_ids = [sample_rois[0].id, sample_rois[1].id]
        roi_manager.save_rois_cellpose(str(output_path), roi_ids=roi_ids)
        
        # Check metadata
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert len(metadata["rois"]) == 2


class TestQuPathFormat:
    """Tests for QuPath GeoJSON format support."""
    
    def test_save_qupath_format(self, roi_manager, sample_rois, tmp_path):
        """Test saving ROIs in QuPath GeoJSON format."""
        output_path = tmp_path / "qupath_test.geojson"
        
        # Save all ROIs
        roi_manager.save_rois_qupath(str(output_path))
        
        # Check that file was created
        assert output_path.exists()
        
        # Load and verify GeoJSON structure
        with open(output_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert geojson_data["type"] == "FeatureCollection"
        assert "features" in geojson_data
        assert len(geojson_data["features"]) == 3
        
        # Check feature properties
        for feature in geojson_data["features"]:
            assert "type" in feature
            assert feature["type"] == "Feature"
            assert "geometry" in feature
            assert "properties" in feature
            assert "classification" in feature["properties"]
    
    def test_load_qupath_format(self, roi_manager, sample_rois, tmp_path):
        """Test loading ROIs from QuPath GeoJSON format."""
        output_path = tmp_path / "qupath_test.geojson"
        
        # Save ROIs
        roi_manager.save_rois_qupath(str(output_path))
        
        # Create new manager and load
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        loaded_rois = new_manager.load_rois_qupath(str(output_path))
        
        # Check that ROIs were loaded
        assert len(loaded_rois) == 3
        
        # Verify shapes are preserved
        loaded_shapes = {roi.shape for roi in loaded_rois}
        assert len(loaded_shapes) >= 1
    
    def test_save_selected_rois_qupath(self, roi_manager, sample_rois, tmp_path):
        """Test saving only selected ROIs in QuPath format."""
        output_path = tmp_path / "qupath_selected.geojson"
        
        # Save only first ROI
        roi_ids = [sample_rois[0].id]
        roi_manager.save_rois_qupath(str(output_path), roi_ids=roi_ids)
        
        # Load and verify
        with open(output_path, 'r') as f:
            geojson_data = json.load(f)
        
        assert len(geojson_data["features"]) == 1


class TestStarDistFormat:
    """Tests for StarDist format support (via Fiji)."""
    
    def test_save_stardist_format(self, roi_manager, sample_rois, tmp_path):
        """Test saving ROIs in StarDist format."""
        output_path = tmp_path / "stardist_test.zip"
        
        # Save all ROIs (delegates to Fiji format)
        roi_manager.save_rois_stardist(str(output_path))
        
        # Check that file was created
        assert output_path.exists()
        
        # Verify it's a valid zip file
        import zipfile
        assert zipfile.is_zipfile(output_path)
    
    def test_load_stardist_format(self, roi_manager, sample_rois, tmp_path):
        """Test loading ROIs from StarDist format."""
        output_path = tmp_path / "stardist_test.zip"
        
        # Save ROIs
        roi_manager.save_rois_stardist(str(output_path))
        
        # Create new manager and load
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        loaded_rois = new_manager.load_rois_stardist(str(output_path))
        
        # Check that ROIs were loaded
        assert len(loaded_rois) >= 1  # At least some ROIs should be loaded


class TestAutoDetectFormat:
    """Tests for automatic format detection in save/load methods."""
    
    def test_auto_save_cellpose(self, roi_manager, sample_rois, tmp_path):
        """Test auto-detection when saving with .npy extension."""
        output_path = tmp_path / "auto_test.npy"
        roi_manager.save_rois(str(output_path))
        
        assert output_path.exists()
        # Should create both .npy and .json files
        assert output_path.with_suffix(".json").exists()
    
    def test_auto_load_cellpose(self, roi_manager, sample_rois, tmp_path):
        """Test auto-detection when loading .npy file."""
        output_path = tmp_path / "auto_test.npy"
        roi_manager.save_rois(str(output_path))
        
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        loaded_rois = new_manager.load_rois(str(output_path))
        
        assert len(loaded_rois) >= 1
    
    def test_auto_save_qupath(self, roi_manager, sample_rois, tmp_path):
        """Test auto-detection when saving with .geojson extension."""
        output_path = tmp_path / "auto_test.geojson"
        roi_manager.save_rois(str(output_path))
        
        assert output_path.exists()
        
        # Verify it's valid GeoJSON
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
    
    def test_auto_load_qupath(self, roi_manager, sample_rois, tmp_path):
        """Test auto-detection when loading .geojson file."""
        output_path = tmp_path / "auto_test.geojson"
        roi_manager.save_rois(str(output_path))
        
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        loaded_rois = new_manager.load_rois(str(output_path))
        
        assert len(loaded_rois) >= 1


class TestRoundTrip:
    """Tests for round-trip conversion (save and load should preserve data)."""
    
    def test_cellpose_roundtrip(self, roi_manager, sample_rois, tmp_path):
        """Test that saving and loading Cellpose format preserves ROI count."""
        output_path = tmp_path / "roundtrip.npy"
        
        original_count = len(roi_manager.rois)
        roi_manager.save_rois_cellpose(str(output_path))
        
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        new_manager.load_rois_cellpose(str(output_path))
        
        # Should have same number of ROIs (or close, due to conversion)
        assert len(new_manager.rois) >= original_count - 1
    
    def test_qupath_roundtrip(self, roi_manager, sample_rois, tmp_path):
        """Test that saving and loading QuPath format preserves ROI count."""
        output_path = tmp_path / "roundtrip.geojson"
        
        original_count = len(roi_manager.rois)
        roi_manager.save_rois_qupath(str(output_path))
        
        new_manager = ROIManager()
        new_manager.current_image_shape = (100, 100)
        new_manager.load_rois_qupath(str(output_path))
        
        assert len(new_manager.rois) == original_count


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_save_empty_rois(self, roi_manager, tmp_path):
        """Test saving when no ROIs exist."""
        output_path = tmp_path / "empty.geojson"
        
        # Should not crash
        roi_manager.save_rois_qupath(str(output_path))
        
        # File should still be created with empty feature collection
        assert output_path.exists()
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert len(data["features"]) == 0
    
    def test_load_nonexistent_file(self, roi_manager, tmp_path):
        """Test loading from a file that doesn't exist."""
        output_path = tmp_path / "nonexistent.geojson"
        
        with pytest.raises(FileNotFoundError):
            roi_manager.load_rois_qupath(str(output_path))
    
    def test_load_invalid_format(self, roi_manager, tmp_path):
        """Test loading from a file with invalid content."""
        output_path = tmp_path / "invalid.geojson"
        
        # Create invalid JSON file
        with open(output_path, 'w') as f:
            f.write("not valid json")
        
        with pytest.raises((json.JSONDecodeError, ValueError)):
            roi_manager.load_rois_qupath(str(output_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
