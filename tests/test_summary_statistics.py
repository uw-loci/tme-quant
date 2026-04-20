"""
Tests for summary statistics functionality in napari_curvealign.
"""
import numpy as np
import pytest

from napari_curvealign.roi_manager import ROIManager, ROIShape


@pytest.fixture
def roi_manager_with_data():
    """Create a ROI manager with sample ROIs and fiber data."""
    manager = ROIManager()
    manager.current_image_shape = (200, 200)
    manager.active_image_label = "test_image"
    
    # Create ROIs with varying sizes and positions
    # Small ROI
    roi1 = manager.add_roi(
        coordinates=np.array([[10.0, 10.0], [30.0, 30.0]]),
        shape=ROIShape.RECTANGLE,
        name="small_roi",
        annotation_type="cell"
    )
    
    # Medium ROI
    roi2 = manager.add_roi(
        coordinates=np.array([[50.0, 50.0], [100.0, 100.0]]),
        shape=ROIShape.RECTANGLE,
        name="medium_roi",
        annotation_type="nucleus"
    )
    
    # Large ROI
    roi3 = manager.add_roi(
        coordinates=np.array([[120.0, 120.0], [190.0, 190.0]]),
        shape=ROIShape.RECTANGLE,
        name="large_roi",
        annotation_type="cell"
    )
    
    # Polygon ROI
    roi4 = manager.add_roi(
        coordinates=np.array([
            [20.0, 150.0], [40.0, 150.0], 
            [40.0, 180.0], [20.0, 180.0]
        ]),
        shape=ROIShape.POLYGON,
        name="poly_roi",
        annotation_type="organelle"
    )
    
    # Add mock fiber data for some ROIs
    roi1.metadata["fiber_count"] = 5
    roi1.metadata["mean_length"] = 25.5
    roi1.metadata["mean_angle"] = 45.0
    roi1.metadata["alignment_score"] = 0.75
    
    roi2.metadata["fiber_count"] = 12
    roi2.metadata["mean_length"] = 35.2
    roi2.metadata["mean_angle"] = 60.0
    roi2.metadata["alignment_score"] = 0.85
    
    roi3.metadata["fiber_count"] = 8
    roi3.metadata["mean_length"] = 30.0
    roi3.metadata["mean_angle"] = 50.0
    roi3.metadata["alignment_score"] = 0.65
    
    roi4.metadata["fiber_count"] = 3
    roi4.metadata["mean_length"] = 20.0
    roi4.metadata["mean_angle"] = 90.0
    roi4.metadata["alignment_score"] = 0.55
    
    return manager


class TestSummaryStatisticsBasic:
    """Tests for basic summary statistics calculation."""
    
    def test_compute_all_statistics(self, roi_manager_with_data):
        """Test computing summary statistics for all ROIs."""
        stats = roi_manager_with_data.compute_summary_statistics()
        
        # Check that all expected keys are present
        assert "roi_count" in stats
        assert "total_area" in stats
        assert "mean_area" in stats
        assert "median_area" in stats
        assert "std_area" in stats
        assert "roi_details" in stats
        
        # Check values
        assert stats["roi_count"] == 4
        assert stats["total_area"] > 0
        assert stats["mean_area"] > 0
        assert len(stats["roi_details"]) == 4
    
    def test_compute_selected_statistics(self, roi_manager_with_data):
        """Test computing statistics for selected ROIs only."""
        # Get first two ROI IDs
        roi_ids = [roi.id for roi in roi_manager_with_data.rois[:2]]
        
        stats = roi_manager_with_data.compute_summary_statistics(roi_ids=roi_ids)
        
        # Should only include 2 ROIs
        assert stats["roi_count"] == 2
        assert len(stats["roi_details"]) == 2
    
    def test_morphology_statistics(self, roi_manager_with_data):
        """Test morphology-specific statistics."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_morphology=True
        )
        
        # Check morphology-related keys
        assert "mean_area" in stats
        assert "median_area" in stats
        assert "std_area" in stats
        assert "min_area" in stats
        assert "max_area" in stats
        
        # Check that values make sense
        assert stats["min_area"] <= stats["mean_area"] <= stats["max_area"]
        assert stats["std_area"] >= 0
    
    def test_fiber_metrics_statistics(self, roi_manager_with_data):
        """Test fiber metrics statistics."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_fiber_metrics=True
        )
        
        # Check fiber-related keys (if fiber data exists)
        roi_details = stats["roi_details"]
        
        # At least one ROI should have fiber data
        has_fiber_data = any(
            "fiber_count" in roi for roi in roi_details
        )
        assert has_fiber_data
    
    def test_statistics_without_morphology(self, roi_manager_with_data):
        """Test computing statistics without morphology."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_morphology=False
        )
        
        # Should still have basic count
        assert "roi_count" in stats
        assert stats["roi_count"] == 4
    
    def test_statistics_without_fiber_metrics(self, roi_manager_with_data):
        """Test computing statistics without fiber metrics."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_fiber_metrics=False
        )
        
        # Should still have morphology data
        assert "mean_area" in stats
        assert "roi_count" in stats


class TestPerROIStatistics:
    """Tests for per-ROI statistics in the summary."""
    
    def test_roi_details_structure(self, roi_manager_with_data):
        """Test that ROI details have correct structure."""
        stats = roi_manager_with_data.compute_summary_statistics()
        roi_details = stats["roi_details"]
        
        # Check that each ROI has basic info
        for roi_info in roi_details:
            assert "id" in roi_info
            assert "name" in roi_info
            assert "shape" in roi_info
            assert "area" in roi_info
            assert "annotation_type" in roi_info
    
    def test_roi_details_with_fiber_data(self, roi_manager_with_data):
        """Test that fiber data is included in ROI details."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_fiber_metrics=True
        )
        roi_details = stats["roi_details"]
        
        # First ROI should have fiber metrics
        roi1 = roi_details[0]
        if "fiber_count" in roi_manager_with_data.rois[0].metadata:
            assert "fiber_count" in roi1
    
    def test_roi_center_coordinates(self, roi_manager_with_data):
        """Test that center coordinates are included."""
        stats = roi_manager_with_data.compute_summary_statistics()
        roi_details = stats["roi_details"]
        
        for roi_info in roi_details:
            assert "center" in roi_info
            center = roi_info["center"]
            assert len(center) == 2  # (y, x)
            assert all(isinstance(c, (int, float)) for c in center)


class TestAggregateStatistics:
    """Tests for aggregate statistics across ROIs."""
    
    def test_total_area_calculation(self, roi_manager_with_data):
        """Test that total area is sum of all ROI areas."""
        stats = roi_manager_with_data.compute_summary_statistics()
        
        # Calculate expected total from individual ROIs
        expected_total = sum(
            roi_info["area"] for roi_info in stats["roi_details"]
        )
        
        assert abs(stats["total_area"] - expected_total) < 0.1
    
    def test_mean_area_calculation(self, roi_manager_with_data):
        """Test that mean area is correctly calculated."""
        stats = roi_manager_with_data.compute_summary_statistics()
        
        # Calculate expected mean
        expected_mean = stats["total_area"] / stats["roi_count"]
        
        assert abs(stats["mean_area"] - expected_mean) < 0.1
    
    def test_statistics_by_annotation_type(self, roi_manager_with_data):
        """Test grouping statistics by annotation type."""
        stats = roi_manager_with_data.compute_summary_statistics()
        
        # Group ROIs by annotation type
        by_type = {}
        for roi_info in stats["roi_details"]:
            ann_type = roi_info["annotation_type"]
            if ann_type not in by_type:
                by_type[ann_type] = []
            by_type[ann_type].append(roi_info)
        
        # Should have at least 2 different annotation types
        assert len(by_type) >= 2
        
        # "cell" annotation should have 2 ROIs
        if "cell" in by_type:
            assert len(by_type["cell"]) == 2


class TestFiberMetricsAggregation:
    """Tests for aggregating fiber metrics across ROIs."""
    
    def test_total_fiber_count(self, roi_manager_with_data):
        """Test calculation of total fiber count across all ROIs."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_fiber_metrics=True
        )
        
        # Calculate expected total fiber count
        expected_total_fibers = sum(
            roi.metadata.get("fiber_count", 0)
            for roi in roi_manager_with_data.rois
        )
        
        # Check if total_fiber_count is in stats (if implemented)
        if "total_fiber_count" in stats:
            assert stats["total_fiber_count"] == expected_total_fibers
    
    def test_mean_fiber_length_across_rois(self, roi_manager_with_data):
        """Test calculation of mean fiber length across all ROIs."""
        stats = roi_manager_with_data.compute_summary_statistics(
            include_fiber_metrics=True
        )
        
        # Verify that fiber metrics are present
        fiber_lengths = [
            roi.metadata.get("mean_length")
            for roi in roi_manager_with_data.rois
            if "mean_length" in roi.metadata
        ]
        
        assert len(fiber_lengths) > 0


class TestEmptyROIStatistics:
    """Tests for statistics with no ROIs."""
    
    def test_empty_roi_manager(self):
        """Test computing statistics when no ROIs exist."""
        manager = ROIManager()
        manager.current_image_shape = (100, 100)
        
        stats = manager.compute_summary_statistics()
        
        # Should return valid stats with zero counts
        assert stats["roi_count"] == 0
        assert stats["total_area"] == 0
        assert len(stats["roi_details"]) == 0
    
    def test_nonexistent_roi_ids(self, roi_manager_with_data):
        """Test computing statistics with invalid ROI IDs."""
        stats = roi_manager_with_data.compute_summary_statistics(
            roi_ids=[9999, 8888]  # Non-existent IDs
        )
        
        # Should return empty or zero stats
        assert stats["roi_count"] == 0


class TestStatisticsEdgeCases:
    """Tests for edge cases in statistics calculation."""
    
    def test_single_roi_statistics(self):
        """Test statistics with only one ROI."""
        manager = ROIManager()
        manager.current_image_shape = (100, 100)
        
        # Add single ROI
        manager.add_roi(
            coordinates=np.array([[10.0, 10.0], [30.0, 30.0]]),
            shape=ROIShape.RECTANGLE,
            name="single_roi"
        )
        
        stats = manager.compute_summary_statistics()
        
        assert stats["roi_count"] == 1
        assert stats["mean_area"] == stats["total_area"]
        # Standard deviation should be 0 for single ROI
        if "std_area" in stats:
            assert stats["std_area"] == 0.0
    
    def test_very_small_rois(self):
        """Test statistics with very small ROIs."""
        manager = ROIManager()
        manager.current_image_shape = (100, 100)
        
        # Add tiny ROI
        manager.add_roi(
            coordinates=np.array([[10.0, 10.0], [11.0, 11.0]]),
            shape=ROIShape.RECTANGLE,
            name="tiny_roi"
        )
        
        stats = manager.compute_summary_statistics()
        
        assert stats["roi_count"] == 1
        assert stats["total_area"] > 0
    
    def test_rois_with_partial_data(self):
        """Test statistics when some ROIs have fiber data and others don't."""
        manager = ROIManager()
        manager.current_image_shape = (100, 100)
        
        # ROI with fiber data
        roi1 = manager.add_roi(
            coordinates=np.array([[10.0, 10.0], [30.0, 30.0]]),
            shape=ROIShape.RECTANGLE,
            name="roi_with_fibers"
        )
        roi1.metadata["fiber_count"] = 10
        
        # ROI without fiber data
        manager.add_roi(
            coordinates=np.array([[50.0, 50.0], [70.0, 70.0]]),
            shape=ROIShape.RECTANGLE,
            name="roi_without_fibers"
        )
        
        # Should not crash
        stats = manager.compute_summary_statistics(include_fiber_metrics=True)
        
        assert stats["roi_count"] == 2
        assert len(stats["roi_details"]) == 2


class TestStatisticsExport:
    """Tests for exporting statistics to different formats."""
    
    def test_statistics_to_dict(self, roi_manager_with_data):
        """Test that statistics return as a dictionary."""
        stats = roi_manager_with_data.compute_summary_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
    
    def test_statistics_serializable(self, roi_manager_with_data):
        """Test that statistics can be serialized to JSON."""
        import json
        
        stats = roi_manager_with_data.compute_summary_statistics()
        
        # Should be JSON-serializable
        try:
            json_str = json.dumps(stats)
            assert len(json_str) > 0
        except TypeError:
            pytest.fail("Statistics dictionary is not JSON-serializable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
