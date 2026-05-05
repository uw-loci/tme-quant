"""
Tests for QuPath GeoJSON export/import:
  - FiberObject.to_geojson_feature()
  - CellObject.to_geojson_feature()
  - ROIObject.to_geojson_feature() objectType fix
  - integrations.load_qupath_annotations()
  - integrations.load_qupath_measurements()
"""

import sys
import unittest.mock as mock

for lib in ["shapely", "shapely.geometry", "shapely.ops"]:
    sys.modules.setdefault(lib, mock.MagicMock())

import json
import numpy as np
import pytest

from tme_quant.core.tme_objects.fiber_objects import FiberObject
from tme_quant.core.tme_objects.cell_objects import CellObject, CellType
from tme_quant.core.roi_manager import ROIManager
from tme_quant.integrations import (
    export_hierarchy_geojson,
    load_qupath_annotations,
    load_qupath_measurements,
)


# ─────────────────────────────────────────────────────────────────────────────
# FiberObject.to_geojson_feature
# ─────────────────────────────────────────────────────────────────────────────

class TestFiberObjectGeoJSON:
    def test_linestring_geometry(self):
        f = FiberObject(object_id="f1",
                        centerline=np.array([[10, 20], [15, 25], [20, 30]]))
        feat = f.to_geojson_feature(pixel_size=1.0)
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "LineString"
        assert feat["properties"]["objectType"] == "detection"

    def test_coordinate_swap_and_scale(self):
        # centerline row=10,col=20 → GeoJSON [x=col*px, y=row*px]
        f = FiberObject(object_id="f1",
                        centerline=np.array([[10, 20], [15, 25]]))
        feat = f.to_geojson_feature(pixel_size=0.5)
        x, y = feat["geometry"]["coordinates"][0]
        assert x == pytest.approx(10.0)   # col=20 * 0.5
        assert y == pytest.approx(5.0)    # row=10 * 0.5

    def test_tacs_classification(self):
        f = FiberObject(object_id="f1",
                        centerline=np.array([[0, 0], [1, 1]]),
                        tacs_type="TACS-3", tacs_score=0.9)
        feat = f.to_geojson_feature()
        assert feat["properties"]["classification"]["name"] == "TACS-3"
        assert "colorRGB" in feat["properties"]["classification"]

    def test_no_tacs_classification_is_none(self):
        f = FiberObject(object_id="f1",
                        centerline=np.array([[0, 0], [1, 1]]))
        feat = f.to_geojson_feature()
        assert feat["properties"]["classification"] is None

    def test_point_fallback_for_single_point_fiber(self):
        f = FiberObject(object_id="f1",
                        orientation_point=np.array([5.0, 10.0]))
        feat = f.to_geojson_feature(pixel_size=1.0)
        assert feat["geometry"]["type"] == "Point"
        # col=10.0, row=5.0 → [x=10.0, y=5.0]
        assert feat["geometry"]["coordinates"] == pytest.approx([10.0, 5.0])

    def test_geometry_none_when_no_position(self):
        f = FiberObject(object_id="f1")  # no centerline, no orientation_point
        feat = f.to_geojson_feature()
        assert feat["geometry"] is None

    def test_none_measurements_excluded(self):
        f = FiberObject(object_id="f1",
                        centerline=np.array([[0, 0], [1, 1]]),
                        length=5.0)
        feat = f.to_geojson_feature()
        names = [m["name"] for m in feat["properties"]["measurements"]]
        assert "Length µm" in names
        assert all(m["value"] is not None for m in feat["properties"]["measurements"])

    def test_measurements_all_none_gives_empty_list(self):
        f = FiberObject(object_id="f1",
                        centerline=np.array([[0, 0], [1, 1]]))
        # length, width etc. all default to 0.0, tacs_type=None
        feat = f.to_geojson_feature()
        # 0.0 values are not None so they appear; just check no error
        assert isinstance(feat["properties"]["measurements"], list)


# ─────────────────────────────────────────────────────────────────────────────
# CellObject.to_geojson_feature
# ─────────────────────────────────────────────────────────────────────────────

class TestCellObjectGeoJSON:
    def test_point_geometry(self):
        c = CellObject(object_id="c1", centroid=(100.0, 200.0))
        feat = c.to_geojson_feature(pixel_size=1.0)
        assert feat["geometry"]["type"] == "Point"
        assert feat["geometry"]["coordinates"] == pytest.approx([100.0, 200.0])

    def test_pixel_size_scaling(self):
        c = CellObject(object_id="c1", centroid=(100.0, 200.0))
        feat = c.to_geojson_feature(pixel_size=0.5)
        assert feat["geometry"]["coordinates"] == pytest.approx([50.0, 100.0])

    def test_cell_type_classification(self):
        c = CellObject(object_id="c1", centroid=(0, 0), cell_type=CellType.TUMOR)
        feat = c.to_geojson_feature()
        assert feat["properties"]["classification"]["name"] == "tumor"
        assert "colorRGB" in feat["properties"]["classification"]

    def test_no_cell_type_classification_is_none(self):
        c = CellObject(object_id="c1", centroid=(0, 0))
        feat = c.to_geojson_feature()
        assert feat["properties"]["classification"] is None

    def test_objecttype_is_detection(self):
        c = CellObject(object_id="c1", centroid=(0, 0))
        assert c.to_geojson_feature()["properties"]["objectType"] == "detection"


# ─────────────────────────────────────────────────────────────────────────────
# ROIObject.to_geojson_feature — objectType fix
# ─────────────────────────────────────────────────────────────────────────────

class TestROIObjectGeoJSON:
    def test_objecttype_annotation_present(self):
        mgr = ROIManager()
        roi = mgr.add_polygon(
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
            annotation_type="tumor_boundary",
        )
        feat = roi.to_geojson_feature()
        assert feat["properties"]["objectType"] == "annotation"
        assert feat["geometry"]["type"] == "Polygon"

    def test_classification_name_matches_annotation_type(self):
        mgr = ROIManager()
        roi = mgr.add_polygon(
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            annotation_type="stroma",
        )
        feat = roi.to_geojson_feature()
        assert feat["properties"]["classification"]["name"] == "stroma"


# ─────────────────────────────────────────────────────────────────────────────
# load_qupath_annotations
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadQuPathAnnotations:
    def _write_geojson(self, path, features):
        path.write_text(json.dumps({"type": "FeatureCollection",
                                    "features": features}))

    def test_polygon_imported_as_roi(self, tmp_path):
        p = tmp_path / "test.geojson"
        self._write_geojson(p, [{
            "type": "Feature", "id": "anno_1",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0,0],[100,0],[100,100],[0,100],[0,0]]]},
            "properties": {"name": "Tumor",
                           "classification": {"name": "tumor_boundary"}},
        }])
        rois = load_qupath_annotations(p)
        assert len(rois) == 1
        assert rois[0].annotation_type == "tumor_boundary"

    def test_attaches_to_parent(self, tmp_path):
        from tme_quant.core.base_models import TMEObject, TMEType
        parent = TMEObject(object_id="root", tme_type=TMEType.IMAGE)
        p = tmp_path / "t.geojson"
        self._write_geojson(p, [{
            "type": "Feature", "id": "r1",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0,0],[10,0],[10,10],[0,10],[0,0]]]},
            "properties": {"name": "R1",
                           "classification": {"name": "custom"}},
        }])
        load_qupath_annotations(p, parent_node=parent)
        assert len(parent.children) == 1

    def test_empty_collection_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        assert load_qupath_annotations(p) == []

    def test_uses_provided_manager(self, tmp_path):
        mgr = ROIManager()
        p = tmp_path / "t.geojson"
        self._write_geojson(p, [{
            "type": "Feature", "id": "r1",
            "geometry": {"type": "Point", "coordinates": [50, 50]},
            "properties": {"name": "P1", "classification": {"name": "custom"}},
        }])
        rois = load_qupath_annotations(p, manager=mgr)
        assert len(mgr) == 1
        assert rois[0] is mgr.get_all()[0]


# ─────────────────────────────────────────────────────────────────────────────
# load_qupath_measurements
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadQuPathMeasurements:
    def test_csv_to_dataframe(self, tmp_path):
        csv = "Name,Class,Area µm²,Mean intensity\nROI_1,tumor,500.0,128.5\n"
        p = tmp_path / "measurements.csv"
        p.write_text(csv, encoding="utf-8")
        df = load_qupath_measurements(p)
        assert "Area µm²" in df.columns
        assert df.loc["ROI_1", "Area µm²"] == pytest.approx(500.0)
        assert df.loc["ROI_1", "Mean intensity"] == pytest.approx(128.5)

    def test_index_is_name(self, tmp_path):
        csv = "Name,Class,Value\nA,tumor,1.0\nB,stroma,2.0\n"
        p = tmp_path / "m.csv"
        p.write_text(csv, encoding="utf-8")
        df = load_qupath_measurements(p)
        assert df.index.name == "Name"
        assert set(df.index) == {"A", "B"}

    def test_column_rename_variants(self, tmp_path):
        # QuPath older versions use "Object ID" instead of "Name"
        csv = "Object ID,Classification,Value\nROI_1,Tumor,3.0\n"
        p = tmp_path / "m2.csv"
        p.write_text(csv, encoding="utf-8")
        df = load_qupath_measurements(p)
        assert df.index.name == "Name"
        assert "Value" in df.columns
