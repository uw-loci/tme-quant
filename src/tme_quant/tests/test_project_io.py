"""
Tests for save_project / load_project round-trip, including the
save_arrays / load_arrays sidecar extension for orientation map pixel arrays.
"""

import sys
import unittest.mock as mock

for lib in ["shapely", "shapely.geometry", "shapely.ops", "cv2"]:
    sys.modules.setdefault(lib, mock.MagicMock())

from pathlib import Path
import numpy as np
import pytest

from tme_quant.core.project import TMEProject
from tme_quant.core.io import save_project, load_project
from tme_quant.core.tme_objects.fiber_objects import (
    RegionOrientationMap, OrientationResult, OrientationMode,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_orientation_result(h: int = 32, w: int = 32) -> OrientationResult:
    return OrientationResult(
        mode=OrientationMode.CURVEALIGN,
        dimension="2D",
        orientation_map=np.random.rand(h, w).astype(np.float32),
        coherency_map=np.random.rand(h, w).astype(np.float32),
        mean_orientation=45.0,
        alignment_score=0.7,
    )


def _make_project_with_map(map_id: str = "map_1") -> TMEProject:
    proj = TMEProject(name="arr_test")
    result = _make_orientation_result()
    om = RegionOrientationMap(object_id=map_id, orientation_result=result)
    proj.orientation_maps[map_id] = om
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# Basic save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadProject:
    def test_basic_roundtrip(self, tmp_path):
        proj = TMEProject(name="test_proj")
        save_project(proj, tmp_path / "snap", overwrite=True)
        proj2 = load_project(tmp_path / "snap")
        assert proj2.name == "test_proj"

    def test_overwrite_guard_raises(self, tmp_path):
        proj = TMEProject(name="p")
        save_project(proj, tmp_path / "snap", overwrite=True)
        with pytest.raises(FileExistsError):
            save_project(proj, tmp_path / "snap", overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path):
        proj1 = TMEProject(name="first")
        save_project(proj1, tmp_path / "snap", overwrite=True)
        proj2 = TMEProject(name="second")
        save_project(proj2, tmp_path / "snap", overwrite=True)
        loaded = load_project(tmp_path / "snap")
        assert loaded.name == "second"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_project(tmp_path / "does_not_exist")


# ─────────────────────────────────────────────────────────────────────────────
# Orientation map sidecar arrays
# ─────────────────────────────────────────────────────────────────────────────

class TestArraySidecars:
    def test_arrays_dir_created_when_flag_true(self, tmp_path):
        proj = _make_project_with_map()
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        assert (tmp_path / "snap" / "arrays").is_dir()

    def test_orientation_npy_written(self, tmp_path):
        proj = _make_project_with_map("map_1")
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        assert (tmp_path / "snap" / "arrays" / "map_1_orientation.npy").exists()

    def test_coherency_npy_written(self, tmp_path):
        proj = _make_project_with_map("map_1")
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        assert (tmp_path / "snap" / "arrays" / "map_1_coherency.npy").exists()

    def test_no_arrays_dir_when_flag_false(self, tmp_path):
        proj = _make_project_with_map()
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=False)
        assert not (tmp_path / "snap" / "arrays").exists()

    def test_arrays_restored_on_load(self, tmp_path):
        proj = _make_project_with_map("map_1")
        orig_orient = proj.orientation_maps["map_1"].orientation_result.orientation_map.copy()
        orig_coh    = proj.orientation_maps["map_1"].orientation_result.coherency_map.copy()

        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        proj2 = load_project(tmp_path / "snap", load_arrays=True)

        restored_orient = proj2.orientation_maps["map_1"].orientation_result.orientation_map
        restored_coh    = proj2.orientation_maps["map_1"].orientation_result.coherency_map

        np.testing.assert_array_almost_equal(orig_orient, restored_orient)
        np.testing.assert_array_almost_equal(orig_coh, restored_coh)

    def test_load_without_flag_leaves_empty_array(self, tmp_path):
        proj = _make_project_with_map("map_1")
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        # load WITHOUT load_arrays — orientation_map should be empty (from from_dict)
        proj2 = load_project(tmp_path / "snap", load_arrays=False)
        om = proj2.orientation_maps["map_1"]
        if om is not None and om.orientation_result is not None:
            arr = om.orientation_result.orientation_map
            # from_dict restores it as np.array([]) — empty, not None
            assert arr is None or (isinstance(arr, np.ndarray) and arr.size == 0)

    def test_multiple_maps_all_restored(self, tmp_path):
        proj = TMEProject(name="multi")
        for i in range(3):
            result = _make_orientation_result()
            om = RegionOrientationMap(object_id=f"map_{i}", orientation_result=result)
            proj.orientation_maps[f"map_{i}"] = om

        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=True)
        proj2 = load_project(tmp_path / "snap", load_arrays=True)

        for i in range(3):
            arr = proj2.orientation_maps[f"map_{i}"].orientation_result.orientation_map
            assert isinstance(arr, np.ndarray) and arr.shape == (32, 32)

    def test_missing_sidecars_tolerated(self, tmp_path):
        proj = _make_project_with_map("map_1")
        # Save WITHOUT arrays, then try to load WITH load_arrays — should not error
        save_project(proj, tmp_path / "snap", overwrite=True, save_arrays=False)
        proj2 = load_project(tmp_path / "snap", load_arrays=True)
        assert "map_1" in proj2.orientation_maps
