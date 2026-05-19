"""Tests for ProjectController signal/state contract."""

import unittest.mock as mock

import pytest


@pytest.fixture
def state():
    from napari_tme_quant.controllers.state import PluginState
    return PluginState()


@pytest.fixture
def controller(state):
    from napari_tme_quant.controllers.project_controller import ProjectController
    return ProjectController(state)


class TestProjectControllerSelectImage:

    def test_select_image_sets_active_image_id(self, state, controller):
        controller.select_image("SHG_001")
        assert state.active_image_id == "SHG_001"

    def test_select_image_overwrites_previous(self, state, controller):
        controller.select_image("SHG_001")
        controller.select_image("SHG_002")
        assert state.active_image_id == "SHG_002"

    def test_select_image_fires_signal(self, controller):
        received = []
        controller.connect_image_selected(received.append)
        controller.select_image("img_A")
        assert received == ["img_A"]

    def test_select_image_fires_multiple_listeners(self, controller):
        a, b = [], []
        controller.connect_image_selected(a.append)
        controller.connect_image_selected(b.append)
        controller.select_image("img_X")
        assert a == ["img_X"]
        assert b == ["img_X"]


class TestProjectControllerImageType:

    def test_set_image_type_stores_in_state(self, state, controller):
        from napari_tme_quant.controllers.state import ImageType
        controller.set_image_type("SHG_001", ImageType.FIBER)
        assert state.image_types["SHG_001"] == ImageType.FIBER

    def test_set_image_type_overwrites(self, state, controller):
        from napari_tme_quant.controllers.state import ImageType
        controller.set_image_type("img", ImageType.FIBER)
        controller.set_image_type("img", ImageType.CELL)
        assert state.image_types["img"] == ImageType.CELL

    def test_set_image_type_fires_signal(self, controller):
        from napari_tme_quant.controllers.state import ImageType
        received = []
        controller.connect_type_changed(lambda iid, t: received.append((iid, t)))
        controller.set_image_type("img", ImageType.MASK)
        assert received == [("img", ImageType.MASK)]

    def test_set_image_type_all_types_accepted(self, state, controller):
        from napari_tme_quant.controllers.state import ImageType
        for itype in ImageType:
            controller.set_image_type(f"img_{itype.name}", itype)
            assert state.image_types[f"img_{itype.name}"] == itype


class TestProjectControllerPairing:

    def test_set_pair_stores_in_state(self, state, controller):
        controller.set_pair("SHG_001", "HE_001")
        assert state.image_pairs["SHG_001"] == "HE_001"

    def test_set_pair_overwrites(self, state, controller):
        controller.set_pair("SHG_001", "HE_001")
        controller.set_pair("SHG_001", "HE_002")
        assert state.image_pairs["SHG_001"] == "HE_002"

    def test_set_pair_fires_signal(self, controller):
        received = []
        controller.connect_pair_changed(lambda fid, cid: received.append((fid, cid)))
        controller.set_pair("SHG_001", "HE_001")
        assert received == [("SHG_001", "HE_001")]

    def test_multiple_pairs_stored_independently(self, state, controller):
        controller.set_pair("SHG_001", "HE_001")
        controller.set_pair("SHG_002", "HE_002")
        assert state.image_pairs["SHG_001"] == "HE_001"
        assert state.image_pairs["SHG_002"] == "HE_002"
