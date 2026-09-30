"""Fast tests for the TME-Quant scikit-ops collection.

These tests do not build the Pixi environment. They verify discovery and call
the decorated functions directly in the development environment.
"""

from pathlib import Path

import numpy as np
import pytest

skop = pytest.importorskip("skop")

from pycurvelets.fiber_ops import fibers_to_vectors  # noqa: E402
from tme_quant_ops import (  # noqa: E402
    analyze_tacs,
    boundary_labels,
    enhance_frangi,
    enhance_tubeness,
    largest_boundary_points,
    segment_threshold,
    summarize_fiber_neighborhoods,
)


def _image() -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:14, 8:14] = 1
    image[20:26, 20:26] = 1
    return image


def test_collection_discovers_without_heavy_import_failures():
    specs, failures = skop.discover("tme_quant_ops")

    assert failures == []
    assert {spec.function for spec in specs} == {
        "analyze_tacs",
        "boundary_labels",
        "enhance_frangi",
        "enhance_tubeness",
        "extract_curvelets",
        "largest_boundary_points",
        "segment_threshold",
        "summarize_fiber_neighborhoods",
    }
    assert {spec.env for spec in specs} == {
        "tme-quant",
        "tme-quant-curvelets",
    }


def test_environment_definition_exists():
    root = Path(__file__).parents[1]
    runner = skop.Runner(root=root / "src", envs_dir=root / "envs")

    assert runner.env_config("tme-quant").is_file()
    assert runner.env_config("tme-quant-curvelets").is_file()


def test_threshold_and_boundary_ops_compose_directly():
    labels = segment_threshold(
        _image(), min_area=4, remove_border_objects=False
    )
    boundaries = boundary_labels(labels, thickness=1)

    assert labels.dtype == np.uint16
    assert labels.max() == 2
    assert boundaries.shape == labels.shape
    assert boundaries.dtype == np.uint16
    assert 0 < np.count_nonzero(boundaries) < np.count_nonzero(labels)


def test_preprocessing_ops_return_float_images():
    image = _image()

    tubeness = enhance_tubeness(image, sigma=1.0)
    vesselness = enhance_frangi(
        image, sigma_min=1.0, sigma_max=2.0, sigma_step=1.0
    )

    assert tubeness.shape == image.shape
    assert vesselness.shape == image.shape
    assert tubeness.dtype == np.float32
    assert vesselness.dtype == np.float32
    assert np.isfinite(tubeness).all()
    assert np.isfinite(vesselness).all()


def test_boundary_points_are_ordered_coordinates():
    labels = segment_threshold(_image(), min_area=4, remove_border_objects=False)

    points = largest_boundary_points(labels, simplify_tolerance=0.0)

    assert points.ndim == 2
    assert points.shape[1] == 2
    assert len(points) > 4


def test_fiber_summary_and_tacs_ops():
    boundary = np.concatenate(
        (
            np.column_stack((np.full(17, 8), np.arange(8, 25))),
            np.column_stack((np.arange(9, 25), np.full(16, 24))),
            np.column_stack((np.full(16, 24), np.arange(23, 7, -1))),
            np.column_stack((np.arange(23, 7, -1), np.full(16, 8))),
        )
    ).astype(float)
    centers = np.array([[6, 12], [6, 16], [6, 20]], dtype=float)
    # Horizontal fibers are tangent to the square's top edge.
    fibers = fibers_to_vectors(centers, np.zeros(3), length=5)

    neighborhoods = summarize_fiber_neighborhoods(fibers, neighbors=2)
    tacs = analyze_tacs(fibers, boundary, max_distance=10)

    assert neighborhoods.fiber_count == 3
    assert neighborhoods.global_alignment == pytest.approx(1.0)
    assert tacs.fiber_count == 3
    assert tacs.associations.shape == (3, 2, 2)
    assert tacs.mean_relative_angle == pytest.approx(0.0)
    assert tacs.tacs3_fraction == pytest.approx(0.0)
