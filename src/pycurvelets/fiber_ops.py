"""Array-based fiber geometry and boundary analysis.

Fiber vectors follow napari/scikit-ops convention: ``(N, 2, D)`` where
``vectors[:, 0]`` is the center and ``vectors[:, 1]`` is a direction vector.
For 2-D images the coordinate order is ``(row/y, column/x)``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class NeighborhoodSummary(NamedTuple):
    fiber_count: int
    mean_nearest_distance: float
    mean_local_alignment: float
    global_alignment: float


class TACSSummary(NamedTuple):
    associations: np.ndarray
    fiber_count: int
    mean_distance: float
    mean_relative_angle: float
    tacs3_fraction: float


def fibers_to_vectors(
    centers: np.ndarray, angles_deg: np.ndarray, *, length: float = 10.0
) -> np.ndarray:
    """Convert fiber centers and axial angles into displayable vectors."""
    centers = np.asarray(centers, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (N, 2) in (row, column) order")
    if angles.shape != (len(centers),):
        raise ValueError("angles_deg must contain one value per center")
    if length <= 0:
        raise ValueError("length must be positive")

    radians = np.deg2rad(angles)
    # Image-coordinate direction: rows grow downward, columns to the right.
    directions = np.column_stack((np.sin(radians), np.cos(radians))) * length
    return np.stack((centers, directions), axis=1)


def vectors_to_fibers(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centers, axial_angles_deg)`` from 2-D fiber vectors."""
    vectors = _validate_vectors(vectors)
    centers = vectors[:, 0].astype(float, copy=False)
    directions = vectors[:, 1].astype(float, copy=False)
    angles = np.rad2deg(np.arctan2(directions[:, 0], directions[:, 1])) % 180
    return centers, angles


def summarize_neighborhoods(
    vectors: np.ndarray, *, neighbors: int = 4
) -> NeighborhoodSummary:
    """Summarize fiber spacing and local axial alignment."""
    from scipy.spatial import cKDTree

    centers, angles = vectors_to_fibers(vectors)
    count = len(centers)
    if count == 0:
        return NeighborhoodSummary(0, np.nan, np.nan, np.nan)
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")

    global_alignment = _axial_resultant(angles)
    if count == 1:
        return NeighborhoodSummary(1, np.nan, np.nan, global_alignment)

    k = min(neighbors, count - 1)
    distances, indices = cKDTree(centers).query(centers, k=k + 1)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    local = np.array(
        [
            _axial_resultant(np.concatenate(([angles[i]], angles[row])))
            for i, row in enumerate(neighbor_indices)
        ]
    )
    return NeighborhoodSummary(
        count,
        float(np.mean(neighbor_distances)),
        float(np.mean(local)),
        global_alignment,
    )


def analyze_tacs(
    vectors: np.ndarray,
    boundary_points: np.ndarray,
    *,
    min_distance: float = 0.0,
    max_distance: float = 200.0,
    tacs3_threshold: float = 60.0,
) -> TACSSummary:
    """Measure fiber angles relative to the nearest ordered boundary point.

    A relative angle of zero is tangential to the boundary; 90 degrees is
    perpendicular. ``associations`` contains vectors from each retained fiber
    center to its nearest boundary point.
    """
    from scipy.spatial import cKDTree

    centers, fiber_angles = vectors_to_fibers(vectors)
    boundary = np.asarray(boundary_points, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError("boundary_points must have shape (N, 2) with N >= 3")
    if min_distance < 0 or max_distance < min_distance:
        raise ValueError("distances must satisfy 0 <= min_distance <= max_distance")
    if not 0 <= tacs3_threshold <= 90:
        raise ValueError("tacs3_threshold must be between 0 and 90 degrees")

    distances, nearest = cKDTree(boundary).query(centers)
    keep = (distances >= min_distance) & (distances <= max_distance)
    if not np.any(keep):
        return TACSSummary(np.empty((0, 2, 2), dtype=float), 0, np.nan, np.nan, np.nan)

    selected_nearest = nearest[keep]
    previous = boundary[(selected_nearest - 1) % len(boundary)]
    following = boundary[(selected_nearest + 1) % len(boundary)]
    tangents = following - previous
    tangent_angles = np.rad2deg(np.arctan2(tangents[:, 0], tangents[:, 1])) % 180
    delta = np.abs(fiber_angles[keep] - tangent_angles) % 180
    relative = np.minimum(delta, 180 - delta)

    selected_centers = centers[keep]
    association_displacements = boundary[selected_nearest] - selected_centers
    associations = np.stack((selected_centers, association_displacements), axis=1)
    return TACSSummary(
        associations,
        int(np.count_nonzero(keep)),
        float(np.mean(distances[keep])),
        float(np.mean(relative)),
        float(np.mean(relative >= tacs3_threshold)),
    )


def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.ndim != 3 or vectors.shape[1:] != (2, 2):
        raise ValueError("vectors must have shape (N, 2, 2)")
    return vectors


def _axial_resultant(angles_deg: np.ndarray) -> float:
    angles = np.asarray(angles_deg, dtype=float)
    if not len(angles):
        return np.nan
    return float(np.abs(np.mean(np.exp(2j * np.deg2rad(angles)))))


__all__ = [
    "NeighborhoodSummary",
    "TACSSummary",
    "analyze_tacs",
    "fibers_to_vectors",
    "summarize_neighborhoods",
    "vectors_to_fibers",
]
