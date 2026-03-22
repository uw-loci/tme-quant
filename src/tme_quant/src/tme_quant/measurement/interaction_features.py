"""
Specialised per-interaction feature extractors.

These functions compute scalar features for individual InteractionPair
objects (cell-fiber, fiber-tumor) that are then aggregated by
MeasurementEngine.  They cover:

  - Mechanical coupling score
  - Migration guidance score
  - Invasive potential score
  - Contact pattern metrics
  - Alignment heterogeneity

All scores are normalised to [0, 1] unless documented otherwise.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ..tme_analysis.config.analysis_params import InteractionPair
from ..core.tme_models.cell_model import CellObject
from ..core.tme_models.fiber_model import FiberObject


# ---------------------------------------------------------------------------
# Mechanical coupling
# ---------------------------------------------------------------------------

def compute_mechanical_coupling_score(
    fiber: FiberObject,
    distance: float,
    max_distance: float = 50.0,
) -> float:
    """
    Estimate the mechanical coupling between a fiber and a nearby cell/boundary.

    Higher scores indicate a stiffer, straighter fiber in close proximity —
    conditions that favour force transmission through the ECM.

    Score = (1 - distance/max_distance) * straightness * width_factor

    Parameters
    ----------
    fiber:
        The collagen fiber involved in the interaction.
    distance:
        Centre-to-centre or nearest-point distance in µm.
    max_distance:
        Normalisation distance (default 50 µm).

    Returns
    -------
    float in [0, 1].
    """
    if distance >= max_distance:
        return 0.0

    proximity = 1.0 - distance / max_distance
    straightness = fiber.straightness if fiber.straightness is not None else 0.5

    # Width contributes to effective stiffness — thicker bundles transmit more force.
    # Normalise against a representative bundle width of 5 µm.
    width = fiber.width if fiber.width is not None else 1.0
    width_factor = min(width / 5.0, 1.0)

    return float(proximity * straightness * (0.7 + 0.3 * width_factor))


# ---------------------------------------------------------------------------
# Migration guidance
# ---------------------------------------------------------------------------

def compute_migration_guidance_score(
    fiber: FiberObject,
    cell: Optional[CellObject] = None,
    alignment_angle: Optional[float] = None,
) -> float:
    """
    Score how well a fiber could guide cell migration.

    Straight, long fibers aligned with the cell's major axis provide
    strong directional cues.  Boundary tangent angle convention applies:
    0° = fiber runs along boundary (guidance parallel to boundary),
    90° = fiber points away from boundary (guidance into tissue).

    Parameters
    ----------
    fiber:
        Collagen fiber.
    cell:
        Optional cell for alignment calculation.
    alignment_angle:
        Pre-computed angle between fiber and cell major axis (0–90°).
        If None and cell is provided, estimated from cell orientation.

    Returns
    -------
    float in [0, 1].
    """
    straightness = fiber.straightness if fiber.straightness is not None else 0.5
    length = fiber.length if fiber.length is not None else 0.0
    # Normalise fiber length against a typical guidance-relevant length of 30 µm.
    length_factor = min(length / 30.0, 1.0)

    # Alignment factor: 0° angle = perfect alignment = score 1.
    if alignment_angle is not None:
        angle_rad = np.radians(min(abs(alignment_angle), 90.0))
        alignment_factor = float(np.cos(angle_rad))
    elif cell is not None and cell.orientation is not None and fiber.angle is not None:
        delta = abs(cell.orientation - fiber.angle) % 180.0
        if delta > 90.0:
            delta = 180.0 - delta
        alignment_factor = float(np.cos(np.radians(delta)))
    else:
        alignment_factor = 0.5  # unknown alignment

    return float(straightness * length_factor * alignment_factor)


# ---------------------------------------------------------------------------
# Invasive potential
# ---------------------------------------------------------------------------

def compute_invasive_potential_score(
    fiber: FiberObject,
    distance_to_boundary: Optional[float] = None,
    tacs_zone_width: float = 100.0,
) -> float:
    """
    Estimate the invasive potential contribution of a fiber.

    Based on TACS-3 logic: perpendicular, straight fibers close to the
    tumor boundary indicate an invasive phenotype.

    Parameters
    ----------
    fiber:
        Collagen fiber.
    distance_to_boundary:
        Distance to the nearest tumor boundary (µm).
        Falls back to ``fiber.nearest_boundary_distance`` if None.
    tacs_zone_width:
        TACS classification zone (µm).  Fibers beyond this distance
        contribute zero invasive potential.

    Returns
    -------
    float in [0, 1].
    """
    dist = (
        distance_to_boundary
        if distance_to_boundary is not None
        else getattr(fiber, 'nearest_boundary_distance', None)
    )
    if dist is None or dist >= tacs_zone_width:
        return 0.0

    proximity = 1.0 - dist / tacs_zone_width
    straightness = fiber.straightness if fiber.straightness is not None else 0.5

    # Perpendicularity: tangent angle 60–90° = TACS-3 territory.
    tangent_angle = getattr(fiber, 'relative_angle_to_boundary_tangent', None)
    if tangent_angle is not None:
        angle = abs(tangent_angle)
        if angle >= 60.0:
            perp_score = (angle - 60.0) / 30.0    # 0 at 60°, 1 at 90°
        else:
            perp_score = 0.0
    else:
        perp_score = 0.5   # unknown

    return float(proximity * straightness * (0.4 + 0.6 * perp_score))


# ---------------------------------------------------------------------------
# Contact pattern
# ---------------------------------------------------------------------------

def compute_contact_metrics(
    fiber: FiberObject,
    cell: CellObject,
    distance: float,
    contact_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute geometric contact metrics between a single fiber and cell.

    Parameters
    ----------
    fiber:
        Collagen fiber.
    cell:
        Cell object with boundary polygon.
    distance:
        Pre-computed centroid distance (µm).
    contact_threshold:
        Distance below which physical contact is assumed (µm).

    Returns
    -------
    dict with keys:
        is_contact (bool), contact_length (float), contact_area (float),
        contact_percentage (float, fraction of cell perimeter).
    """
    is_contact = distance <= contact_threshold

    # Estimate contact length from fiber width and proximity.
    # Physical contact length ≈ fiber_width when distance ≈ 0,
    # decreasing linearly to 0 at the contact threshold.
    fiber_width = fiber.width if fiber.width is not None else 1.0
    if is_contact and contact_threshold > 0:
        proximity = 1.0 - distance / contact_threshold
        contact_length = float(fiber_width * proximity)
    else:
        contact_length = 0.0

    # Contact area ≈ contact_length × cell boundary layer depth (1 µm proxy).
    contact_area = contact_length * 1.0

    # Contact percentage: fraction of cell perimeter within contact_threshold.
    if cell.perimeter and cell.perimeter > 0 and is_contact:
        contact_percentage = min(contact_length / cell.perimeter, 1.0)
    else:
        contact_percentage = 0.0

    return {
        'is_contact':          is_contact,
        'contact_length':      contact_length,
        'contact_area':        contact_area,
        'contact_percentage':  contact_percentage,
    }


# ---------------------------------------------------------------------------
# Alignment heterogeneity
# ---------------------------------------------------------------------------

def compute_alignment_heterogeneity(
    fibers: List[FiberObject],
    use_tangent_angle: bool = True,
) -> Dict[str, float]:
    """
    Quantify heterogeneity in fiber alignment relative to the tumor boundary.

    Parameters
    ----------
    fibers:
        List of FiberObject instances with boundary-relative angles set.
    use_tangent_angle:
        When True, uses ``relative_angle_to_boundary_tangent`` (0° = parallel,
        90° = perpendicular).  When False, uses ``angle``.

    Returns
    -------
    dict with keys:
        alignment_mean, alignment_std, alignment_cv, alignment_entropy,
        alignment_bimodality_index, coherence_index.
    """
    if use_tangent_angle:
        angles = [
            abs(f.relative_angle_to_boundary_tangent)
            for f in fibers
            if getattr(f, 'relative_angle_to_boundary_tangent', None) is not None
        ]
    else:
        angles = [
            f.angle for f in fibers
            if getattr(f, 'angle', None) is not None
        ]

    if not angles:
        return {}

    angles_arr = np.array(angles, dtype=float)
    mean_a = float(np.mean(angles_arr))
    std_a  = float(np.std(angles_arr))

    features: Dict[str, float] = {
        'alignment_mean': mean_a,
        'alignment_std':  std_a,
        'alignment_cv':   std_a / mean_a if mean_a > 0 else 0.0,
    }

    # Entropy of 10-bin histogram over [0, 90].
    hist, _ = np.histogram(angles_arr, bins=10, range=(0.0, 90.0), density=True)
    hist_pos = hist[hist > 0]
    features['alignment_entropy'] = float(-np.sum(hist_pos * np.log2(hist_pos)))

    # Bimodality index: variance of the distribution normalised to [0, 1].
    # Perfectly bimodal (all 0° or 90°) → index ≈ 1.
    max_var = (90.0 / 2.0) ** 2   # variance of uniform [0, 90]
    features['alignment_bimodality_index'] = float(
        min(np.var(angles_arr) / max_var, 1.0) if max_var > 0 else 0.0
    )

    # Coherence index: mean resultant length of doubled angles (von Mises).
    angles_rad = np.radians(2.0 * angles_arr)   # double to handle 0/90 symmetry
    mean_cos = float(np.mean(np.cos(angles_rad)))
    mean_sin = float(np.mean(np.sin(angles_rad)))
    features['coherence_index'] = float(np.sqrt(mean_cos ** 2 + mean_sin ** 2))

    return features


# ---------------------------------------------------------------------------
# Batch helper — populate InteractionPair fields
# ---------------------------------------------------------------------------

def annotate_interaction_pairs(
    pairs: List[InteractionPair],
    fibers: List[FiberObject],
    cells: Optional[List[CellObject]] = None,
    contact_threshold: float = 5.0,
    tacs_zone_width: float = 100.0,
) -> List[InteractionPair]:
    """
    Compute and store mechanical / invasive / guidance scores on each pair.

    Mutates the pairs in-place and returns them for convenience.

    Parameters
    ----------
    pairs:
        List of InteractionPair objects from InteractionDetector.
    fibers:
        All fibers in the region (used to look up by object_id).
    cells:
        All cells in the region (optional, used for guidance scores).
    contact_threshold:
        Physical contact distance in µm.
    tacs_zone_width:
        TACS zone width in µm.

    Returns
    -------
    The mutated pairs list.
    """
    fiber_map = {str(f.object_id): f for f in fibers}
    cell_map  = {str(c.object_id): c for c in cells} if cells else {}

    for pair in pairs:
        fiber = fiber_map.get(str(pair.source_id))
        if fiber is None:
            continue

        cell = cell_map.get(str(pair.target_id)) if cell_map else None

        pair.mechanical_coupling_score = compute_mechanical_coupling_score(
            fiber=fiber,
            distance=pair.distance,
        )
        pair.invasive_potential_score = compute_invasive_potential_score(
            fiber=fiber,
            distance_to_boundary=pair.distance,
            tacs_zone_width=tacs_zone_width,
        )
        pair.migration_guidance_score = compute_migration_guidance_score(
            fiber=fiber,
            cell=cell,
        )

        if cell is not None:
            contact = compute_contact_metrics(
                fiber=fiber,
                cell=cell,
                distance=pair.distance,
                contact_threshold=contact_threshold,
            )
            pair.contact_length     = contact['contact_length']
            pair.contact_area       = contact['contact_area']
            pair.contact_percentage = contact['contact_percentage']

    return pairs


__all__ = [
    'compute_mechanical_coupling_score',
    'compute_migration_guidance_score',
    'compute_invasive_potential_score',
    'compute_contact_metrics',
    'compute_alignment_heterogeneity',
    'annotate_interaction_pairs',
]