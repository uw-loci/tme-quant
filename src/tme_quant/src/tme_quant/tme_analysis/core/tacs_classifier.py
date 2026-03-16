"""
TACS (Tumor-Associated Collagen Signatures) Classification.

Classifies fibers based on their orientation relative to tumor boundaries.

Angle convention
----------------
All classification functions expect ``angle_to_tangent``: the acute angle
(0–90°) between the fiber orientation and the LOCAL BOUNDARY TANGENT LINE.

    angle_to_tangent = 0°   → fiber runs along (parallel to) boundary → TACS-2
    angle_to_tangent = 90°  → fiber crosses boundary perpendicularly  → TACS-3
    angle_to_tangent = 30–60° or curly fiber                          → TACS-1

This is the biologically meaningful quantity because it directly expresses
how a fiber is oriented *relative to the boundary surface*.

Relationship to the boundary normal:
    angle_to_normal = 90° - angle_to_tangent

If you have the angle to the boundary NORMAL, convert before calling:
    angle_to_tangent = 90 - angle_to_normal

References
----------
- Provenzano et al. (2006) - TACS identification
- Conklin et al. (2011)   - TACS-3 and prognosis
"""

import numpy as np
from typing import Optional, Tuple

from shapely.geometry import Point, Polygon

from ...fiber_analysis.utils.geometry_utils import (
    compute_angle_to_boundary_normal,
    compute_angle_to_boundary_normal_simplified,
)


def classify_fiber_tacs(
    angle_to_tangent: float,
    straightness: float,
    distance_to_boundary: float,
    tacs_zone_width: float = 100.0,
    straightness_threshold: float = 0.7,
) -> Optional[str]:
    """
    Classify a single fiber as TACS-1, TACS-2, or TACS-3.

    Parameters
    ----------
    angle_to_tangent:
        Acute angle (0–90°) between fiber orientation and the LOCAL BOUNDARY
        TANGENT.  0° = parallel to boundary, 90° = perpendicular (invasive).
    straightness:
        Fiber straightness coefficient (0–1).  Values below
        *straightness_threshold* indicate a curly fiber (TACS-1).
    distance_to_boundary:
        Distance from the fiber to the nearest tumor boundary point (µm).
    tacs_zone_width:
        Only fibers within this distance of the boundary are classified.
        Default 100 µm.
    straightness_threshold:
        Minimum straightness required for TACS-2 or TACS-3.  Default 0.7.

    Returns
    -------
    'TACS-1', 'TACS-2', 'TACS-3', or None (if outside the TACS zone).

    Classification rules
    --------------------
    TACS-3  angle_to_tangent 60–90° AND straightness ≥ threshold  (INVASIVE)
    TACS-2  angle_to_tangent  0–30° AND straightness ≥ threshold  (parallel)
    TACS-1  angle_to_tangent 30–60° OR  straightness < threshold  (random)

    Examples
    --------
    >>> # Perpendicular, straight fiber within TACS zone -> INVASIVE
    >>> classify_fiber_tacs(75, 0.85, 50)
    'TACS-3'

    >>> # Parallel, straight fiber
    >>> classify_fiber_tacs(15, 0.85, 50)
    'TACS-2'

    >>> # Perpendicular but curly -> TACS-1 (not straight enough for TACS-3)
    >>> classify_fiber_tacs(75, 0.5, 50)
    'TACS-1'

    >>> # Outside TACS zone
    >>> classify_fiber_tacs(75, 0.85, 200)
    None
    """
    if distance_to_boundary > tacs_zone_width:
        return None

    if angle_to_tangent is None or np.isnan(angle_to_tangent):
        return None

    # Normalise to [0, 90]
    angle = float(np.abs(angle_to_tangent) % 90)

    if 60 <= angle <= 90:
        # Perpendicular to boundary — invasive pattern
        return 'TACS-3' if straightness >= straightness_threshold else 'TACS-1'

    elif 0 <= angle < 30:
        # Parallel to boundary
        return 'TACS-2' if straightness >= straightness_threshold else 'TACS-1'

    else:
        # Intermediate (30–60°) — always TACS-1 regardless of straightness
        return 'TACS-1'


def classify_fiber_segment_tacs_like(
    angle_to_tangent: float,
    distance_to_boundary: float,
    tacs_zone_width: float = 100.0,
) -> Optional[str]:
    """
    Classify a fiber segment (e.g. from CurveAlign) without straightness.

    Parameters
    ----------
    angle_to_tangent:
        Acute angle (0–90°) between fiber orientation and boundary tangent.
    distance_to_boundary:
        Distance to the nearest boundary point (µm).
    tacs_zone_width:
        TACS classification zone width (µm).

    Returns
    -------
    'TACS-1-like', 'TACS-2-like', 'TACS-3-like', or None.
    """
    if distance_to_boundary > tacs_zone_width:
        return None

    if angle_to_tangent is None or np.isnan(angle_to_tangent):
        return None

    angle = float(np.abs(angle_to_tangent) % 90)

    if 60 <= angle <= 90:
        return 'TACS-3-like'
    elif 0 <= angle < 30:
        return 'TACS-2-like'
    elif 30 <= angle < 60:
        return 'TACS-1-like'
    return None


def get_tacs_color(tacs_type: str) -> Tuple[int, int, int]:
    """
    Return the RGB display colour for a TACS classification.

    TACS-3 → RED   (255, 0,   0)  — perpendicular, invasive, high risk
    TACS-2 → GREEN (  0, 255, 0)  — parallel, medium risk
    TACS-1 → BLUE  (  0, 0, 255)  — random/intermediate, low risk
    """
    colors = {
        'TACS-3':      (255,   0,   0),
        'TACS-2':      (  0, 255,   0),
        'TACS-1':      (  0,   0, 255),
        'TACS-3-like': (255,   0,   0),
        'TACS-2-like': (  0, 255,   0),
        'TACS-1-like': (  0,   0, 255),
    }
    return colors.get(tacs_type, (128, 128, 128))   # grey for unknown


__all__ = [
    'classify_fiber_tacs',
    'classify_fiber_segment_tacs_like',
    'get_tacs_color',
]