try:
    from .new_curv import new_curv  # type: ignore

    HAS_CURVELETS = True
except Exception:
    new_curv = None
    HAS_CURVELETS = False

from .SHG_HE_registration import (
    BDcreation_reg2,
    SHGHERegistrationParameters,
    has_simpleitk,
    shg_he_registration,
)
from .tumor_annotation_from_HE import (
    BDcreationHE,
    BDcreationHE2,
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)
from ._registration_quality import (
    compute_mask_boundary_metrics,
    compute_registration_quality_metrics,
    compute_shg_alignment_metrics,
    make_checkerboard,
)

__all__ = [
    "HAS_CURVELETS",
    "new_curv",
    "SHGHERegistrationParameters",
    "has_simpleitk",
    "shg_he_registration",
    "BDcreation_reg2",
    "TumorAnnotationFromHEParameters",
    "tumor_annotation_from_he",
    "BDcreationHE2",
    "BDcreationHE",
    "compute_registration_quality_metrics",
    "compute_shg_alignment_metrics",
    "compute_mask_boundary_metrics",
    "make_checkerboard",
]
