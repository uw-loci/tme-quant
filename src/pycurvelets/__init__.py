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
    BDcreationHE2,
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
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
]
