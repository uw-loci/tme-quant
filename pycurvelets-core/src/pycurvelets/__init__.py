from ._registration_quality import (
    compute_mask_boundary_metrics,
    compute_registration_quality_metrics,
    compute_shg_alignment_metrics,
    make_checkerboard,
)
from .new_curv import new_curv
from .SHG_HE_registration import (
    BDcreation_reg,
    BDcreation_reg2,
    SHGHERegistrationParameters,
    shg_he_registration,
)
from .tumor_annotation_from_HE import (
    BDcreationHE,
    BDcreationHE2,
    TumorAnnotationFromHEParameters,
    tumor_annotation_from_he,
)

__all__ = [
    "HAS_CURVELETS",
    "new_curv",
    "SHGHERegistrationParameters",
    "shg_he_registration",
    "BDcreation_reg2",
    "BDcreation_reg",
    "TumorAnnotationFromHEParameters",
    "tumor_annotation_from_he",
    "BDcreationHE2",
    "BDcreationHE",
    "compute_registration_quality_metrics",
    "compute_shg_alignment_metrics",
    "compute_mask_boundary_metrics",
    "make_checkerboard",
]


def __getattr__(name):
    # HAS_CURVELETS is computed on access: loading a backend imports its native
    # library, which should not happen just because pycurvelets was imported.
    if name == "HAS_CURVELETS":
        from .backends import has_working_backend

        return has_working_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
