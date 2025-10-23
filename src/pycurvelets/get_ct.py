import numpy as np

from pycurvelets.models import CurveletControlParameters, FeatureControlParameters
from pycurvelets.new_curv import new_curv


def get_ct(
    img_name: str,
    img: np.ndarray,
    curve_cp: CurveletControlParameters,
    feature_cp: FeatureControlParameters,
):
    """
    Loads fiber output from CT-Fire and converts it into a structured format usable by CurveAlign.

    Parameters
    ----------
    img_name : str
        Name of the image we are processing.
    img : np.ndarray
        2D image array of size [M, N] to be analyzed.
    curve_cp : CurveletControlParameters
        Control parameters for curvelets application.
    feature_cp : FeatureControlParameters
        Control parameters for extracted features (e.g., minimum fibers, box size).

    Returns
    -------
    CTResult
        Dataclass containing:
            - objects: info about each fiber segment (position, angle)
            - fiber_key: index of beginning of each fiber
            - total_length_list
            - end_length_list
            - curvature_list
            - width_list
            - density_list
            - alignment_list
            - Ct: processed curvelet/fiber data
    """

    fiber_structure, curvelet_coefficients, _ = new_curv(img, curve_cp)

    object_list = []
    fiber_key = []
    total_length_list = []
    end_length_list = []
    curvature_list = []
    width_list = []
    density_list = []
    alignment_list = []

    if len(fiber_structure) == 0:
        return (
            object_list,
            fiber_key,
            total_length_list,
            end_length_list,
            curvature_list,
            width_list,
            density_list,
            alignment_list,
            curvelet_coefficients,
        )

    fiber_number = len(fiber_structure)

    return (
        object_list,
        fiber_key,
        total_length_list,
        end_length_list,
        curvature_list,
        width_list,
        density_list,
        alignment_list,
        curvelet_coefficients,
    )
