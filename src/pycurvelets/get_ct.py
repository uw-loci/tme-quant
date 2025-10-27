import numpy as np

from pycurvelets.models import (
    CurveletControlParameters,
    FeatureControlParameters,
    FiberFeatures,
)
from pycurvelets.new_curv import new_curv
from pycurvelets.process_fibers import process_fibers


def get_ct(
    img: np.ndarray,
    curve_cp: CurveletControlParameters,
    feature_cp: FeatureControlParameters,
):
    """
    Loads fiber output from CT-Fire and converts it into a structured format usable by CurveAlign.

    Parameters
    ----------
    img : np.ndarray
        2D image array of size [M, N] to be analyzed.
    curve_cp : CurveletControlParameters
        Control parameters for curvelets application.
    feature_cp : FeatureControlParameters
        Control parameters for extracted features (e.g., minimum fibers, box size).

    Returns
    -------
    fiber_structure: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``angle``: orientation angle in degrees
        - ``center_row``: row coordinate of the curvelet center
        - ``center_col``: column coordinate of the curvelet center
        - ``width``: width of the curvelet
    density_df: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``density_mean``: mean density of fibers
        - ``density_std``: standard deviation of density of fibers
    alignment_df: pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``alignment_mean``: mean alignment of fibers
        - ``alignment_std``: standard deviation of alignment of fibers
    """

    fiber_structure, curvelet_coefficients, _ = new_curv(img, curve_cp)

    if len(fiber_structure) == 0:
        return fiber_structure

    density_df, alignment_df = process_fibers(
        fiber_structure=fiber_structure, feature_cp=feature_cp
    )

    return fiber_structure, density_df, alignment_df
