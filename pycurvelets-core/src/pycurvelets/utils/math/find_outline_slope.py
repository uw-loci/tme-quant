import numpy as np
from ..connectivity.find_connected_pts import find_connected_pts


def find_outline_slope(coords, idx, num=21):
    """
    Compute the absolute angle of the boundary outline around a given index.
    Similar to MATLAB's FindOutlineSlope.

    Inputs:
        coords: np.ndarray of shape (N, 2) where each row is (row, col) = (y, x)
        idx: integer index into coords
        num: number of points to sample around idx (default 21)

    Returns:
        slope: float, angle in degrees from 0 to 180
    """
    slope = None
    con_pts = find_connected_pts(coords, idx, num)

    if con_pts.shape[0] < num:
        return np.nan

    # Calculate rough slope first
    rise = con_pts[-1, 1] - con_pts[0, 1]
    run = con_pts[-1, 0] - con_pts[0, 0]

    if run == 0:
        slope = 90.0
    else:
        slope = np.degrees(np.arctan(rise / run)) % 180

    # Determine dominant direction
    if slope < 45 or slope > 135:
        # Fit y as function of x (horizontal)
        x_fit = np.linspace(con_pts[0, 0], con_pts[-1, 0], 50)
        y_coeffs = np.polyfit(con_pts[:, 0], con_pts[:, 1], 2)
        y_fit = np.polyval(y_coeffs, x_fit)
    else:
        # Fit x as function of y (vertical)
        y_fit = np.linspace(con_pts[0, 1], con_pts[-1, 1], 50)
        x_coeffs = np.polyfit(con_pts[:, 1], con_pts[:, 0], 2)
        x_fit = np.polyval(x_coeffs, y_fit)

    run2 = x_fit[25] - x_fit[23]
    rise2 = y_fit[25] - y_fit[23]

    theta2 = np.degrees(np.arctan2(rise2, run2))
    slope2 = theta2 % 180

    return slope2
