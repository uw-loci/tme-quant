import numpy as np
from pycurvelets.utils.math import round_mlab


def get_segment_pixels(point_1, point_2):
    """
    Generate the set of pixel coordinates that approximate a straight line segment
    between two given points on a 2D grid.

    This function walks from `point_1` to `point_2` by incrementally stepping
    along the line defined by the two points, rounding intermediate positions to
    the nearest integer grid coordinates. The result is a discrete set of pixel
    coordinates that approximate the continuous line segment. The algorithm is
    similar in spirit to Bresenham's line algorithm, but uses fractional steps
    and rounding instead of integer-only arithmetic.

    Parameters
    ----------
    point_1 : array_like of shape (2,)
        Starting point of the segment, specified as (row, col) or (y, x).
    point_2 : array_like of shape (2,)
        Ending point of the segment, specified as (row, col) or (y, x).

    Returns
    -------
    segment_points : ndarray of shape (N, 2)
        Array of integer pixel coordinates representing the line segment,
        starting from `point_1` and ending at `point_2`. Each row contains
        (row, col) coordinates. If `point_1` and `point_2` are the same,
        returns `None`.

    Notes
    -----
    - The function rounds the input points to the nearest integer pixel
      coordinates before computing the segment.
    - If the rise or run is zero, the function still generates the appropriate
      vertical or horizontal line.
    - The internal array `segment_points` is expanded dynamically, so for long
      segments this may be less memory-efficient compared to classical
      rasterization algorithms.

    Examples
    --------
    >>> p1 = np.array([2, 3])
    >>> p2 = np.array([6, 8])
    >>> get_segment_pixels(p1, p2)
    array([[2., 3.],
           [3., 4.],
           [4., 5.],
           [5., 7.],
           [6., 8.]])
    """
    segment_points = point_1
    absolute_angle = np.nan

    point_1 = round_mlab(point_1)
    point_2 = round_mlab(point_2)

    # get slope
    rise = point_2[1] - point_1[1]
    run = point_2[0] - point_1[0]

    # check if points are same
    maxrr = np.max([np.abs(rise), np.abs(run)])
    if maxrr == 0:
        return

    absolute_angle = np.arctan(rise / run)  # range -pi to pi

    # walk this distance each iteration (sub pixel)
    fraction_rise = 0.5 * rise / maxrr
    fraction_run = 0.5 * run / maxrr

    spt = point_1
    x = spt[0]
    y = spt[1]
    i = 2  # index into output arr

    # initialize output (this will grow in memory, but segment should be short)
    segment_points = [spt]

    while True:
        if spt[0] == point_2[0] and spt[1] == point_2[1]:
            break

        # fractional accumulators
        y = y + fraction_rise
        x = x + fraction_run

        # round to pixel values
        round_y = round_mlab(y)
        round_x = round_mlab(x)
        if round_x != spt[0] or round_y != spt[1]:
            spt = [round_x, round_y]
            segment_points.append(spt)
            i = i + 1

    return segment_points, absolute_angle
