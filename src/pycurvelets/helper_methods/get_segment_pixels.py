import numpy as np


def get_segment_pixels(point_1, point_2):
    segment_points = point_1
    absolute_angle = np.nan

    point_1 = np.round(point_1)
    point_2 = np.round(point_2)

    # get slope
    rise = point_2[0] - point_1[0]
    run = point_2[1] - point_1[1]

    # check if points are same
    maxrr = np.max(np.abs(rise), np.abs(run))
    if maxrr == 0:
        return

    absolute_angle = np.arctan(rise / run)  # range -pi to pi

    # walk this distance each iteration (sub pixel)
    fraction_rise = 0.5 * rise / maxrr
    fraction_run = 0.5 * run / maxrr

    spt = point_1
    y = spt[0]
    x = spt[1]
    i = 2  # index into output arr

    # initialize output (this will grow in memory, but segment should be short)
    segment_points = spt

    while True:
        if spt[0] == point_2[0] and spt[1] == point_2[1]:
            break

        # fractional accumulators
        y = y + fraction_rise
        x = x + fraction_run

        # round to pixel values
        round_y = np.round(y)
        round_x = np.round(x)
        if round_y != spt[0] or round_x != spt[1]:
            spt = [round_y, round_x]
            segment_points[i, :] = spt
            i = i + 1
