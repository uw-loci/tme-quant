"""
Draw curvelets/fibers on an image as points (centers) and lines.

Converted from MATLAB drawCurvs.m
Laboratory for Optical and Computational Instrumentation 2013
"""

import numpy as np


def draw_curvs(
    fiber_data,
    ax,
    length,
    color_flag,
    angles,
    mark_size,
    line_width,
    boundary_measurement,
):
    """
    Draw curvelets/fibers on an image as points (centers) and lines.

    Parameters
    ----------
    fiber_data : pd.DataFrame or list
        Fiber/curvelet data with center coordinates and angles.
    ax : matplotlib.axes.Axes
        Axes object where the curvelets should be drawn.
    length : float
        Length of the curvelet indicator line.
    color_flag : int
        0 for green (used fibers), 1 for red (excluded fibers).
    angles : array_like
        Fiber angles in degrees.
    mark_size : float
        Marker size for center points.
    line_width : float
        Line width for fiber lines.
    boundary_measurement : bool
        True if boundary measurement is being performed.
    """
    # Handle DataFrame or list input
    if hasattr(fiber_data, "iterrows"):
        # DataFrame
        centers = []
        for _, row in fiber_data.iterrows():
            if "center_row" in fiber_data.columns:
                centers.append([row["center_row"], row["center_col"]])
            else:
                centers.append([row["center_1"], row["center_2"]])
        centers = np.array(centers)
    else:
        # Assume it's already array-like
        centers = np.array(fiber_data)

    if len(centers) == 0:
        print(f"draw_curvs: No fibers to draw (color_flag={color_flag})")
        return

    print(
        f"draw_curvs: Drawing {len(centers)} fibers (color_flag={color_flag}, boundary={boundary_measurement})"
    )

    # Draw each fiber - logic matches MATLAB exactly
    if boundary_measurement:
        # With boundary measurement
        for i in range(len(fiber_data)):

            xc = fiber_data["center_col"].iloc[i]
            yc = fiber_data["center_row"].iloc[i]

            # Plot center point
            if color_flag == 0:
                ax.plot(xc, yc, "g.", markersize=mark_size)
            else:
                ax.plot(xc, yc, "r.", markersize=mark_size)

            # Calculate line endpoints
            ca = np.deg2rad(fiber_data["angle"].iloc[i])
            xc1 = xc - length * np.cos(ca)
            xc2 = xc + length * np.cos(ca)
            yc1 = yc + length * np.sin(ca)
            yc2 = yc - length * np.sin(ca)

            # Plot line (always drawn when boundary measurement)
            if color_flag == 0:
                ax.plot([xc1, xc2], [yc1, yc2], "g-", linewidth=line_width)
            else:
                ax.plot([xc1, xc2], [yc1, yc2], "r-", linewidth=line_width)
    else:
        # No boundary measurement - only draw for color_flag == 0
        if color_flag == 0:
            for i, (center, angle) in enumerate(zip(centers, angles)):
                xc = center[1]  # column
                yc = center[0]  # row

                # Plot center point (red)
                ax.plot(xc, yc, "r.", markersize=mark_size)

                # Calculate line endpoints
                ca = np.deg2rad(angle)
                xc1 = xc - length * np.cos(ca)
                xc2 = xc + length * np.cos(ca)
                yc1 = yc + length * np.sin(ca)
                yc2 = yc - length * np.sin(ca)

                # Plot line (green)
                ax.plot([xc1, xc2], [yc1, yc2], "g-", linewidth=line_width)
        # else: don't draw anything for color_flag == 1 when no boundary
