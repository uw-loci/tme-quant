"""
draw_map.py - Creates a heatmap where grey levels at each fiber center
correspond to angle information.

Converted from MATLAB drawMap.m
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from pycurvelets.utils.math import circ_r


def draw_map(fiber_structure, angles, img, boundary_measurement, map_params):
    """
    Create a heatmap showing fiber alignment patterns.

    Parameters
    ----------
    fiber_structure : pd.DataFrame
        DataFrame with fiber information including center positions
    angles : ndarray
        Array of fiber angles
    img : ndarray
        Original image
    boundary_measurement : bool
        Flag indicating if analysis is with respect to a boundary
    map_params : dict
        Control parameters for drawing the map:
        - STDfilter_size: Size for standard deviation filter
        - SQUAREmaxfilter_size: Size for square max filter
        - GAUSSIANdiscfilter_sigma: Sigma for Gaussian filter

    Returns
    -------
    rawmap : ndarray
        2D image where grey levels indicate angle information
    procmap : ndarray
        Filtered version of rawmap
    """
    J, I = img.shape  # height, width

    # Initialize raw map with NaN
    rawmap = np.full((J, I), np.nan, dtype=np.float64)

    # Get fiber centers
    if "center_row" in fiber_structure.columns:
        centers = fiber_structure[["center_row", "center_col"]].values
    else:
        centers = fiber_structure[["center_1", "center_2"]].values

    # Fill rawmap with angle values at fiber centers
    for i, (center, angle) in enumerate(zip(centers, angles)):
        xc = int(np.round(center[1]))  # col
        yc = int(np.round(center[0]))  # row

        # Check bounds
        if xc >= I or xc < 0 or yc >= J or yc < 0:
            continue

        if boundary_measurement:
            # Scale 0 to 90 degrees into 0 to 255
            rawmap[yc, xc] = 255.0 * (angle / 90.0)
        else:
            # Scale 0 to 180 degrees into 0 to 255
            rawmap[yc, xc] = 255.0 * (angle / 180.0)

    # Find positions of all non-nan values
    map2 = rawmap.copy()
    ind = np.where(~np.isnan(rawmap))
    y, x = ind[0], ind[1]

    if not boundary_measurement:
        # Standard deviation filter (circular statistics)
        fSize2 = map_params.get("STDfilter_size", 24)
        map2 = np.full((J, I), np.nan, dtype=np.float64)

        for i in range(len(y)):
            # Find positions in square region around current fiber
            mask = (
                (x > x[i] - fSize2)
                & (x < x[i] + fSize2)
                & (y > y[i] - fSize2)
                & (y < y[i] + fSize2)
            )

            # Get values in this region
            vals = rawmap[y[mask], x[mask]]
            vals = vals[~np.isnan(vals)]

            if len(vals) > 2:
                # Circular angle uniformity test
                # Scale from 0-255 to 0-2*pi, then compute circular mean resultant length
                # Scale result from 0-1 to 0-255
                map2[y[i], x[i]] = circ_r(vals * np.pi / 127.5) * 255

    # Max filter with normalization by fiber density
    fSize = map_params.get("SQUAREmaxfilter_size", 12)
    fSize2 = int(np.ceil(fSize / 2))
    map4 = np.full(img.shape, np.nan, dtype=np.float64)

    for i in range(len(y)):
        val = map2[y[i], x[i]]

        if np.isnan(val):
            continue

        # Define square region
        row_start = max(0, y[i] - fSize2)
        row_end = min(J, y[i] + fSize2 + 1)
        col_start = max(0, x[i] - fSize2)
        col_end = min(I, x[i] + fSize2 + 1)

        # Create meshgrid for square region
        rows = np.arange(row_start, row_end)
        cols = np.arange(col_start, col_end)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

        # Count number of fibers in this region (for normalization)
        region_vals = rawmap[row_grid, col_grid]
        num_fibs = np.sum(~np.isnan(region_vals))

        if num_fibs > 0:
            # Normalize by fiber density and take max
            normalized_val = val / num_fibs
            map4[row_grid, col_grid] = np.fmax(map4[row_grid, col_grid], normalized_val)

    # Gaussian filter
    sig = map_params.get("GAUSSIANdiscfilter_sigma", 4)

    # Convert NaN to 0 for filtering (like MATLAB's uint8 conversion)
    map4_for_filter = np.nan_to_num(map4, nan=0.0)

    # Apply Gaussian filter
    procmap = gaussian_filter(map4_for_filter, sigma=sig, mode="nearest")

    # Convert to uint8 (0-255 range)
    procmap = np.clip(procmap, 0, 255).astype(np.uint8)

    return rawmap, procmap
