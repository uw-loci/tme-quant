import math

import numpy as np
import pandas as pd

from pycurvelets.backends import load_backend
from pycurvelets.models import CurveletControlParameters
from pycurvelets.utils.math import round_mlab


def new_curv(img, curve_cp: CurveletControlParameters):
    """
    Python implementation of newCurv.m

    This function applies a discrete curvelet transform to an image, then extracts
    the curvelet coefficients at a given scale with magnitude above a given threshold.
    The orientation (angle, in degrees) and center point of each curvelet is then stored.

    The transform comes from the backend named by ``curve_cp.backend`` (see
    :mod:`pycurvelets.backends.registry`). With the ``fdct`` backend the output
    matches MATLAB CurveAlign.

    Parameters:
    -----------
    img : ndarray
        Input image
    curve_cp : CurveletControlParameters
        Dataclass containing curvelet control parameters:
            - keep : float
                Fraction of the curvelets to be kept.
            - scale : float
                Scale to be analyzed.
            - radius : float
                Radius to group adjacent curvelets.
            - backend : str or None
                Curvelet backend name; None picks one as described in
                :func:`pycurvelets.backends.load_backend`.

    Returns:
    --------
    in_curves : pandas.DataFrame
        DataFrame containing one row per curvelet, with columns:
        - ``angle``: orientation angle in degrees
        - ``center_row``: row coordinate of the curvelet center
        - ``center_col``: column coordinate of the curvelet center
    curvelet_coefficients : list of lists
        A nested list containing the thresholded curvelet coefficients
    inc : float
        Angle increment used
    """
    keep = curve_cp.keep
    s_scale = curve_cp.scale
    radius = curve_cp.radius

    # Apply the curvelet transform to the image
    backend = load_backend(curve_cp.backend)
    c = backend.forward(img)

    # Create an empty structure of the same dimensions
    curvelet_coefficients = [
        [np.zeros_like(c[cc][dd]) for dd in range(len(c[cc]))] for cc in range(len(c))
    ]

    # Select the scale at which the coefficients will be used
    s = len(c) - s_scale - 1  # Same formula as MATLAB: length(C) - Sscale

    # Take absolute value of coefficients
    for ee in range(len(c[s])):
        c[s][ee] = np.abs(c[s][ee])

    # Find the maximum coefficient value
    abs_max = max(np.max(arr) for arr in c[s])

    # MATLAB: bins = 0:.01*absMax:absMax;
    bins = np.linspace(0, abs_max, 101)  # 101 centers including endpoints

    # Convert MATLAB centers -> NumPy edges
    bin_width = bins[1] - bins[0]
    bin_edges = np.concatenate(([bins[0] - bin_width / 2], bins + bin_width / 2))

    hist_per_wedge = []
    for arr in c[s]:
        hist_w, _ = np.histogram(arr.flatten(), bins=bin_edges)
        hist_per_wedge.append(hist_w)

    # Sum across wedges — same as MATLAB
    sum_hist = np.sum(hist_per_wedge, axis=0)
    cum_sum = np.cumsum(sum_hist)

    # Find threshold index like MATLAB: first index where cumulative > (1-keep)*total
    threshold_idx = np.where(cum_sum > (1 - keep) * cum_sum[-1])[0][0]
    max_val = bins[threshold_idx]  # Use bin center, not edge!

    # Threshold coefficients — keep values >= max_val, zero otherwise
    for dd in range(len(c[s])):
        mask = np.abs(c[s][dd]) >= max_val
        curvelet_coefficients[s][dd] = c[s][dd] * mask

    # Debug: print threshold info
    # print(f"Debug: scale s={s}, len(c)={len(c)}, s_scale={s_scale}")
    # print(f"Debug: abs_max={abs_max:.6f}, max_val={max_val:.6f}, keep={keep}")
    # total_nnz = sum((arr != 0).sum() for arr in ct[s])
    # print(f"Debug: total non-zero coefficients after thresholding: {total_nnz}")

    # Get locations of curvelet centers and find angles
    geometry = backend.scale_geometry(img.shape, s)
    long = len(geometry.wedges)
    angs = [np.array([]) for _ in range(long)]
    row = [np.array([]) for _ in range(long)]
    col = [np.array([]) for _ in range(long)]
    inc = geometry.angle_step_deg

    for w, wedge in enumerate(geometry.wedges):
        # Find non-zero coefficients
        # MATLAB: test = find(Ct{s}{w});
        test = np.flatnonzero(curvelet_coefficients[s][w])

        if len(test) > 0:
            angs[w] = np.full(len(test), wedge.angle_deg)
            row[w] = round_mlab(wedge.rows[test])
            col[w] = round_mlab(wedge.cols[test])
        else:
            angs[w] = np.array([0])
            row[w] = np.array([0])
            col[w] = np.array([0])

    # Find non-empty arrays
    c_test = [len(c) > 0 and not (len(c) == 1 and c[0] == 0) for c in col]
    bb = np.where(c_test)[0]
    # print(f"Debug: found {len(bb)} non-empty wedges out of {len(col)} total wedges")

    if len(bb) == 0:  # No curvelets found
        print("Error; no curvelets found")
        return [], curvelet_coefficients, inc

    # Concatenate non-empty arrays
    col_flat = np.concatenate([col[i] for i in bb])
    row_flat = np.concatenate([row[i] for i in bb])
    angs_flat = np.concatenate([angs[i] for i in bb])
    # print(f"Debug: total curvelets before grouping: {len(col_flat)}")

    curves = np.column_stack((row_flat, col_flat, angs_flat))
    curves2 = curves.copy()

    # Group all curvelets that are closer than 'radius'
    # Match MATLAB exactly: groups = cell(1,length(curves));
    groups = [[] for _ in range(len(curves2))]
    valid_curvelets = 0
    for xx in range(len(curves2)):
        if np.all(curves2[xx, :]):  # Check if the curvelet is valid
            valid_curvelets += 1
            # MATLAB: cLow = curves2(:,2) > ceil(curves2(xx,2) - radius);
            # MATLAB: cHi = curves2(:,2) < floor(curves2(xx,2) + radius);
            # MATLAB: cRad = cHi .* cLow;
            c_low = curves2[:, 1] > math.ceil(curves2[xx, 1] - radius)
            c_hi = curves2[:, 1] < math.floor(curves2[xx, 1] + radius)
            c_rad = c_hi & c_low  # MATLAB uses .* (element-wise AND)

            # MATLAB: rHi = curves2(:,1) < ceil(curves2(xx,1) + radius);
            # MATLAB: rLow = curves2(:,1) > floor(curves2(xx,1) - radius);
            # MATLAB: rRad = rHi .* rLow;
            r_hi = curves2[:, 0] < math.ceil(curves2[xx, 0] + radius)
            r_low = curves2[:, 0] > math.floor(curves2[xx, 0] - radius)
            r_rad = r_hi & r_low  # MATLAB uses .* (element-wise AND)

            # MATLAB: inNH = logical(cRad .* rRad);
            in_nh = c_rad & r_rad
            groups[xx] = np.where(in_nh)[0]  # Store indices of grouped curvelets

            # MATLAB: curves2(inNH,:) = 0;
            curves2[in_nh, :] = 0  # Mark grouped curvelets as processed

    # print(f"Debug: valid curvelets for grouping: {valid_curvelets}")

    # Keep only non-empty groups
    not_empty = [len(g) > 0 for g in groups]
    comb_nh = [g for g in groups if len(g) > 0]
    n_hoods = [curves[g] for g in comb_nh]

    # print(f"Debug: curvelets after grouping: {len(n_hoods)}")

    # Helper function for fixing angles
    def fix_angle(angles, inc):
        """
        Match MATLAB's fixAngle function to properly adjust angles
        """
        x = np.array(angles, dtype=float)
        bins = np.arange(np.min(x), np.max(x) + inc, inc)

        temp = x.copy()
        angs = x.copy()
        stdev = []

        for aa in range(len(bins) - 1):
            idx = temp >= bins[-(aa + 1)]
            temp_adj = temp.copy()
            temp_adj[idx] -= 180
            stdev.append(np.std(temp_adj))

        stdev = [np.std(x)] + stdev
        stdev = np.array(stdev)
        I = np.argmin(stdev)
        C = stdev[I]

        if C < np.std(angs) and I < len(bins) - 1:
            idx = angs >= bins[-(I + 1)]
            angs[idx] -= 180

            if I > 0.5 * len(bins):
                angs += 180

        if np.any(angs < 0):
            angs += 180

        return np.mean(angs)

    angles = [fix_angle(nh[:, 2], inc) for nh in n_hoods]
    centers = [
        np.array([round_mlab(np.median(nh[:, 0])), round_mlab(np.median(nh[:, 1]))])
        for nh in n_hoods
    ]

    objects = [
        {"center": center, "angle": angle} for center, angle in zip(centers, angles)
    ]

    def group6(objects):
        for i in range(len(objects)):
            angle = objects[i]["angle"]
            objects[i]["angle"] = (180 + angle) % 180
        return objects

    objects = group6(objects)

    # Remove curvelets too close to the edge
    all_center_points = np.vstack([obj["center"] for obj in objects])
    cen_row = all_center_points[:, 0]  # First column is row in the center points
    cen_col = all_center_points[:, 1]  # Second column is col in the center points
    im_rows, im_cols = img.shape
    edge_buf = math.ceil(min(im_rows, im_cols) / 100)

    # Find indices of curvelets that are not too close to the edge
    in_idx = np.where(
        (cen_row < im_rows - edge_buf)
        & (cen_col < im_cols - edge_buf)
        & (cen_row > edge_buf)
        & (cen_col > edge_buf)
    )[0]

    in_curves = pd.DataFrame(
        [
            {
                "center_row": obj["center"][0],
                "center_col": obj["center"][1],
                "angle": obj["angle"],
            }
            for i, obj in enumerate(objects)
            if i in in_idx
        ]
    ).reset_index(drop=True)

    return in_curves, curvelet_coefficients, inc
