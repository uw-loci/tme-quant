import numpy as np


def circ_r(alpha, w=None, d=0, axis=0):
    """
    Computes mean resultant vector length for circular data.

    Parameters:
    - alpha : array-like
        Sample of angles in radians.
    - w : array-like, optional
        Weights (number of incidences). Default is uniform weights.
    - d : float, optional
        Spacing of bin centers for binned data, in radians. Used for bias correction.
    - axis : int, optional
        Axis along which to compute the result. Default is 0.

    Returns:
    - r : float or array
        Mean resultant length.
    """
    alpha = np.asarray(alpha)

    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w)
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    # Compute weighted sum of unit vectors
    r = np.sum(w * np.exp(1j * alpha), axis=axis)

    # Mean resultant vector length
    r = np.abs(r) / np.sum(w, axis=axis)

    # Bias correction for binned data (Zar, p. 601, eq. 26.16)
    if d != 0:
        c = d / (2 * np.sin(d / 2))
        r *= c

    return r
