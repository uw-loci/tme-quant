"""
fiber2beam - Convert fiber array to reduced beam-interpolated fiber array.

Ports MATLAB's fiber2beam.m, bestcurv.m, plotbeam.m, len3d.m, Rcalc3.m
from /curvelets/src/FIRE/beamproc/.

Algorithm
---------
1. For each fiber, identify "pinned" vertices: the two endpoints plus any
   internal vertex that belongs to more than one fiber (a crosslink).
2. Fit a Hermite cubic spline through the pinned vertices using least-squares
   minimization (``bestcurv``).
3. Evaluate the spline at enough points so adjacent nodes are ≤ ``minspace``
   apart (``plotbeam``).
4. Append new interpolation nodes to X and rebuild F with the new vertex
   indices; then compact via ``trimxfv``.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional

from ctfire_py.utils.trimxfv import trimxfv


# ---------------------------------------------------------------------------
# Hermite basis functions
# ---------------------------------------------------------------------------

def _h1(x: np.ndarray) -> np.ndarray:
    """H1(x) = x³ - 2x² + x  (MATLAB: inline('x.^3-2*x.^2+x'))"""
    return x**3 - 2 * x**2 + x


def _h2(x: np.ndarray) -> np.ndarray:
    """H2(x) = -x³ + x²  (MATLAB: inline('-x.^3 + x.^2'))"""
    return -(x**3) + x**2


# ---------------------------------------------------------------------------
# Rcalc3 - rotation matrix mapping a vector to [1, 0, 0]
# ---------------------------------------------------------------------------

def rcalc3(v: np.ndarray) -> np.ndarray:
    """
    Compute a 3-D rotation matrix R such that R @ v / |v| ≈ [1, 0, 0].

    Faithful port of MATLAB Rcalc3.m.

    Parameters
    ----------
    v : array_like, shape (3,) or (1, 3)

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    v = np.asarray(v, dtype=float).ravel()

    # First rotate in the XY plane so the projection onto XY aligns with X.
    lxy = np.linalg.norm(v[:2])
    if lxy == 0:
        R1 = np.eye(3)
    else:
        ca, sa = v[0] / lxy, v[1] / lxy
        R1 = np.array([[ ca, sa, 0],
                       [-sa, ca, 0],
                       [  0,  0, 1]], dtype=float)

    vv = R1 @ v

    # Then rotate in the XZ plane to fully align with X.
    lxz = np.linalg.norm(vv[[0, 2]])
    if lxz == 0:
        R2 = np.eye(3)
    else:
        cb, sb = vv[0] / lxz, vv[2] / lxz
        R2 = np.array([[ cb, 0, sb],
                       [  0, 1,  0],
                       [-sb, 0, cb]], dtype=float)

    return R2 @ R1


# ---------------------------------------------------------------------------
# len3d - arc length of a 3-D polyline
# ---------------------------------------------------------------------------

def len3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Segment lengths of a polyline given by arrays x, y, z.

    Returns an array of length max(0, len(x)-1).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if len(x) == 0:
        return np.array([])
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return np.sqrt(dx**2 + dy**2 + dz**2)


# ---------------------------------------------------------------------------
# bestcurv - fit Hermite cubic spline through pinned vertices
# ---------------------------------------------------------------------------

def bestcurv(
    X: np.ndarray,
    pins: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a piecewise Hermite cubic spline through the pinned vertices.

    Faithful port of MATLAB bestcurv.m.

    Parameters
    ----------
    X    : ndarray, shape (N, 3) - all vertex coordinates for this fiber
    pins : 1-D int array - 0-based indices into X identifying the pinned
           vertices (endpoint, crosslinks, endpoint)
    lam  : float - Tikhonov regularization constant (typically 0.01)

    Returns
    -------
    AL : ndarray, shape (n_segments, 2) - left-tangent (y, z) coefficients
    AR : ndarray, shape (n_segments, 2) - right-tangent (y, z) coefficients
    """
    pins = np.asarray(pins, dtype=int)
    n_seg = len(pins) - 1
    AL = np.zeros((n_seg, 2))
    AR = np.zeros((n_seg, 2))

    for i in range(n_seg):
        # Vertices in this segment (0-based index range)
        ii = np.arange(pins[i], pins[i + 1] + 1)
        Xi = X[ii, :]

        if len(Xi) <= 2:
            # Only two points - no need for optimisation
            AL[i] = [0.0, 0.0]
            AR[i] = [0.0, 0.0]
            continue

        # Transform to local coordinate frame aligned with the segment axis.
        X1 = Xi[0, :]
        Xi_local = Xi - X1
        R = rcalc3(Xi_local[-1, :])       # rotate so endpoint → [1,0,0]
        Xir = (R @ Xi_local.T).T          # shape (m, 3)

        L = Xir[-1, 0]                    # length along local x
        if L == 0:
            AL[i] = [0.0, 0.0]
            AR[i] = [0.0, 0.0]
            continue

        x = Xir[:, 0] / L                 # normalised arc coordinate in [0,1]
        y = Xir[:, 1]
        z = Xir[:, 2]

        h1 = _h1(x)
        h2 = _h2(x)

        # Regularised normal equations  (A @ c = B)
        A = np.array([
            [lam + np.sum(h1 * h1), np.sum(h1 * h2)],
            [np.sum(h1 * h2),       lam + np.sum(h2 * h2)],
        ])
        By = np.array([[np.sum(h1 * y)], [np.sum(h2 * y)]])
        Bz = np.array([[np.sum(h1 * z)], [np.sum(h2 * z)]])

        try:
            cy = np.linalg.solve(A, By).ravel()
            cz = np.linalg.solve(A, Bz).ravel()
        except np.linalg.LinAlgError:
            cy = np.zeros(2)
            cz = np.zeros(2)

        AL[i] = [cy[0], cz[0]]
        AR[i] = [cy[1], cz[1]]

    return AL, AR


# ---------------------------------------------------------------------------
# plotbeam - evaluate Hermite spline at N evenly-spaced points per segment
# ---------------------------------------------------------------------------

def plotbeam(
    X: np.ndarray,
    AL: np.ndarray,
    AR: np.ndarray,
    N: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a piecewise Hermite spline at N points per segment.

    Faithful port of MATLAB plotbeam.m (without plotting).

    Parameters
    ----------
    X  : ndarray, shape (n, 3) - pinned vertex coordinates
    AL : ndarray, shape (n-1, 2) - left-tangent (y, z) coefficients
    AR : ndarray, shape (n-1, 2) - right-tangent (y, z) coefficients
    N  : int - number of evaluation points per segment (including endpoints)

    Returns
    -------
    x, y, z : ndarray, shape (n_pts,) - interpolated global coordinates
    """
    n = len(X)
    xs, ys, zs = [], [], []

    for i in range(n - 1):
        X1 = X[i,  :]
        X2 = X[i + 1, :]
        diff = X2 - X1
        Lseg = np.linalg.norm(diff)

        if Lseg == 0:
            xs.append(X1[0])
            ys.append(X1[1])
            zs.append(X1[2])
            continue

        R3 = rcalc3(diff)               # rotation: diff → [1, 0, 0]
        t = np.linspace(0.0, 1.0, N)
        xloc = t                         # normalised arc position
        yloc = AL[i, 0] * _h1(xloc) + AR[i, 0] * _h2(xloc)
        zloc = AL[i, 1] * _h1(xloc) + AR[i, 1] * _h2(xloc)
        xloc = xloc * Lseg

        # Transform back to global frame
        Xglob = R3.T @ np.vstack([xloc, yloc, zloc]) + X1[:, np.newaxis]

        xs.append(Xglob[0, :])
        ys.append(Xglob[1, :])
        zs.append(Xglob[2, :])

    if not xs:
        return np.array([]), np.array([]), np.array([])

    x_out = np.concatenate([np.atleast_1d(a) for a in xs])
    y_out = np.concatenate([np.atleast_1d(a) for a in ys])
    z_out = np.concatenate([np.atleast_1d(a) for a in zs])
    return x_out, y_out, z_out


# ---------------------------------------------------------------------------
# fiber2beam - main function
# ---------------------------------------------------------------------------

def fiber2beam(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: np.ndarray,
    minspace: float,
    lam: float,
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Convert fiber array to a reduced beam-interpolated fiber array.

    Faithful port of MATLAB fiber2beam.m.

    Parameters
    ----------
    X        : ndarray, shape (N, 3) - vertex coordinates (1-based MATLAB
               convention: vertex k is at X[k-1])
    F        : list of dicts with 'v' (1-based vertex indices) and optional 'r'
    V        : list of dicts with 'f' (fiber memberships, 1-based)
    R        : ndarray, shape (N,) - per-vertex radii (unused here, but
               forwarded to trimxfv)
    minspace : float - minimum desired spacing between interpolated nodes (px)
    lam      : float - Tikhonov regularisation for bestcurv (e.g. 0.01)

    Returns
    -------
    X, F, V  : updated arrays after interpolation and trimxfv compaction
    """
    if not F:
        return X, F, V

    print(f"    finding curves for {len(F)} fibers")

    # Ensure X is (N, 3) float
    X = np.asarray(X, dtype=float)
    if X.ndim == 2 and X.shape[1] == 2:
        # 2-D image: pad with zeros in the z column
        X = np.column_stack([X, np.zeros(len(X))])

    # ---- Pass 1: identify pinned vertices and fit splines ----
    AL_list: List[np.ndarray] = []          # Hermite left-tangent per fiber
    AR_list: List[np.ndarray] = []          # Hermite right-tangent per fiber
    Xcrit_list: List[np.ndarray] = []       # critical (pinned) X positions
    Fred: List[Dict] = []                   # reduced fiber list (pinned only)

    for i, fiber in enumerate(F):
        vi = list(fiber['v'])               # 1-based vertex indices (MATLAB convention)
        vi0 = [v - 1 for v in vi]          # 0-based for X indexing

        # Identify pinned vertex positions within vi
        # The first and last vertices are always pinned.
        pin_local_idx = [0]                 # local index into vi (0-based)
        for jj in range(1, len(vi) - 1):
            vj = vi[jj]                     # 1-based global vertex index
            vj0 = vj - 1                    # 0-based for V indexing
            if 0 <= vj0 < len(V) and len(V[vj0]['f']) > 1:
                pin_local_idx.append(jj)
        pin_local_idx.append(len(vi) - 1)

        # Build reduced fiber with only pinned vertices (keep 1-based for trimxfv)
        fred_v = [vi[j] for j in pin_local_idx]
        fred_dict: Dict = {'v': fred_v}
        if 'r' in fiber:
            fred_dict['r'] = fiber['r']
        Fred.append(fred_dict)

        # X positions for this fiber (convert to 0-based for indexing)
        Xi = X[vi0, :]
        pins_arr = np.array(pin_local_idx, dtype=int)

        AL, AR = bestcurv(Xi, pins_arr, lam)
        AL_list.append(AL)
        AR_list.append(AR)
        Xcrit_list.append(X[[v - 1 for v in fred_v], :])   # pinned coords

    # ---- Pass 2: add interpolation nodes to X ----
    N_verts = len(X)
    # We'll accumulate new rows in a list to avoid repeated vstack
    X_extra: List[np.ndarray] = []

    for i in range(len(F)):
        Xcrit = Xcrit_list[i]
        AL = AL_list[i]
        AR = AR_list[i]

        for j in range(len(Xcrit) - 1, 0, -1):
            # Working *backwards* to match MATLAB's index arithmetic
            X1 = Xcrit[j - 1, :]
            X2 = Xcrit[j,     :]

            # Evaluate at 10 points just to measure arc length
            x_pts, y_pts, z_pts = plotbeam(
                np.vstack([X1, X2]),
                AL[j - 1: j, :],
                AR[j - 1: j, :],
                N=10,
            )
            d = np.sum(len3d(x_pts, y_pts, z_pts))

            if d > minspace:
                n = int(round(d / minspace))
                x_pts, y_pts, z_pts = plotbeam(
                    np.vstack([X1, X2]),
                    AL[j - 1: j, :],
                    AR[j - 1: j, :],
                    N=n + 2,
                )
                if len(x_pts) > 1:
                    # Interior interpolation nodes (exclude first & last)
                    new_rows = np.column_stack([
                        x_pts[1: n + 1],
                        y_pts[1: n + 1],
                        z_pts[1: n + 1],
                    ])
                    new_start = N_verts              # 0-based: first new row is at X[N_verts]
                    new_end   = N_verts + n - 1

                    # Insert new 0-based indices into Fred[i].v
                    fred_v = Fred[i]['v']
                    idx_before = fred_v[j - 1]       # pinned vertex before
                    idx_after  = fred_v[j]            # pinned vertex after
                    pos_before = fred_v.index(idx_before)
                    pos_after  = fred_v.index(idx_after)
                    # Splice new indices between pos_before and pos_after
                    new_indices = list(range(new_start, new_end + 1))
                    Fred[i]['v'] = (
                        fred_v[:pos_before + 1]
                        + new_indices
                        + fred_v[pos_after:]
                    )

                    X_extra.append(new_rows)
                    N_verts += n

    # Concatenate all new vertices
    if X_extra:
        X = np.vstack([X, np.vstack(X_extra)])

    # Compact with trimxfv
    X_out, F_out, V_out = trimxfv(X, Fred, V)
    return X_out, F_out, V_out
