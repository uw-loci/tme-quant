"""
beamproc - Post-process fiber network for FEA.

Ports MATLAB's beamproc.m and its helpers from
/curvelets/src/FIRE/beamproc/:
  find_boundary.m, find1path.m, remove_floppy_edges.m,
  fiber2beam.m (see fiber2beam.py).

Workflow
--------
1. Scale X coordinates by p['scale'].
2. Identify boundary nodes (B1 near min, B2 near max along blist axes).
3. Build a vertex adjacency list (needed for path finding).
4. Remove "floppy" edges – fibers that don't lie on any path connecting
   the two boundaries.
5. Remove duplicate X nodes.
6. Assign per-fiber mean radius.
7. Interpolate fibers with fiber2beam.
8. Add orientation angles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ctfire_py.fiber_processing.fiber2beam import fiber2beam
from ctfire_py.utils.trimxfv import trimxfv


# ---------------------------------------------------------------------------
# find_boundary
# ---------------------------------------------------------------------------

def find_boundary(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    db: float,
    blist: List[int],
) -> Tuple[List[Dict], List[int], List[int]]:
    """
    Identify boundary vertices (endpoints near the image edges).

    Faithful port of MATLAB find_boundary.m.

    Parameters
    ----------
    X     : (N, 3) float array - vertex coordinates (1-based indexing from F/V)
    F     : fiber list, each with 'v' (1-based vertex indices)
    V     : vertex list
    db    : float - boundary thickness (in the same units as X)
    blist : list of ints - Python column indices into X (i.e. 0=row, 1=col, 2=z)

    Returns
    -------
    V         : updated V with a 'b' field (0=interior, 1=boundary1, 2=boundary2)
    boundary1 : list of 1-based vertex indices near the minimum
    boundary2 : list of 1-based vertex indices near the maximum
    """
    if len(X) == 0:
        for v in V:
            v['b'] = 0
        return V, [], []

    boundary1: Set[int] = set()
    boundary2: Set[int] = set()

    for col in blist:
        bmin = float(X[:, col].min())
        bmax = float(X[:, col].max())

        for fiber in F:
            for v_idx in [fiber['v'][0], fiber['v'][-1]]:  # endpoints only (1-based)
                v_idx0 = v_idx - 1                          # 1-based → 0-based
                b = X[v_idx0, col]
                if b < bmin + db:
                    boundary1.add(v_idx0)
                elif b > bmax - db:
                    boundary2.add(v_idx0)

    # Mark vertices
    b1_list = sorted(boundary1)
    b2_list = sorted(boundary2)
    b1_set = set(b1_list)
    b2_set = set(b2_list)

    for vi, vert in enumerate(V):
        if vi in b1_set:
            vert['b'] = 1
        elif vi in b2_set:
            vert['b'] = 2
        else:
            vert['b'] = 0

    return V, b1_list, b2_list


# ---------------------------------------------------------------------------
# fiber2edge - build condensed edge matrix and neighbour list
# ---------------------------------------------------------------------------

def fiber2edge(
    F: List[Dict],
    V: List[Dict],
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Derive condensed edge list and per-vertex neighbour list from fibers.

    Faithful port of MATLAB fiber2edge.m.  Edges connect only "special"
    vertices along each fiber: fiber endpoints (b==1 after reset) and
    interior crosslink vertices (vertex belongs to >1 unique fiber).
    Interior degree-2 vertices are NOT added to the edge list, giving the
    same condensed (pin-to-pin) graph that MATLAB uses for path finding.

    Parameters
    ----------
    F : fiber list (v = 1-based vertex indices)
    V : vertex list with 'f' field (list of 1-based fiber indices)

    Returns
    -------
    E  : ndarray, shape (n_edges, 2) - 1-based vertex pairs
    Ve : list of vertex dicts with 'v' (1-based neighbours), 'e' (edge ids),
         'f' (fiber ids) added – mirrors MATLAB's Vout struct
    """
    n = len(V)

    # Step 1: reset b, then mark fiber endpoints (b=1)
    for vert in V:
        vert['b'] = 0
    for fiber in F:
        V[fiber['v'][0] ]['b'] = 1
        V[fiber['v'][-1]]['b'] = 1

    # Step 2: initialise output adjacency struct
    Vout: List[Dict] = [{'v': [], 'e': [], 'f': []} for _ in range(n)]

    edges: List[Tuple[int, int]] = []
    ei = 0

    for fi, fiber in enumerate(F):
        v_arr = fiber['v']
        v1: Optional[int] = None  # last special vertex seen along this fiber

        for vj in v_arr:
            # Special = crosslink OR fiber endpoint
            n_unique_fibers = len(set(V[vj].get('f', [vj])))
            is_special = (n_unique_fibers > 1) or (V[vj]['b'] > 0)

            if is_special:
                if v1 is None:
                    v1 = vj
                else:
                    ei += 1
                    v2 = v1
                    v1 = vj
                    edges.append((v1, v2))
                    Vout[v1]['v'].append(v2)
                    Vout[v1]['e'].append(ei)
                    Vout[v1]['f'].append(fi)
                    Vout[v2]['v'].append(v1)
                    Vout[v2]['e'].append(ei)

    E = np.array(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
    return E, Vout


# ---------------------------------------------------------------------------
# find1path - DFS with greedy max-coordinate tie-breaking
# ---------------------------------------------------------------------------

def find1path(
    vstart: int,
    vend_indic: np.ndarray,
    vavoid_indic: np.ndarray,
    X: np.ndarray,
    Ve: List[Dict],
    direction: int = 0,
) -> List[int]:
    """
    Find a single path from ``vstart`` to any vertex where vend_indic is 1.

    Faithful port of MATLAB find1path.m: DFS with greedy tie-breaking –
    at each step the unvisited, un-avoided neighbor with the maximum
    coordinate along ``direction`` is visited first.

    Parameters
    ----------
    vstart       : 0-based start vertex index
    vend_indic   : boolean array, size n, 0-based indexed
    vavoid_indic : boolean array, size n, 0-based indexed
    X            : (N, >=1) vertex coordinates
    Ve           : vertex list with 'v' field (0-based neighbour list)
    direction    : column index of X used for greedy tie-breaking (default 0)

    Returns
    -------
    path : list of 0-based vertex indices, or [] if no path exists
    """
    if vend_indic[vstart]:
        return [vstart]

    n = len(Ve)
    visited = np.zeros(n, dtype=bool)
    path_arr = np.zeros(n + 1, dtype=int)   # path_arr[level] = vertex

    path_arr[1] = vstart
    level = 1
    vcurr = vstart
    visited[vstart] = True

    while level != 0 and not vend_indic[vcurr]:
        neighbors = Ve[vcurr]['v']
        # Unvisited and not avoided (0-based)
        vnext = [v for v in neighbors if not visited[v] and not vavoid_indic[v]]

        if not vnext:
            # Backtrack
            level -= 1
            if level != 0:
                vcurr = path_arr[level]
        else:
            # Greedy: pick neighbor with maximum X along `direction`
            x_vals = X[np.array(vnext, dtype=int), direction]
            jj = int(np.argmax(x_vals))
            vcurr = vnext[jj]
            level += 1
            path_arr[level] = vcurr
            visited[vcurr] = True

    if level == 0:
        return []
    if vend_indic[vcurr]:
        return path_arr[1: level + 1].tolist()
    return []


# ---------------------------------------------------------------------------
# remove_floppy_edges
# ---------------------------------------------------------------------------

def remove_floppy_edges(
    X: np.ndarray,
    F: List[Dict],
    Ve: List[Dict],
    R: np.ndarray,
    B1: List[int],
    B2: List[int],
    direction: int = 0,
) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]:
    """
    Remove edges that don't contribute to network stiffness.

    Faithful port of MATLAB remove_floppy_edges.m (all three passes).

    Pass 1 : mark paths B1 → B2.
    Pass 2 : mark paths B2 → stiff ∪ B1.
    Pass 3 : for each non-stiff interior vertex with neighbours, check
             whether two vertex-disjoint paths to stiff vertices exist;
             if so, mark both paths stiff (biconnectivity criterion).

    Parameters
    ----------
    X         : (N, 3) vertex coordinates
    F         : fiber list
    Ve        : vertex list with 'v' (0-based neighbour list) and 'e' fields
                produced by the condensed ``fiber2edge``
    R         : (N,) per-vertex radii
    B1, B2    : 0-based vertex indices for the two boundaries
    direction : axis index for greedy path tie-breaking (matches p.blist)

    Returns
    -------
    Xr, Fr, Vr, Rr : reduced arrays after trimxfv compaction
    """
    n = len(Ve)
    stiff = np.zeros(n, dtype=int)
    z     = np.zeros(n, dtype=bool)  # empty avoid set

    # Pass 1: B1 → B2; grow vend to include found path vertices
    vend = z.copy()
    if B2:
        vend[np.array(B2, dtype=int)] = True
    for v1 in B1:
        p = find1path(v1, vend, z, X, Ve, direction)
        if p:
            stiff[np.array(p, dtype=int)] = 1
            vend[np.array(p, dtype=int)]  = True

    # Pass 2: B2 → stiff ∪ B1
    vend2 = (stiff == 1).copy()
    if B1:
        vend2[np.array(B1, dtype=int)] = True
    for v1 in B2:
        p = find1path(v1, vend2, z, X, Ve, direction)
        if p:
            stiff[np.array(p, dtype=int)] = 1
            vend2[np.array(p, dtype=int)] = True

    # Pass 3: biconnectivity check (always runs – mirrors MATLAB behaviour)
    stiff_arr  = stiff.astype(bool)
    bound_arr  = z.copy()
    if B1:
        bound_arr[np.array(B1, dtype=int)] = True
    if B2:
        bound_arr[np.array(B2, dtype=int)] = True

    for vi in range(n):
        if stiff[vi] or bound_arr[vi]:
            continue
        if not Ve[vi].get('e'):   # no edges in condensed graph → skip
            continue

        p1 = find1path(vi, stiff_arr, z, X, Ve, direction)
        if not p1:
            continue
        avoid2 = z.copy()
        avoid2[np.array(p1, dtype=int)] = True
        p2 = find1path(vi, stiff_arr, avoid2, X, Ve, direction)
        if p2:
            stiff[np.array(p1, dtype=int)] = 1
            stiff[np.array(p2, dtype=int)] = 1
            stiff_arr = stiff.astype(bool)

    # Trim each fiber to its first..last stiff vertex; drop if < 2 stiff
    vstiff = set(int(v) for v in np.where(stiff == 1)[0])

    F_trimmed: List[Dict] = []
    for fiber in F:
        v = fiber['v']
        si = [1 if vx in vstiff else 0 for vx in v]
        if sum(si) < 2:
            continue
        stiff_pos = [k for k, s in enumerate(si) if s == 1]
        fiber_new = dict(fiber)
        fiber_new['v'] = v[stiff_pos[0]: stiff_pos[-1] + 1]
        F_trimmed.append(fiber_new)

    V_list = [dict(vert) for vert in Ve]
    Xr, Fr, Vr, Rr = trimxfv(X, F_trimmed, V_list, R)
    return Xr, Fr, Vr, Rr


# ---------------------------------------------------------------------------
# remove_repeatX - merge spatially coincident vertices
# ---------------------------------------------------------------------------

def remove_repeatX(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: np.ndarray,
) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]:
    """
    Remove duplicate rows in X (vertices at the same position).

    Faithful port of MATLAB remove_repeatX.m: coordinates are quantised to
    a 1000-step grid via ``ceil(X / max(|X|) * 1000)``, then linear indices
    identify coincident vertices.  Consecutive duplicate vertex ids within a
    fiber are also collapsed before calling trimxfv.

    Parameters
    ----------
    X : (N, 3) vertex coordinates
    F : fiber list (1-based vertex indices)
    V : vertex list (with 'f' field listing fiber membership)
    R : (N,) radii

    Returns
    -------
    X, F, V, R : compacted arrays after merging and trimxfv
    """
    if len(X) == 0:
        return X, F, V, R

    X = np.asarray(X, dtype=float)
    x_max = float(np.max(np.abs(X)))
    if x_max == 0:
        return trimxfv(X, [dict(f) for f in F], [dict(v) for v in V], R)

    # Quantise to integer grid, minimum value 1 (mirrors MATLAB's max(1,...))
    Xq = np.maximum(1, np.ceil(X / x_max * 1000)).astype(int)
    s = Xq.max(axis=0)   # grid dimensions (row vector in MATLAB)

    # Linear indices using Fortran (column-major) order to match MATLAB sub2ind
    ind = np.ravel_multi_index(
        (Xq[:, 0] - 1, Xq[:, 1] - 1, Xq[:, 2] - 1),
        dims=(int(s[0]), int(s[1]), int(s[2])),
        order='F',
    )

    isort  = np.argsort(ind, kind='stable')
    dsort  = np.diff(ind[isort])
    idiff  = np.where(dsort == 0)[0]
    # same: pairs of 0-based indices whose quantised coords coincide
    same = list(zip(isort[idiff], isort[idiff + 1]))

    F_copy = [dict(f) for f in F]
    V_copy = [dict(v) for v in V]

    # Replace vi with vj in all fibers that contain vi
    for vi, vj in same:
        for fj in V_copy[vi].get('f', []):
            fj_idx = fj  # already 0-based
            if fj_idx < len(F_copy):
                F_copy[fj_idx]['v'] = [vj if x == vi else x for x in F_copy[fj_idx]['v']]

    # Remove consecutive duplicate vertex ids within each fiber
    for fiber in F_copy:
        vv = fiber['v']
        if len(vv) < 2:
            continue
        new_v = [vv[0]]
        for k in range(1, len(vv)):
            if vv[k] != new_v[-1]:
                new_v.append(vv[k])
        fiber['v'] = new_v

    return trimxfv(X, F_copy, V_copy, R)


# ---------------------------------------------------------------------------
# add_angle - compute per-fiber orientation angle
# ---------------------------------------------------------------------------

def add_angle(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Append orientation angle to each fiber dict.

    Returns
    -------
    F     : fiber list with 'a' (angle in degrees) added to each fiber
    A     : (n_fibers,) array of angles
    Amap  : same as A (for compatibility with MATLAB signature)
    """
    A = np.zeros(len(F))
    for fi, fiber in enumerate(F):
        v = fiber['v']
        if len(v) < 2:
            A[fi] = 0.0
            continue
        x0, y0 = X[v[0]  - 1, 0], X[v[0]  - 1, 1]   # 1-based → 0-based
        x1, y1 = X[v[-1] - 1, 0], X[v[-1] - 1, 1]
        A[fi] = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        fiber['a'] = A[fi]

    return F, A, A.copy()


# ---------------------------------------------------------------------------
# beamproc - main entry point
# ---------------------------------------------------------------------------

def beamproc(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: np.ndarray,
    p: Dict,
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Post-process a fiber network for FEA: scale, prune floppy edges,
    interpolate, and annotate with angles.

    Faithful port of MATLAB beamproc.m.

    Parameters
    ----------
    X : (N, 3) vertex coordinates
    F : fiber list, each with 'v' (0-based vertex indices)
    V : vertex list
    R : (N,) per-vertex radii
    p : parameter dict with keys:
          'blist'       - MATLAB-convention axis index for boundary detection
                          (1 = x/col, 2 = y/row, 3 = z); converted internally
                          to Python column indices into [row, col, z] X array
          'scale'       - [sx, sy, sz] microns/pixel
          's_boundthick' - boundary thickness in microns (before scaling)
          's_maxspace'  - max interpolation spacing (pixels, before scaling)
          'lambda'      - Tikhonov regularisation for fiber2beam (e.g. 0.01)

    Returns
    -------
    Xr, Fr, Vr : processed arrays
    """
    if not F:
        return X, F, V

    # ---- Scale X ----
    X = np.asarray(X, dtype=float).copy()
    scale = np.asarray(p.get('scale', [1.0, 1.0, 1.0]), dtype=float)
    if X.shape[1] == 2:
        X = np.column_stack([X, np.zeros(len(X))])
    for col in range(min(3, X.shape[1])):
        X[:, col] *= scale[col]

    # ---- Boundary parameters ----
    # MATLAB coordinate convention: X = [x, y, z] where x = image col, y = image row
    # Python  coordinate convention: X = [row, col, z]
    # Mapping: MATLAB blist=1 (x/col) → Python column index 1
    #          MATLAB blist=2 (y/row) → Python column index 0
    #          MATLAB blist=3 (z)     → Python column index 2
    _MATLAB_AXIS_TO_PY_COL = {1: 1, 2: 0, 3: 2}
    blist_raw = p.get('blist', 1)
    if not hasattr(blist_raw, '__iter__') or isinstance(blist_raw, int):
        blist_raw_list = [int(blist_raw)]
    else:
        blist_raw_list = [int(b) for b in blist_raw]
    blist = [_MATLAB_AXIS_TO_PY_COL.get(b, b - 1) for b in blist_raw_list]

    s_boundthick = float(p.get('s_boundthick', 10))
    # MATLAB: db = p.s_boundthick * sc(blist), sc = p.scale = [sx, sy, sz]
    # Use MATLAB's 1-based axis index into the scale array to preserve the same
    # scale factor that MATLAB would select (e.g. blist=1 → scale[0] = sx).
    db = s_boundthick * float(scale[blist_raw_list[0] - 1])

    # ---- Identify boundary ----
    V, B1, B2 = find_boundary(X, F, V, db, blist)

    # ---- Build condensed adjacency (pin-to-pin graph for path finding) ----
    E, Ve = fiber2edge(F, V)

    # ---- Remove floppy edges (all 3 passes, matching MATLAB) ----
    direction = blist[0]                    # prefer motion along boundary axis
    Xr, Fr, Vr, Rr = remove_floppy_edges(X, F, Ve, R, B1, B2, direction)

    # ---- Re-identify boundary after pruning ----
    Vr, B1r, B2r = find_boundary(Xr, Fr, Vr, db, blist)

    # ---- Remove duplicate nodes ----
    Xr, Fr, Vr, Rr = remove_repeatX(Xr, Fr, Vr, Rr)

    # ---- Assign mean radius to each fiber ----
    Rr_scaled = Rr * float(scale[0])
    for fiber in Fr:
        v = fiber['v']
        fiber['r'] = float(np.mean(Rr_scaled[np.array(v)]))

    # ---- Interpolate fibers ----
    # MATLAB passes p.s_maxspace directly after X is already scaled; no
    # additional scale factor is applied here.
    minspace = float(p.get('s_maxspace', 5))
    lam = float(p.get('lambda', 0.01))
    print("    interpolating fibers")
    Xr, Fr, Vr = fiber2beam(Xr, Fr, Vr, Rr, minspace, lam)

    # ---- Add angles ----
    Fr, A, Amap = add_angle(Xr, Fr, Vr)

    return Xr, Fr, Vr
