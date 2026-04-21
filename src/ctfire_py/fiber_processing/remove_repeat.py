"""
remove_repeat - clean up duplicate / overlapping fiber vertices.

Faithful port of MATLAB ``remove_repeat.m`` from
``curvelets/src/FIRE/fiberproc``. MATLAB calls this function once at the top
of ``fiberproc.m`` and again after every one of the 5 ``fiberlink``
iterations. The C++ ``process_fibers`` does not perform any equivalent step,
so the Python pipeline invokes it once after ``process_fibers`` returns.

The MATLAB algorithm, in pseudocode, is::

    repeat
        # (a) merge coincident-position vertices (X hash to integer grid)
        # (b) dedupe consecutive repeats inside each fiber
        # (c) if F.v[0] appears elsewhere in F.v, drop F.v[0]; same for F.v[-1]
        # (d) trimxfv(X, F, [], R)   -- rebuilds V from F
        # (e) for every vertex shared by >1 fiber, if fiber fj overlaps fi
        #     on >=2 shared vertices, replace fj with the two pieces of fj
        #     lying outside the overlap interval and keep fi as-is
        # (f) trimxfv(X, F, V, R)
    until len(F) stops changing

Notes on MATLAB quirks that are reproduced here:
* ``remove_repeat.m`` has a typo that leaves ``thresh_emerge`` at Inf
  whenever the function is called with the standard 4 positional args
  (both ``if nargin<5`` branches fire). We therefore default to
  ``thresh_emerge = float('inf')`` so the overlap-split gate is always open.
* MATLAB's ``intersect`` returns the *first* position of each unique shared
  value in ``vj``; duplicates are ignored. We replicate that with a
  seen-set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _dedupe_consecutive(v: List[int]) -> List[int]:
    if len(v) <= 1:
        return list(v)
    out = [v[0]]
    for x in v[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _drop_endpoint_if_in_middle(v: List[int]) -> List[int]:
    """Replicates MATLAB's ``F(i).v(1) = []`` / ``F(i).v(end) = []`` blocks.

    If the first element appears anywhere else in ``v`` the MATLAB code drops
    index 1 (one element). The same rule is then applied to the last element
    against the (already mutated) vector.
    """
    if len(v) < 2:
        return list(v)
    vv = list(v)
    ve1 = vv[0]
    if any(x == ve1 for x in vv[1:]):
        vv = vv[1:]
    if len(vv) < 2:
        return vv
    ve2 = vv[-1]
    if any(x == ve2 for x in vv[:-1]):
        vv = vv[:-1]
    return vv


def _hash_vertex_coords(X: np.ndarray) -> np.ndarray:
    """Return integer hash per vertex matching MATLAB's ``sub2ind`` scheme.

    MATLAB code::

        Xr  = max(ceil(X/max(X(:))*1000), 1);   % per-coord bin in 1..1000
        s   = max(Xr);                          % per-column max
        ind = sub2ind(s, Xr(:,1), Xr(:,2), Xr(:,3));
    """
    if X.size == 0:
        return np.zeros(0, dtype=np.int64)
    X = np.asarray(X, dtype=float)
    max_val = float(np.max(X)) if np.max(X) > 0 else 1.0
    Xr = np.maximum(np.ceil(X / max_val * 1000.0), 1.0).astype(np.int64)
    s = Xr.max(axis=0).astype(np.int64)
    # sub2ind with size s, indices 1-based -> flat index is
    # (Xr[:,0]-1) + (Xr[:,1]-1)*s[0] + (Xr[:,2]-1)*s[0]*s[1] + ...
    flat = np.zeros(X.shape[0], dtype=np.int64)
    stride = 1
    for d in range(Xr.shape[1]):
        flat += (Xr[:, d] - 1) * stride
        stride *= max(int(s[d]), 1)
    return flat


def _coincident_vertex_pairs(X: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of 0-based vertex-index pairs that hash to the same grid cell.

    Matches MATLAB's "same = [idsort(idiff) idsort(idiff+1)]" pairing: each
    consecutive pair inside a tied group is captured.
    """
    if X.shape[0] < 2:
        return []
    flat = _hash_vertex_coords(X)
    idsort = np.argsort(flat, kind="stable")
    isort = flat[idsort]
    diff_zero = np.diff(isort) == 0
    left = idsort[:-1][diff_zero]
    right = idsort[1:][diff_zero]
    return list(zip(left.tolist(), right.tolist()))


def _merge_coincident_vertices(
    F: List[Dict], V: List[Dict], same_pairs: List[Tuple[int, int]]
) -> None:
    """Replace every occurrence of ``vi`` (1-based) with ``vj`` in all fibers.

    ``F`` is mutated in place. ``V`` is used only to locate the fibers that
    reference ``vi``; it is not repaired here (the following ``trimxfv``
    rebuild takes care of V).
    """
    for vi_0, vj_0 in same_pairs:
        if vi_0 < 0 or vi_0 >= len(V):
            continue
        vi_1, vj_1 = vi_0 + 1, vj_0 + 1
        for f_1 in V[vi_0].get("f", []):
            f_0 = int(f_1) - 1
            if f_0 < 0 or f_0 >= len(F):
                continue
            vs = F[f_0].get("v", [])
            if not vs:
                continue
            F[f_0]["v"] = [vj_1 if int(x) == vi_1 else int(x) for x in vs]


def _unique_shared_positions_in_vj(
    vj: List[int], vi_set: set
) -> List[int]:
    """Equivalent to MATLAB ``[vshare, ij] = intersect(vj, vi)``.

    Returns the 0-based positions in ``vj`` of the first occurrence of each
    unique value that also appears in ``vi_set``.
    """
    seen = set()
    positions = []
    for k, x in enumerate(vj):
        xi = int(x)
        if xi in vi_set and xi not in seen:
            positions.append(k)
            seen.add(xi)
    return positions


def _split_overlapping_fibers(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    thresh_emerge: float,
) -> None:
    """Step (e) of MATLAB remove_repeat.

    For every vertex that belongs to >1 fiber, walk the pairwise combinations
    of fibers at that vertex. If a pair shares >=2 distinct vertex ids, blank
    the second fiber (``fj``) and append 0 / 1 / 2 new fibers containing the
    portions of ``fj`` that lie outside the shared interval.
    """
    new_fibers: List[Dict] = []
    cleared: set = set()

    for ii in range(len(V)):
        fiber_ids = V[ii].get("f", [])
        if len(fiber_ids) <= 1:
            continue
        ids = [int(x) - 1 for x in fiber_ids]
        for i in range(len(ids) - 1):
            fi_0 = ids[i]
            if fi_0 in cleared or fi_0 < 0 or fi_0 >= len(F):
                continue
            vi_list = F[fi_0].get("v", [])
            if len(vi_list) < 2:
                continue
            vi_set = {int(x) for x in vi_list}

            for j in range(i + 1, len(ids)):
                fj_0 = ids[j]
                if fj_0 in cleared or fj_0 == fi_0:
                    continue
                if fj_0 < 0 or fj_0 >= len(F):
                    continue
                vj_list = F[fj_0].get("v", [])
                if len(vj_list) < 2:
                    continue

                ij_positions = _unique_shared_positions_in_vj(vj_list, vi_set)
                if len(ij_positions) < 2:
                    continue

                min_pos = min(ij_positions)
                max_pos = max(ij_positions)
                v1 = int(vj_list[min_pos])
                v2 = int(vj_list[max_pos])
                if v1 - 1 < 0 or v2 - 1 < 0 or v1 - 1 >= X.shape[0] \
                        or v2 - 1 >= X.shape[0]:
                    continue
                if float(np.linalg.norm(X[v1 - 1] - X[v2 - 1])) >= thresh_emerge:
                    continue

                # MATLAB: if min(ij) > 1, F(end+1).v = vj(1:min(ij));
                if min_pos > 0:
                    new_f: Dict = {
                        "v": [int(x) for x in vj_list[: min_pos + 1]]
                    }
                    if "r" in F[fj_0]:
                        new_f["r"] = F[fj_0]["r"]
                    new_fibers.append(new_f)
                # MATLAB: if max(ij) < length(vj), F(end+1).v = vj(max(ij):end);
                if max_pos < len(vj_list) - 1:
                    new_f = {"v": [int(x) for x in vj_list[max_pos:]]}
                    if "r" in F[fj_0]:
                        new_f["r"] = F[fj_0]["r"]
                    new_fibers.append(new_f)
                # MATLAB: F(fj).v = [];
                F[fj_0]["v"] = []
                cleared.add(fj_0)

    if new_fibers:
        F.extend(new_fibers)


def remove_repeat(
    X: np.ndarray,
    F: List[Dict],
    V: Optional[List[Dict]],
    R: Optional[np.ndarray],
    thresh_emerge: float = float("inf"),
    max_iters: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[Dict], List[Dict], Optional[np.ndarray]]:
    """Faithful port of MATLAB ``remove_repeat`` (see module docstring).

    Parameters
    ----------
    X, F, V, R
        Vertex positions, fiber list, vertex list and radius array, in the
        same conventions as the rest of the Python ctfire_py package
        (1-based ids inside F.v and V.f).
    thresh_emerge : float
        MATLAB default is effectively ``Inf`` (see module docstring).
    max_iters : int
        Safety bound on the outer ``while`` loop.
    verbose : bool
        If True, print per-iteration fiber counts.

    Returns
    -------
    (X, F, V, R) after cleanup and a final ``trimxfv``.
    """
    from ctfire_py.utils import trimxfv

    # Ensure we have a V structure to start with; MATLAB code happily accepts
    # [] and trimxfv in step (d) regenerates it. Our Python trimxfv only
    # reconstructs V when V is a (possibly empty) list, so we pass [] here.
    if V is None:
        X, F, V, R = trimxfv(X, F, [], R)

    len_prev = -1
    len_curr = len(F)
    iters = 0
    while len_prev != len_curr and iters < max_iters:
        len_prev = len_curr
        iters += 1

        # (a) merge coincident-position vertices
        same_pairs = _coincident_vertex_pairs(X)
        if same_pairs:
            _merge_coincident_vertices(F, V, same_pairs)

        # (b) dedupe consecutive repeats
        for i in range(len(F)):
            v = F[i].get("v", [])
            if len(v) >= 2:
                F[i]["v"] = _dedupe_consecutive([int(x) for x in v])

        # (c) drop endpoint vertices that appear in the middle
        for i in range(len(F)):
            v = F[i].get("v", [])
            if len(v) >= 2:
                F[i]["v"] = _drop_endpoint_if_in_middle([int(x) for x in v])

        # (d) rebuild V via trimxfv with empty V (pass [] so V is rebuilt).
        X, F, V, R = trimxfv(X, F, [], R)

        # (e) split fibers that share >= 2 vertices
        _split_overlapping_fibers(X, F, V, thresh_emerge)

        # (f) trimxfv on the (now possibly blanked / appended) F
        X, F, V, R = trimxfv(X, F, V, R)

        len_curr = len(F)
        if verbose:
            n_coinc = len(_coincident_vertex_pairs(X))
            print(f"  remove_repeat iter {iters}: {len_prev} -> {len_curr} fibers, "
                  f"coincident_vertex_pairs_remaining={n_coinc}")

    return X, F, V, R
