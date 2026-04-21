"""
fiberremove - remove small short fibers.

Port of MATLAB ``fiberremove.m`` from curvelets/src/FIRE/fiberproc. It is
the final clean-up step inside ``fiberproc`` that removes:

* Short fibers (length <= ``thresh_flen``) whose vertices touch at most one
  cross-link - these are dangling fragments.
* A star/triangle pattern: two short fibers that both connect to the same
  pair of "third-party" fibers along their whole length.

The C++ ``process_fibers`` entry point does not call ``fiberremove`` yet, so
``fire_2d_angle`` invokes this Python implementation immediately afterwards
to match MATLAB's ``fiberproc.m`` pipeline end-to-end.

MATLAB reference (``fiberremove.m``)::

    for fi = length(F):-1:1
        if Len(fi) <= thresh_len
            %% dangler check
            if #cross-link vertices of fi <= 1
                remove fi
            %% star/triangle check
            if fi has exactly 2 neighbour fibers f2, f3
                if f2 has only {fi, f3} as neighbors and Len(f2) <= thresh_len
                    remove fi, f2
                elseif f3 has only {fi, f2} as neighbors and Len(f3) <= thresh_len
                    remove fi, f3
    F(fremove) = []
    [X F V R] = trimxfv(X, F, V, R)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _fiber_length(X: np.ndarray, v_list_1b: List[int]) -> float:
    if len(v_list_1b) < 2:
        return 0.0
    idx = np.asarray(v_list_1b, dtype=int) - 1
    idx = idx[(idx >= 0) & (idx < X.shape[0])]
    if idx.size < 2:
        return 0.0
    diffs = np.diff(X[idx].astype(float), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _neighbor_fibers(fi_0b: int, F: List[Dict], V: List[Dict]) -> List[int]:
    """Return 0-based fiber indices that share a vertex with fiber ``fi_0b``.

    Silently drops indices that point outside ``F`` / ``V``. Stale fiber ids
    can linger inside ``V[v]['f']`` when the caller hasn't rerun trimxfv yet
    (e.g. after a previous ``fiberlinkgap`` merge); they shouldn't crash the
    cleanup pass.
    """
    if fi_0b < 0 or fi_0b >= len(F):
        return []
    neighbors = set()
    for v_1b in F[fi_0b].get("v", []):
        v_0b = int(v_1b) - 1
        if v_0b < 0 or v_0b >= len(V):
            continue
        for f_1b in V[v_0b].get("f", []):
            f_0b = int(f_1b) - 1
            if f_0b == fi_0b or f_0b < 0 or f_0b >= len(F):
                continue
            neighbors.add(f_0b)
    return sorted(neighbors)


def _count_crosslink_vertices(fi_0b: int, F: List[Dict], V: List[Dict]) -> int:
    """Count distinct vertices of fiber ``fi_0b`` that are on >1 fibers."""
    if fi_0b < 0 or fi_0b >= len(F):
        return 0
    unique_xlinks = set()
    for v_1b in F[fi_0b].get("v", []):
        v_0b = int(v_1b) - 1
        if v_0b < 0 or v_0b >= len(V):
            continue
        if len(V[v_0b].get("f", [])) > 1:
            unique_xlinks.add(v_0b)
    return len(unique_xlinks)


def fiberremove(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: Optional[np.ndarray],
    thresh_flen: float = 15.0,
    thresh_numv: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[Dict], List[Dict], Optional[np.ndarray]]:
    """
    Remove short fibers that are danglers or redundant parallel pairs.

    Mirrors the behavior of MATLAB's ``fiberremove.m``. Vertex and fiber
    indices inside ``F`` / ``V`` stay 1-based throughout, matching the rest
    of the Python port.

    Parameters
    ----------
    X : np.ndarray
        Vertex coordinates, shape (N, >=2).
    F : List[Dict]
        Fiber list; each entry has at least ``'v'`` (1-based vertex ids).
        If ``'len'`` is present it's used; otherwise length is recomputed.
    V : List[Dict]
        Vertex list; each entry has ``'f'`` (1-based fiber ids).
    R : np.ndarray, optional
        Radii at each vertex.
    thresh_flen : float
        Length threshold below which a fiber is eligible for removal.
        Matches ``p.thresh_flen`` in MATLAB.
    thresh_numv : int
        Retained for API parity with MATLAB; the reference implementation
        comments out the vertex-count check, so this argument is currently
        unused.
    verbose : bool
        If True, print the number of fibers removed.

    Returns
    -------
    X, F, V, R : same types as inputs, after trimxfv.
    """
    from ctfire_py.utils import trimxfv

    n_fibers = len(F)
    if n_fibers == 0:
        return X, F, V, R

    # Precompute fiber lengths (use the cached 'len' field when available).
    lengths = np.zeros(n_fibers, dtype=float)
    for fi in range(n_fibers):
        if isinstance(F[fi], dict) and "len" in F[fi]:
            lengths[fi] = float(F[fi]["len"])
        else:
            lengths[fi] = _fiber_length(X, F[fi].get("v", []))

    to_remove = set()

    # Process fibers from last to first (matches MATLAB "for fi=length(F):-1:1").
    # The reverse order only matters for deterministic tie-breaking since we
    # reconcile removals in a set.
    for fi in range(n_fibers - 1, -1, -1):
        if fi in to_remove:
            continue
        if lengths[fi] > thresh_flen:
            continue

        # Dangler check: does fi touch at most one cross-link vertex?
        n_xlinks = _count_crosslink_vertices(fi, F, V)
        if n_xlinks <= 1:
            to_remove.add(fi)
            # MATLAB falls through to the star-pattern block even after the
            # dangler rule fires; to_remove is a set so that's fine.

        # Star-pattern check: fi has exactly two neighbor fibers that mutually
        # connect through fi and at least one of them is also short.
        fconn = _neighbor_fibers(fi, F, V)
        if len(fconn) == 2:
            f2, f3 = fconn
            fconn2 = _neighbor_fibers(f2, F, V)
            fconn3 = _neighbor_fibers(f3, F, V)
            if len(fconn2) == 2 and f3 in fconn2 and lengths[f2] <= thresh_flen:
                to_remove.add(fi)
                to_remove.add(f2)
            elif len(fconn3) == 2 and f2 in fconn3 and lengths[f3] <= thresh_flen:
                to_remove.add(fi)
                to_remove.add(f3)

    if verbose:
        print(f"  fiberremove: removing {len(to_remove)} short fibers "
              f"({len(to_remove) / max(n_fibers, 1):.1%} of {n_fibers})")

    if to_remove:
        keep = [F[i] for i in range(n_fibers) if i not in to_remove]
        F = keep

    # trimxfv renumbers vertex / fiber indices after the removals.
    if V is not None:
        return trimxfv(X, F, V, R)
    return trimxfv(X, F, V, R)
