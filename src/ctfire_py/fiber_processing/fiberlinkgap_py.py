"""
fiberlinkgap_py - Python port of MATLAB ``fiberlinkgap`` (the sub-function
defined inside ``fiberproc.m``).

The C++ ``fiberproc_native.cpp`` contains a reasonable port of the same
routine but is coupled with a bundled 5-iteration greedy ``fiberlink`` pass,
so it cannot be invoked on its own. MATLAB's order is::

    trimxfv -> remove_repeat -> (fiberlink + remove_repeat)*5 ->
    fiberlinkgap -> fiberremove

We need to reproduce that exact sequence in Python, which means we need a
stand-alone ``fiberlinkgap`` that can be called just once after the Python
fiberlink cleanup converges.

The MATLAB reference (fiberproc.m lines 395-529) operates like this:

1. For every fiber, cache ``F(fi).pos`` and ``F(fi).dir`` at both ends
   (using ``getvect`` with ``sp = anglecomp_space``).
2. Pass 1 - scout for fuse pairs:
   for each fiber end (fi, j in {1, 2}), find every other fiber endpoint
   within ``thresh_linkd`` that is NOT already in fi's connectivity list.
   For each candidate (fek, m), score it with the *worst* of two dot
   products:
     - angle between the two fiber tangents at their ends
     - angle between fi's tangent and the straight connecting segment
   Pick the candidate with the smallest max; if its score is < thresh_linka
   AND neither fi's end j nor fek's end m has been claimed yet, mark the
   pair for fusion.
3. Pass 2 - fuse: walk the fuse list in order and call ``mergefiber_sep``
   on each pair. When fiber f2 is absorbed into f1, rewrite any remaining
   entries in the fuse list that still reference f2.
4. ``trimxfv`` the network.

Note on thresholds: MATLAB's ``thresh_linka`` here is the same *cosine*
threshold used by ``fiberlink`` (default -0.866 = cos(150 deg)), but the
score compared against it is ``max(cos of tangent angles, cos of link
direction)`` so the effective gate is "both tangent pair and link-segment
must be colinear enough".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit_vector(
    X: np.ndarray, v_from_1b: int, v_to_1b: int
) -> np.ndarray:
    if v_from_1b <= 0 or v_to_1b <= 0:
        return np.zeros(X.shape[1], dtype=float)
    if v_from_1b > X.shape[0] or v_to_1b > X.shape[0]:
        return np.zeros(X.shape[1], dtype=float)
    vec = X[v_to_1b - 1].astype(float) - X[v_from_1b - 1].astype(float)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return np.zeros(X.shape[1], dtype=float)
    return vec / n


def _end_direction(
    X: np.ndarray, vi_1b: int, fv_1b: List[int], sp: int
) -> np.ndarray:
    """MATLAB ``getvect``: unit vector from ``vi`` inward along the fiber."""
    n = len(fv_1b)
    if n < 2:
        return np.zeros(X.shape[1], dtype=float)
    if fv_1b[0] == vi_1b:
        ii = min(sp, n - 1)
        return _unit_vector(X, vi_1b, fv_1b[ii])
    if fv_1b[-1] == vi_1b:
        ii = max(0, n - 1 - sp)
        return _unit_vector(X, vi_1b, fv_1b[ii])
    return np.zeros(X.shape[1], dtype=float)


def _mergefiber_sep(
    F: List[Dict], f1_0b: int, e1: int, f2_0b: int, e2: int
) -> None:
    """Replicates MATLAB ``mergefiber_sep`` (fuse without a shared vertex).

    ``e1 / e2`` are 1 or 2 selecting the start or end of each fiber.
    The merged sequence is written back to F[f1]; F[f2] is emptied.
    """
    fiber1 = list(F[f1_0b].get("v", []))
    fiber2 = list(F[f2_0b].get("v", []))
    if not fiber1 or not fiber2:
        return
    if e1 == 1 and e2 == 1:
        fmerge = list(reversed(fiber2)) + fiber1
    elif e1 == 1 and e2 == 2:
        fmerge = fiber2 + fiber1
    elif e1 == 2 and e2 == 1:
        fmerge = fiber1 + fiber2
    elif e1 == 2 and e2 == 2:
        fmerge = fiber1 + list(reversed(fiber2))
    else:
        return
    F[f1_0b]["v"] = [int(x) for x in fmerge]
    F[f2_0b]["v"] = []


def fiberlinkgap(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: Optional[np.ndarray],
    sp: int,
    thresh_linkd: float,
    thresh_linka: float,
) -> Tuple[np.ndarray, List[Dict], List[Dict], Optional[np.ndarray]]:
    """MATLAB-faithful fiberlinkgap.

    Parameters mirror MATLAB signature order: ``sp = p.s_fiberdir``,
    ``thresh_linkd = p.thresh_linkd``, ``thresh_linka = p.thresh_linka``.
    The extra ``R`` argument is piped through unchanged so the caller can
    keep radii aligned via the trailing ``trimxfv``.
    """
    from ctfire_py.utils import trimxfv

    n = len(F)
    if n == 0:
        return trimxfv(X, F, V, R)

    X_arr = np.asarray(X, dtype=float)

    # Gather endpoint vertex ids and tangent directions for every fiber.
    # Matches MATLAB's "for fi=1:length(F): F(fi).pos/dir" preamble.
    end_v: List[Tuple[int, int]] = []
    end_dir: List[Tuple[np.ndarray, np.ndarray]] = []
    end_pos: List[Tuple[np.ndarray, np.ndarray]] = []
    for fi in range(n):
        v = F[fi].get("v", [])
        if len(v) < 2:
            end_v.append((-1, -1))
            end_dir.append((np.zeros(X_arr.shape[1]), np.zeros(X_arr.shape[1])))
            end_pos.append((np.zeros(X_arr.shape[1]), np.zeros(X_arr.shape[1])))
            continue
        v1, v2 = int(v[0]), int(v[-1])
        end_v.append((v1, v2))
        end_dir.append(
            (_end_direction(X_arr, v1, v, sp),
             _end_direction(X_arr, v2, v, sp))
        )
        end_pos.append(
            (X_arr[v1 - 1].astype(float), X_arr[v2 - 1].astype(float))
        )

    # Spatial index on endpoint positions.
    endpoints = []
    endpoint_owner: List[Tuple[int, int]] = []  # (fiber_0b, end_side 1 or 2)
    for fi in range(n):
        if end_v[fi] == (-1, -1):
            continue
        endpoints.append(end_pos[fi][0])
        endpoint_owner.append((fi, 1))
        endpoints.append(end_pos[fi][1])
        endpoint_owner.append((fi, 2))
    if not endpoints:
        return trimxfv(X, F, V, R)
    ep_arr = np.vstack(endpoints)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ep_arr)
    except Exception:
        tree = None

    # Pass 1 - scout for fuse pairs.
    # MATLAB tracks fuseflag[(fi, end)] so no fiber end ends up in two fuses.
    fuseflag = np.zeros((n, 2), dtype=bool)
    fuse: List[Tuple[int, int, int, int]] = []  # (f1_0b, e1, f2_0b, e2)

    for fi in range(n):
        if end_v[fi] == (-1, -1):
            continue

        # fconnect: 1-based fiber ids that currently touch any vertex in fi
        fconnect: set = set()
        for vi_1b in F[fi].get("v", []):
            vi_0 = int(vi_1b) - 1
            if 0 <= vi_0 < len(V):
                for f_1b in V[vi_0].get("f", []):
                    fconnect.add(int(f_1b))

        for j in [1, 2]:
            if fuseflag[fi, j - 1]:
                continue
            xj = end_pos[fi][j - 1]
            vectj = end_dir[fi][j - 1]
            if np.linalg.norm(vectj) == 0:
                continue

            # Candidate endpoints within thresh_linkd of this fiber end.
            if tree is not None:
                cand_idx = tree.query_ball_point(xj, r=thresh_linkd)
            else:
                d = np.linalg.norm(ep_arr - xj[None, :], axis=1)
                cand_idx = np.where(d <= thresh_linkd)[0].tolist()

            best_score = np.inf
            best_target: Optional[Tuple[int, int]] = None  # (fk_0b, m)
            seen: set = set()
            for ci in cand_idx:
                fk_0b, m = endpoint_owner[ci]
                if fk_0b == fi:
                    continue
                if fk_0b <= fi:
                    # MATLAB's "if fek > fi" gate: only consider higher-indexed
                    # pairs so each pair is evaluated once.
                    continue
                key = (fk_0b, m)
                if key in seen:
                    continue
                seen.add(key)
                # Skip if fk is already directly connected via a shared vertex.
                if (fk_0b + 1) in fconnect:
                    continue
                if fuseflag[fk_0b, m - 1]:
                    continue
                vect_k = end_dir[fk_0b][m - 1]
                if np.linalg.norm(vect_k) == 0:
                    continue
                # Angle between fiber tangents (both pointing inward along
                # their respective fibers - MATLAB's dotcalc is just np.dot).
                a_tangent = float(np.dot(vectj, vect_k))

                xk = end_pos[fk_0b][m - 1]
                link_vec = xk - xj
                link_n = float(np.linalg.norm(link_vec))
                if link_n < 1e-12:
                    continue
                link_unit = link_vec / link_n
                a_link = float(np.dot(vectj, link_unit))

                # MATLAB: angle = max(a(:, 3:4), [], 2); picks the worse of
                # the two colinearity scores. Cosine close to -1 is best.
                score = max(a_tangent, a_link)
                if score < best_score:
                    best_score = score
                    best_target = (fk_0b, m)

            if best_target is not None and best_score < thresh_linka:
                fk_0b, m = best_target
                if not fuseflag[fi, j - 1] and not fuseflag[fk_0b, m - 1]:
                    fuseflag[fi, j - 1] = True
                    fuseflag[fk_0b, m - 1] = True
                    fuse.append((fi, j, fk_0b, m))

    # Pass 2 - apply fuses in order, rewriting later references when f2 is
    # absorbed into f1. MATLAB lines 488-525.
    for i, (f1_0b, e1, f2_0b, e2) in enumerate(fuse):
        if not F[f1_0b].get("v") or not F[f2_0b].get("v"):
            continue
        _mergefiber_sep(F, f1_0b, e1, f2_0b, e2)
        # Rewrite any remaining (f2, *) entries to point at f1.
        for k in range(i + 1, len(fuse)):
            a1, a2, b1, b2 = fuse[k]
            if a1 == f2_0b:
                fuse[k] = (f1_0b, e1, b1, b2)
            if b1 == f2_0b:
                a1b, a2b, _, _ = fuse[k]
                fuse[k] = (a1b, a2b, f1_0b, e1)

    return trimxfv(X, F, V, R)
