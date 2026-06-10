"""
fiberlink_py - faithful Python port of MATLAB ``fiberlink`` (inside
``fiberproc.m``).

MATLAB's implementation differs from the C++ port in one important way: when
``n > 2`` fibers share an endpoint vertex, MATLAB computes the full pairwise
cosine matrix ``A`` and then picks the **minimum-cosine** pair (most
colinear) via ``min2(A)``, merges them, zeroes out the rows/columns of the
merged fibers, and repeats until the minimum rises above ``thresha``. The
C++ version walks pairs in index order and merges the first pair with
``a < thresha`` encountered, which misses better merges available at busy
junctions.

Because the C++ ``process_fibers`` already ran 5 iterations of its greedy
``fiberlink``, we only need to catch the **residual** merges it missed.
Running this Python port once after ``remove_repeat`` accomplishes that.

Inputs / outputs follow the rest of the Python ctfire_py conventions:
* ``X`` is an ``(N, 3)`` float array of vertex coordinates.
* ``F`` is a list of dicts with a 1-based ``'v'`` vertex-id list.
* ``V`` is a list of dicts with ``'f'``, ``'fe'``, ``'vall'`` fields
  (1-based fiber ids).
* ``R`` is an optional ``(N,)`` radius array (may be ``None``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _fiber_direction(
    X: np.ndarray, vi_1b: int, fv_1b: List[int], sp: int
) -> np.ndarray:
    """Direction vector pointing along the fiber starting at endpoint ``vi``.

    Mirrors MATLAB ``getvect``: takes the ``sp``-th vertex from the end
    where ``vi`` sits and computes the unit vector from ``vi`` to it.
    """
    n = len(fv_1b)
    if n < 2:
        return np.zeros(3, dtype=float)
    if fv_1b[0] == vi_1b:
        ii = min(sp, n - 1)
        vj_1b = fv_1b[ii]
    elif fv_1b[-1] == vi_1b:
        ii = max(0, n - 1 - sp)
        vj_1b = fv_1b[ii]
    else:
        return np.zeros(3, dtype=float)
    vect = X[vj_1b - 1] - X[vi_1b - 1]
    norm = float(np.linalg.norm(vect))
    if norm < 1e-12:
        return np.zeros(3, dtype=float)
    return vect / norm


def _merge_fibers_at_shared_vertex(
    F: List[Dict], V: List[Dict], f1_0b: int, f2_0b: int
) -> None:
    """Replicates MATLAB ``mergefiber`` (the shared-vertex variant).

    Concatenates F[f2] into F[f1] such that they meet at the common vertex,
    blanks F[f2], and keeps the V bookkeeping consistent so the outer loop
    keeps working.
    """
    f1 = F[f1_0b]["v"]
    f2 = F[f2_0b]["v"]
    if not f1 or not f2:
        return
    a1, z1 = f1[0], f1[-1]
    a2, z2 = f2[0], f2[-1]

    if a1 == a2:
        fmerge = list(reversed(f2[1:])) + list(f1)
        vm = a1
        ve = z2
    elif a1 == z2:
        fmerge = list(f2[:-1]) + list(f1)
        vm = a1
        ve = a2
    elif z1 == a2:
        fmerge = list(f1) + list(f2[1:])
        vm = z1
        ve = z2
    elif z1 == z2:
        fmerge = list(f1) + list(reversed(f2[:-1]))
        vm = z1
        ve = a2
    else:
        return  # fibers don't share an endpoint -> nothing to do

    F[f1_0b]["v"] = [int(x) for x in fmerge]
    F[f2_0b]["v"] = []

    # Maintain V.fe for the remaining outer iterations. Indices are 1-based.
    def _drop(lst: List[int], val: int) -> List[int]:
        return [x for x in lst if x != val]

    vm_0 = vm - 1
    ve_0 = ve - 1
    if 0 <= vm_0 < len(V):
        V[vm_0]["fe"] = sorted(set(
            _drop(_drop(V[vm_0]["fe"], f1_0b + 1), f2_0b + 1)
        ))
    if 0 <= ve_0 < len(V):
        fe_ve = _drop(V[ve_0]["fe"], f2_0b + 1)
        fe_ve.append(f1_0b + 1)
        V[ve_0]["fe"] = sorted(set(fe_ve))

    for vi_1 in f2:
        vi_0 = int(vi_1) - 1
        if 0 <= vi_0 < len(V):
            V[vi_0]["f"] = sorted(set(
                _drop(V[vi_0]["f"], f2_0b + 1) + [f1_0b + 1]
            ))
            V[vi_0]["vall"] = sorted(set(
                _drop(V[vi_0]["vall"], 0) + [int(x) for x in fmerge]
            ))


def fiberlink(
    X: np.ndarray,
    F: List[Dict],
    V: List[Dict],
    R: Optional[np.ndarray],
    thresha: float,
    sp: int,
) -> Tuple[np.ndarray, List[Dict], List[Dict], Optional[np.ndarray]]:
    """One pass of MATLAB-faithful fiberlink.

    Iterates every fiber, checks the ``fe`` (fiber-end) list at each
    endpoint, and merges the most colinear pair found at each junction
    via the ``min2(A)`` / while-loop scheme from MATLAB.
    """
    from ctfire_py.utils import trimxfv

    for i in range(len(F)):
        fv = F[i].get("v", [])
        if len(fv) < 2:
            continue
        endpoints = [fv[0], fv[-1]]

        for vi_1b in endpoints:
            vi_0b = int(vi_1b) - 1
            if vi_0b < 0 or vi_0b >= len(V):
                continue
            fe_ids_1b = list(V[vi_0b].get("fe", []))
            n = len(fe_ids_1b)
            if n < 2:
                continue

            if n == 2:
                fj_0b = fe_ids_1b[0] - 1
                fk_0b = fe_ids_1b[1] - 1
                if fj_0b == fk_0b:
                    continue
                if not (0 <= fj_0b < len(F)) or not (0 <= fk_0b < len(F)):
                    continue
                if not F[fj_0b]["v"] or not F[fk_0b]["v"]:
                    continue
                v1 = _fiber_direction(X, vi_1b, F[fj_0b]["v"], sp)
                v2 = _fiber_direction(X, vi_1b, F[fk_0b]["v"], sp)
                a = float(np.dot(v1, v2))
                if a < thresha:
                    _merge_fibers_at_shared_vertex(F, V, fj_0b, fk_0b)
                continue

            # n > 2: compute full pairwise cosine matrix and greedily merge
            # the minimum (most colinear) pair, MATLAB min2-style.
            A = np.full((n, n), np.inf, dtype=float)
            vects = []
            for j in range(n):
                fj_0b = fe_ids_1b[j] - 1
                if 0 <= fj_0b < len(F) and F[fj_0b]["v"]:
                    vects.append(
                        _fiber_direction(X, vi_1b, F[fj_0b]["v"], sp)
                    )
                else:
                    vects.append(np.zeros(3))
            for j in range(n - 1):
                for k in range(j + 1, n):
                    if np.linalg.norm(vects[j]) == 0 or np.linalg.norm(vects[k]) == 0:
                        continue
                    A[j, k] = float(np.dot(vects[j], vects[k]))
            # Loop "while min(A) < thresha": merge best pair, mask it out, repeat.
            while True:
                jl, km = np.unravel_index(np.argmin(A), A.shape)
                a = float(A[jl, km])
                if not np.isfinite(a) or a >= thresha:
                    break
                f1_0b = fe_ids_1b[jl] - 1
                f2_0b = fe_ids_1b[km] - 1
                if (
                    0 <= f1_0b < len(F)
                    and 0 <= f2_0b < len(F)
                    and f1_0b != f2_0b
                    and F[f1_0b]["v"]
                    and F[f2_0b]["v"]
                ):
                    _merge_fibers_at_shared_vertex(F, V, f1_0b, f2_0b)
                A[jl, :] = np.inf
                A[km, :] = np.inf
                A[:, jl] = np.inf
                A[:, km] = np.inf

    return trimxfv(X, F, V, R)
