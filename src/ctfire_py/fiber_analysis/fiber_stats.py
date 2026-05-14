import numpy as np
from typing import Dict, Any, Optional, Tuple


def compute_fiber_straightness(data: Dict[str, Any], FN: np.ndarray) -> np.ndarray:
    """
    Compute per-fiber straightness for the fibers indexed by FN.

    Straightness is defined as the Euclidean distance between a fiber's start and
    end vertex divided by its arc length (always in [0, 1]).

    Parameters
    ----------
    data : dict
        Output dict from fire_2d_angle, containing:
        - ``Xa``: (N, 2) float array of vertex coordinates
        - ``Fa``: list of dicts, each with key ``v`` (list of vertex indices)
        - ``M``: dict with key ``L`` (1-D float array of arc lengths, one per fiber)
    FN : np.ndarray
        1-D integer array of fiber indices (0-based) to compute straightness for.

    Returns
    -------
    straightness : np.ndarray
        1-D float array of straightness values, one entry per fiber in FN.
    """
    Xa = data["Xa"]
    Fa = data["Fa"]
    L = data["M"]["L"]

    fnum = len(Fa)
    dse = np.zeros(fnum)

    for i in range(fnum):
        v = Fa[i]["v"]
        if len(v) < 2:
            dse[i] = 0.0
        else:
            start = Xa[v[0], :]
            end = Xa[v[-1], :]
            dse[i] = np.linalg.norm(end - start)

    # Guard against zero-length fibers
    lengths = L.copy()
    lengths[lengths == 0] = np.nan
    fstr = dse / lengths

    return fstr[FN]


def compute_fiber_widths(
    data: Dict[str, Any],
    FN: np.ndarray,
    wid_th: float,
    wid_opt: int = 1,
    wid_mp: int = 5,
    wid_sigma: float = 1.0,
    wid_max: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute per-fiber average (and optionally maximum) widths for fibers in FN.

    Mirrors MATLAB ctFIRE_1.m lines 791–911.  For each fiber, vertex-level widths
    are ``2 * data['Ra'][v]``.  Points exceeding ``wid_th`` are excluded as
    artifacts.  When ``wid_opt != 1`` and the remaining sample is large enough
    (>= ``wid_mp``), a mean ± ``wid_sigma``·std clipping pass is applied.

    Parameters
    ----------
    data : dict
        Output dict from fire_2d_angle, containing:
        - ``Fa``: list of dicts, each with key ``v`` (list of vertex indices)
        - ``Ra``: 1-D float array of per-vertex radii (half-widths)
    FN : np.ndarray
        1-D integer array of fiber indices (0-based) to compute widths for.
    wid_th : float
        Maximum allowed fiber width; vertices exceeding this are excluded.
    wid_opt : int, optional
        1 = use all passing points; other = apply mean ± sigma clipping. Default 1.
    wid_mp : int, optional
        Minimum sample size required for sigma clipping to activate. Default 5.
    wid_sigma : float, optional
        Number of standard deviations for clipping window. Default 1.0.
    wid_max : int, optional
        0 = return only average widths; 1 = also return maximum widths. Default 0.

    Returns
    -------
    widave : np.ndarray
        1-D float array of average fiber widths, one per fiber in FN.
        Entries are NaN when no vertices survive the width threshold.
    widmax : np.ndarray or None
        1-D float array of maximum fiber widths, one per fiber in FN, or None
        when ``wid_max == 0``.
    """
    Ra = data["Ra"]
    Fa = data["Fa"]
    LFa = len(FN)

    widave_sp = np.full(LFa, np.nan)
    widmax_sp = np.full(LFa, np.nan)

    for idx, fiber_idx in enumerate(FN):
        v = Fa[fiber_idx]["v"]
        widall = 2.0 * Ra[v]

        passing = widall[widall <= wid_th]
        if len(passing) == 0:
            continue

        if wid_opt == 1:
            widave_sp[idx] = np.mean(passing)
            widmax_sp[idx] = np.max(passing)
        else:
            if len(passing) > wid_mp:
                wstd = np.std(passing)
                wmean = np.mean(passing)
                clipped = passing[
                    (passing >= wmean - wid_sigma * wstd)
                    & (passing <= wmean + wid_sigma * wstd)
                ]
                widave_sp[idx] = np.mean(clipped) if len(clipped) > 0 else np.mean(passing)
                widmax_sp[idx] = np.max(clipped) if len(clipped) > 0 else np.max(passing)
            else:
                widave_sp[idx] = np.mean(passing)
                widmax_sp[idx] = np.max(passing)

    return widave_sp, (widmax_sp if wid_max == 1 else None)
