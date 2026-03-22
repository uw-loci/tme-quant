# -*- coding: utf-8 -*-
"""
Ridge Detection plugin bridge.

Provides a self-contained interface to Fiji's Ridge Detection plugin with
three backends selected automatically by :class:`RidgeDetectionBridge`:

  1. **pyimagej** — full plugin via the ``imagej`` Python package.
  2. **subprocess** — Fiji headless macro runner called as a child process.
  3. **numpy** — Hessian/Frangi re-implementation (always available).

The public API is a single method:

    result = bridge.run(image, params)
    # result keys: 'lines', 'line_width_map'

Reference
---------
Steger (1998) An unbiased detector of curvilinear structures. IEEE PAMI.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._fiji_utils import (
    FijiBackendMixin,
    normalise_image,
    contrast_value,
    contrast_to_fraction,
    order_by_nearest_neighbour,
)


# ─────────────────────────────────────────────────────────────────────────────
# Macro template (Ridge Detection 2.0 API)
# ─────────────────────────────────────────────────────────────────────────────

_MACRO = """\
open("{input_path}");
run("Ridge Detection", "line_width={line_width} \
high_contrast={high_contrast} low_contrast={low_contrast} \
estimate_width=false extend_line={extend_line_flag} \
make_binary=false show_ids=false");
saveAs("Results", "{out_results}");
"""


class RidgeDetectionBridge(FijiBackendMixin):
    """
    Python interface to Fiji's Ridge Detection plugin.

    Parameters
    ----------
    fiji_path : str or None
        Path to the ``Fiji.app`` directory.  Falls back to the
        ``FIJI_PATH`` environment variable, then to NumPy-only mode.

    Usage
    -----
    >>> bridge = RidgeDetectionBridge()
    >>> result = bridge.run(image, {
    ...     'line_width':    3.0,
    ...     'high_contrast': 0.5,   # accepts 0-1 fraction OR 0-255
    ...     'low_contrast':  0.2,
    ...     'extend_line':   True,
    ... })
    >>> lines          = result['lines']           # list of (N,2) arrays
    >>> line_width_map = result['line_width_map']  # (H,W) or None
    """

    def run(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run Ridge Detection on *image* using the best available backend.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image (any dtype; normalised internally to [0, 1]).
        params : dict
            ===================  =========================================
            ``line_width``       float — expected line width in pixels.
                                 Default: ``3.0``.
            ``high_contrast``    float — upper hysteresis threshold.
                                 Values ≥ 1 are treated as 0–255 scale;
                                 values < 1 as 0–1 fraction.
                                 Default: ``0.5`` (≈ 128 / 255).
            ``low_contrast``     float — lower hysteresis threshold
                                 (same scale as ``high_contrast``).
                                 Default: ``0.2`` (≈ 51 / 255).
            ``extend_line``      bool — extend lines to the point of
                                 highest curvature at endpoints.
                                 Default: ``True``.
            ``make_binary``      bool — return binary ridge image instead
                                 of the full response. Default: ``False``.
            ===================  =========================================

        Returns
        -------
        dict
            ===================  ===========================================
            ``'lines'``          list of ndarray (N, 2) float32 — each
                                 array gives (row, col) coordinates of one
                                 detected ridge/fiber.
            ``'line_width_map'`` ndarray (H, W) float32 — per-pixel
                                 estimated line width from Ridge Detection.
                                 ``None`` in NumPy mode.
            ===================  ===========================================
        """
        if self._backend == "pyimagej":
            return self._run_pyimagej(image, params)
        if self._backend == "subprocess":
            return self._run_subprocess(image, params)
        return self._run_numpy(image, params)

    # ── pyimagej backend ──────────────────────────────────────────────────────

    def _run_pyimagej(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Ridge Detection via pyimagej (API v1 and v2 compatible)."""
        ij = self._get_ij()

        lw   = float(params.get("line_width", 3.0))
        high = contrast_value(params.get("high_contrast", 0.5))
        low  = contrast_value(params.get("low_contrast", 0.2))
        ext  = "true" if params.get("extend_line", True) else "false"

        img_norm = normalise_image(image)
        imp = self._to_imageplus(ij, (img_norm * 255).astype(np.uint8))
        ij.ui().show("rd_input", imp)

        macro = (
            f'run("Ridge Detection", "line_width={lw} '
            f'high_contrast={high:.1f} low_contrast={low:.1f} '
            f'estimate_width=false extend_line={ext} '
            f'make_binary=false show_ids=false");'
        )
        ij.py.run_macro(macro)

        # Parse ResultsTable — columns: X, Y, Group, and optionally Width
        table = self._get_results_table(ij)
        if table is None:
            return {"lines": [], "line_width_map": None}

        xs     = list(ij.py.from_java(table.getColumn("X")))
        ys     = list(ij.py.from_java(table.getColumn("Y")))
        groups = list(ij.py.from_java(table.getColumn("Group")))
        try:
            widths = list(ij.py.from_java(table.getColumn("Width")))
        except Exception:
            widths = [None] * len(xs)

        lines = _group_ridge_table(xs, ys, groups)
        line_width_map = _build_width_map(xs, ys, widths, image.shape)

        try:
            ij.WindowManager.getImage("rd_input").close()
            table.reset()
        except Exception:
            pass

        return {"lines": lines, "line_width_map": line_width_map}

    # ── subprocess backend ────────────────────────────────────────────────────

    def _run_subprocess(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Ridge Detection via Fiji headless subprocess."""
        import tifffile

        lw   = float(params.get("line_width", 3.0))
        high = contrast_value(params.get("high_contrast", 0.5))
        low  = contrast_value(params.get("low_contrast", 0.2))
        ext  = "true" if params.get("extend_line", True) else "false"

        fiji_exe = self._find_fiji_executable()
        if fiji_exe is None:
            warnings.warn(
                "Fiji executable not found; falling back to NumPy Ridge Detection.",
                RuntimeWarning, stacklevel=3,
            )
            return self._run_numpy(image, params)

        img_norm = normalise_image(image)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp        = Path(tmpdir)
            in_p       = tmp / "input.tif"
            out_csv    = tmp / "results.csv"
            macro_p    = tmp / "run_rd.ijm"

            tifffile.imwrite(str(in_p), (img_norm * 255).astype(np.uint8))

            macro_text = _MACRO.format(
                input_path=str(in_p).replace("\\", "/"),
                line_width=lw,
                high_contrast=f"{high:.1f}",
                low_contrast=f"{low:.1f}",
                extend_line_flag=ext,
                out_results=str(out_csv).replace("\\", "/"),
            )
            macro_p.write_text(macro_text)

            proc = subprocess.run(
                [str(fiji_exe), "--headless", "--console", "-macro", str(macro_p)],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                warnings.warn(
                    f"Ridge Detection subprocess failed (rc={proc.returncode}): "
                    f"{proc.stderr[-400:]}\nFalling back to NumPy.",
                    RuntimeWarning, stacklevel=3,
                )
                return self._run_numpy(image, params)

            if not out_csv.exists():
                return {"lines": [], "line_width_map": None}

            xs, ys, groups, widths_raw = [], [], [], []
            with open(out_csv, newline="") as f:
                for row in csv.DictReader(f):
                    xs.append(float(row.get("X", 0)))
                    ys.append(float(row.get("Y", 0)))
                    groups.append(row.get("Group", "0"))
                    widths_raw.append(row.get("Width") or None)

        lines = _group_ridge_table(xs, ys, groups)
        line_width_map = _build_width_map(xs, ys, widths_raw, image.shape)

        return {"lines": lines, "line_width_map": line_width_map}

    # ── NumPy fallback ────────────────────────────────────────────────────────

    def _run_numpy(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hessian/Frangi ridge detection (NumPy/skimage fallback).

        Uses multi-sigma Frangi vesselness enhancement + hysteresis
        thresholding + skeletonization to approximate Ridge Detection output.
        """
        from skimage.filters import frangi, apply_hysteresis_threshold
        from skimage.morphology import skeletonize
        from skimage.measure import label

        lw   = float(params.get("line_width", 3.0))
        high = contrast_to_fraction(params.get("high_contrast", 0.5))
        low  = contrast_to_fraction(params.get("low_contrast", 0.2))

        img_norm = normalise_image(image).astype(np.float32)

        # Multi-sigma Frangi vesselness to match Ridge Detection's scale range
        sigma = max(lw / 2.0, 0.5)
        enhanced = frangi(
            img_norm,
            sigmas=[sigma, sigma * 1.5, sigma * 2.0],
            black_ridges=False,
        )
        e_max = enhanced.max()
        if e_max > 0:
            enhanced = enhanced / e_max

        # Hysteresis threshold — mimics Ridge Detection's two-level scheme
        binary = apply_hysteresis_threshold(enhanced, low, high)

        # Remove speckle (< 4 px components)
        lbl_sp = label(binary, connectivity=2)
        for rid in range(1, int(lbl_sp.max()) + 1):
            if (lbl_sp == rid).sum() < 4:
                binary[lbl_sp == rid] = False

        # Skeletonize to single-pixel-wide ridges
        skeleton = skeletonize(binary)

        # Label connected components → individual lines
        labeled = label(skeleton, connectivity=2)
        lines: List[np.ndarray] = []
        for rid in range(1, int(labeled.max()) + 1):
            coords = np.argwhere(labeled == rid).astype(np.float32)
            if len(coords) >= 2:
                lines.append(order_by_nearest_neighbour(coords))

        return {"lines": lines, "line_width_map": None}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (used by both pyimagej and subprocess backends)
# ─────────────────────────────────────────────────────────────────────────────

def _group_ridge_table(
    xs: List[float],
    ys: List[float],
    groups: List[Any],
) -> List[np.ndarray]:
    """
    Convert parallel X / Y / Group columns from a Ridge Detection ResultsTable
    into a list of per-line (row, col) coordinate arrays.
    """
    buckets: Dict[Any, List[Tuple[float, float]]] = defaultdict(list)
    for x, y, g in zip(xs, ys, groups):
        buckets[g].append((float(y), float(x)))   # (row, col)

    return [
        np.array(pts, dtype=np.float32)
        for pts in buckets.values()
        if len(pts) >= 2
    ]


def _build_width_map(
    xs: List[float],
    ys: List[float],
    widths: List[Optional[Any]],
    shape: tuple,
) -> Optional[np.ndarray]:
    """
    Build a per-pixel width map from Ridge Detection Width column data.
    Returns ``None`` if no width data is available.
    """
    valid_widths = [w for w in widths if w is not None and w != ""]
    if not valid_widths:
        return None

    width_map = np.zeros(shape, dtype=np.float32)
    for x, y, w in zip(xs, ys, widths):
        if w is not None and w != "":
            r, c = int(round(float(y))), int(round(float(x)))
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                width_map[r, c] = float(w)
    return width_map


__all__ = ["RidgeDetectionBridge"]