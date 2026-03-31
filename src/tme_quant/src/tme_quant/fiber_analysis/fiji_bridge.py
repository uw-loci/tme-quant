"""
Fiji / ImageJ bridge for fiber analysis.

Provides:
  - FijiBackendMixin  — backend detection, pyimagej init, version compat
  - normalise_image   — float32 [0,1] normalisation
  - contrast_value    — normalise contrast threshold to 0-255 scale
  - contrast_to_fraction — normalise contrast threshold to 0-1 fraction
  - make_color_survey — synthesise OrientationJ-style HSB colour survey
  - order_by_nearest_neighbour — order skeleton coords into a sequential path
"""


import csv
import os
import subprocess
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


class FijiBackendMixin:
    """
    Mixin that provides backend detection and pyimagej lifecycle management
    for OrientationJBridge and RidgeDetectionBridge.

    Subclasses must call ``super().__init__(fiji_path)`` (or set
    ``self._fiji_path`` themselves) before calling any other method.
    """

    def __init__(self, fiji_path: Optional[str] = None) -> None:
        self._fiji_path: str = fiji_path or os.environ.get("FIJI_PATH", "")
        self._ij = None                        # lazy pyimagej handle
        self._ij_version: Optional[int] = None  # 1 or 2
        self._backend: str = self._detect_backend()

    # ── Backend detection ─────────────────────────────────────────────────────

    def _detect_backend(self) -> str:
        """Return ``'pyimagej'``, ``'subprocess'``, or ``'numpy'``."""
        if self._fiji_path and os.path.isdir(self._fiji_path):
            try:
                import imagej  # type: ignore  # noqa: F401
                return "pyimagej"
            except ImportError:
                pass
            if self._find_fiji_executable() is not None:
                return "subprocess"
        return "numpy"

    def _find_fiji_executable(self) -> Optional[Path]:
        """Return the Fiji executable path for the current OS, or None."""
        root = Path(self._fiji_path)
        for candidate in [
            root / "ImageJ-linux64",
            root / "ImageJ-linux32",
            root / "ImageJ-win64.exe",
            root / "ImageJ-win32.exe",
            root / "Contents" / "MacOS" / "ImageJ-macosx",
        ]:
            if candidate.exists():
                return candidate
        return None

    def is_fiji_available(self) -> bool:
        """Return True if a real Fiji backend (pyimagej or subprocess) is active."""
        return self._backend in ("pyimagej", "subprocess")

    # ── pyimagej lifecycle ────────────────────────────────────────────────────

    def _get_ij(self):
        """Initialise and return the pyimagej ImageJ instance (lazy)."""
        if self._ij is None:
            import imagej  # type: ignore
            self._ij = imagej.init(self._fiji_path, mode="headless")
            # Detect API version: 2.x has ij.py.to_java; 1.x has to_imageplus
            self._ij_version = 2 if hasattr(self._ij.py, "to_java") else 1
        return self._ij

    def _to_imageplus(self, ij: Any, arr: np.ndarray) -> Any:
        """Convert ndarray to ImagePlus, handling pyimagej v1/v2 API."""
        if self._ij_version == 2:
            return ij.py.to_java(arr)
        return ij.py.to_imageplus(arr)

    def _get_results_table(self, ij: Any) -> Any:
        """Return the active ResultsTable, handling pyimagej v1/v2 API."""
        if self._ij_version == 2:
            return ij.ResultsTable.getActiveTable()
        return ij.ResultsTable.getResultsTable()

    def close(self) -> None:
        """Shut down the pyimagej JVM if it was started."""
        if self._ij is not None:
            try:
                self._ij.dispose()
            except Exception:
                pass
            self._ij = None


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function utilities (no class needed)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_image(image: np.ndarray) -> np.ndarray:
    """Return a float32 copy of *image* normalised to [0, 1]."""
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


def contrast_value(value: Any) -> float:
    """
    Normalise *value* to the 0–255 scale Fiji expects.

    Values ≥ 1 are assumed to already be on the 0–255 scale.
    Values < 1 are assumed to be a 0–1 fraction and are multiplied by 255.
    """
    v = float(value)
    return v if v >= 1.0 else v * 255.0


def contrast_to_fraction(value: Any) -> float:
    """
    Normalise *value* to a 0–1 fraction for internal use.

    Values ≥ 1 are divided by 255.  Values < 1 are returned as-is.
    """
    v = float(value)
    return v / 255.0 if v >= 1.0 else v


def make_color_survey(
    orientation_map: np.ndarray,
    coherency_map: np.ndarray,
) -> np.ndarray:
    """
    Synthesise an OrientationJ-style HSB colour-survey image.

    Encoding
    --------
    Hue   = orientation mapped from [−90°, +90°] → [0°, 360°]
    Sat   = coherency (0 = grey, 1 = fully saturated)
    Value = 1.0 for valid pixels, 0 for NaN orientations

    Returns
    -------
    ndarray of shape (H, W, 3), dtype uint8, RGB colour order.
    """
    import colorsys

    h_map = (orientation_map + 90.0) / 180.0   # hue in [0, 1]
    s_map = np.clip(coherency_map, 0.0, 1.0)
    valid = ~np.isnan(orientation_map)

    rgb = np.zeros((*orientation_map.shape, 3), dtype=np.uint8)
    for r in range(orientation_map.shape[0]):
        for c in range(orientation_map.shape[1]):
            if valid[r, c]:
                rv, gv, bv = colorsys.hsv_to_rgb(
                    float(h_map[r, c]), float(s_map[r, c]), 1.0
                )
                rgb[r, c] = (int(rv * 255), int(gv * 255), int(bv * 255))
    return rgb


def order_by_nearest_neighbour(coords: np.ndarray) -> np.ndarray:
    """
    Order an unordered set of skeleton coordinates into a sequential path.

    Uses a greedy nearest-neighbour walk starting from the first point.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)

    Returns
    -------
    ndarray, shape (N, 2)
    """
    if len(coords) <= 2:
        return coords

    remaining = list(range(len(coords)))
    path      = [remaining.pop(0)]

    while remaining:
        current = coords[path[-1]]
        dists   = np.linalg.norm(coords[remaining] - current, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        path.append(nearest)
        remaining.remove(nearest)

    return coords[path]



# ─────────────────────────────────────────────────────────────────────────────
# FijiBridge coordinator
# ─────────────────────────────────────────────────────────────────────────────

class FijiBridge:
    """
    Coordinator that delegates to :class:`OrientationJBridge` and
    :class:`RidgeDetectionBridge`.

    Callers (``orientationj.py``, ``ridge_detection.py``) only need:

    * :meth:`is_fiji_available`
    * :meth:`call_orientationj`
    * :meth:`call_ridge_detection`

    The individual plugin bridges are also accessible directly via
    :attr:`orientationj` and :attr:`ridge_detection` for advanced use.

    Parameters
    ----------
    fiji_path : str or None
        Path to ``Fiji.app``.  Falls back to ``FIJI_PATH`` env var,
        then to NumPy-only mode.
    """

    def __init__(self, fiji_path: Optional[str] = None) -> None:
        self.orientationj    = OrientationJBridge(fiji_path)
        self.ridge_detection = RidgeDetectionBridge(fiji_path)

    # ── Public interface (used by method files) ───────────────────────────────

    def is_fiji_available(self) -> bool:
        """
        Return True if either plugin bridge has a real Fiji backend active.

        Both bridges share the same ``FIJI_PATH`` so their backends will
        always agree; we check the OrientationJ bridge as the canonical one.
        """
        return self.orientationj.is_fiji_available()

    @property
    def _backend(self) -> str:
        """Active backend name (``'pyimagej'``, ``'subprocess'``, or ``'numpy'``)."""
        return self.orientationj._backend

    def call_orientationj(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Run OrientationJ and return orientation, coherency, and energy maps.

        Delegates to :meth:`OrientationJBridge.run`.
        See :class:`~.orientationj_bridge.OrientationJBridge` for full
        parameter and return-value documentation.
        """
        return self.orientationj.run(image, params)

    def call_ridge_detection(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run Ridge Detection and return detected line coordinates.

        Delegates to :meth:`RidgeDetectionBridge.run`.
        See :class:`~.ridge_detection_bridge.RidgeDetectionBridge` for full
        parameter and return-value documentation.
        """
        return self.ridge_detection.run(image, params)

    def close(self) -> None:
        """Shut down any pyimagej JVM instances held by the plugin bridges."""
        self.orientationj.close()
        self.ridge_detection.close()



# ─────────────────────────────────────────────────────────────────────────────
# OrientationJ bridge
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Macro templates (OrientationJ 2.0 API)
# ─────────────────────────────────────────────────────────────────────────────

_OJ_MACRO = """\
open("{input_path}");
run("OrientationJ Analysis", "tensor={sigma} gradient=[{gradient}] \
color-survey={survey_flag} s-coherency=on s-energy=on \
min-coherency={min_coherency} min-energy={min_energy} \
orientation=on coherency=on energy=on");
selectWindow("OJ-Orientation-1");
saveAs("Tiff", "{out_orientation}");
selectWindow("OJ-Coherency-1");
saveAs("Tiff", "{out_coherency}");
selectWindow("OJ-Energy-1");
saveAs("Tiff", "{out_energy}");
"""

_OJ_MACRO_SURVEY = """\
open("{input_path}");
run("OrientationJ Analysis", "tensor={sigma} gradient=[{gradient}] \
color-survey=on s-coherency=on s-energy=on \
min-coherency={min_coherency} min-energy={min_energy} \
orientation=on coherency=on energy=on");
selectWindow("OJ-Color-survey-1");
saveAs("Tiff", "{out_survey}");
selectWindow("OJ-Orientation-1");
saveAs("Tiff", "{out_orientation}");
selectWindow("OJ-Coherency-1");
saveAs("Tiff", "{out_coherency}");
selectWindow("OJ-Energy-1");
saveAs("Tiff", "{out_energy}");
"""


class OrientationJBridge(FijiBackendMixin):
    """
    Python interface to Fiji's OrientationJ plugin.

    Parameters
    ----------
    fiji_path : str or None
        Path to the ``Fiji.app`` directory.  Falls back to the
        ``FIJI_PATH`` environment variable, then to NumPy-only mode.

    Usage
    -----
    >>> bridge = OrientationJBridge()
    >>> result = bridge.run(image, {
    ...     'gradient':     'Gaussian',
    ...     'sigma':        2.0,
    ...     'min-coherency': 0.1,
    ...     'min-energy':   0.0,
    ...     'color-survey': False,
    ... })
    >>> orientation_map = result['orientation']   # (H, W) float32, degrees
    >>> coherency_map   = result['coherency']     # (H, W) float32, 0-1
    >>> energy_map      = result['energy']        # (H, W) float32, 0-1
    """

    def run(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Run OrientationJ on *image* using the best available backend.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            2-D grayscale image (any dtype; normalised internally to [0, 1]).
        params : dict
            ===================  =========================================
            ``gradient``         str — gradient method for OrientationJ.
                                 Options: ``'Gaussian'`` (default),
                                 ``'Finite difference'``, ``'Fourier'``,
                                 ``'Riesz'``, ``'Cubic spline'``.
            ``sigma``            float — structure tensor smoothing sigma
                                 in pixels. Default: ``2.0``.
            ``min-coherency``    float — coherency threshold in [0, 1].
                                 Pixels below are set to NaN in the
                                 orientation map. Default: ``0.0``.
            ``min-energy``       float — energy threshold in [0, 1].
                                 Default: ``0.0``.
            ``color-survey``     bool — also return the HSB colour-survey
                                 image. Default: ``False``.
            ===================  =========================================

        Returns
        -------
        dict
            ====================  ==========================================
            ``'orientation'``     ndarray (H, W) float32 — degrees [−90,+90]
            ``'coherency'``       ndarray (H, W) float32 — [0, 1]
            ``'energy'``          ndarray (H, W) float32 — normalised [0, 1]
            ``'color_survey'``    ndarray (H, W, 3) uint8 — RGB colour survey
                                  (only present when ``color-survey=True``)
            ====================  ==========================================
        """
        if self._backend == "pyimagej":
            return self._run_pyimagej(image, params)
        if self._backend == "subprocess":
            return self._run_subprocess(image, params)
        return self._run_numpy(image, params)

    # ── pyimagej backend ──────────────────────────────────────────────────────

    def _run_pyimagej(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Call OrientationJ via pyimagej (API v1 and v2 compatible)."""
        ij = self._get_ij()

        gradient  = params.get("gradient", "Gaussian")
        sigma     = float(params.get("sigma", 2.0))
        min_coh   = float(params.get("min-coherency", 0.0))
        min_eng   = float(params.get("min-energy", 0.0))
        do_survey = bool(params.get("color-survey", False))

        img_norm = normalise_image(image)
        imp      = self._to_imageplus(ij, img_norm)
        ij.ui().show("oj_input", imp)

        macro = (
            f'run("OrientationJ Analysis", '
            f'"tensor={sigma} gradient=[{gradient}] '
            f'color-survey={"on" if do_survey else "off"} '
            f's-coherency=on s-energy=on '
            f'min-coherency={min_coh} min-energy={min_eng} '
            f'orientation=on coherency=on energy=on");'
        )
        ij.py.run_macro(macro)

        result: Dict[str, np.ndarray] = {}
        for title, key in [
            ("OJ-Orientation-1", "orientation"),
            ("OJ-Coherency-1",   "coherency"),
            ("OJ-Energy-1",      "energy"),
        ]:
            win = ij.WindowManager.getImage(title) \
                  or ij.WindowManager.getCurrentImage()
            result[key] = np.array(ij.py.from_java(win), dtype=np.float32)
            win.close()

        if do_survey:
            win = ij.WindowManager.getImage("OJ-Color-survey-1")
            if win is not None:
                result["color_survey"] = np.array(
                    ij.py.from_java(win), dtype=np.uint8
                )
                win.close()

        try:
            ij.WindowManager.getImage("oj_input").close()
        except Exception:
            pass

        return result

    # ── subprocess backend ────────────────────────────────────────────────────

    def _run_subprocess(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Call OrientationJ via Fiji headless subprocess."""
        import tifffile

        gradient  = params.get("gradient", "Gaussian")
        sigma     = float(params.get("sigma", 2.0))
        min_coh   = float(params.get("min-coherency", 0.0))
        min_eng   = float(params.get("min-energy", 0.0))
        do_survey = bool(params.get("color-survey", False))

        fiji_exe = self._find_fiji_executable()
        if fiji_exe is None:
            warnings.warn(
                "Fiji executable not found; falling back to NumPy OrientationJ.",
                RuntimeWarning, stacklevel=3,
            )
            return self._run_numpy(image, params)

        img_norm = normalise_image(image)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp  = Path(tmpdir)
            in_p = tmp / "input.tif"
            out_ori = tmp / "orientation.tif"
            out_coh = tmp / "coherency.tif"
            out_eng = tmp / "energy.tif"
            out_srv = tmp / "survey.tif"
            macro_p = tmp / "run_oj.ijm"

            tifffile.imwrite(str(in_p), img_norm)

            tmpl = _OJ_MACRO_SURVEY if do_survey else _OJ_MACRO
            macro_text = tmpl.format(
                input_path=str(in_p).replace("\\", "/"),
                sigma=sigma,
                gradient=gradient,
                survey_flag="on" if do_survey else "off",
                min_coherency=min_coh,
                min_energy=min_eng,
                out_orientation=str(out_ori).replace("\\", "/"),
                out_coherency=str(out_coh).replace("\\", "/"),
                out_energy=str(out_eng).replace("\\", "/"),
                out_survey=str(out_srv).replace("\\", "/"),
            )
            macro_p.write_text(macro_text)

            proc = subprocess.run(
                [str(fiji_exe), "--headless", "--console", "-macro", str(macro_p)],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                warnings.warn(
                    f"OrientationJ subprocess failed (rc={proc.returncode}): "
                    f"{proc.stderr[-400:]}\nFalling back to NumPy.",
                    RuntimeWarning, stacklevel=3,
                )
                return self._run_numpy(image, params)

            result: Dict[str, np.ndarray] = {
                "orientation": tifffile.imread(str(out_ori)).astype(np.float32),
                "coherency":   tifffile.imread(str(out_coh)).astype(np.float32),
                "energy":      tifffile.imread(str(out_eng)).astype(np.float32),
            }
            if do_survey and out_srv.exists():
                result["color_survey"] = tifffile.imread(str(out_srv)).astype(np.uint8)

        return result

    # ── NumPy fallback ────────────────────────────────────────────────────────

    def _run_numpy(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Structure-tensor orientation analysis (NumPy/SciPy).

        Produces output closely matching OrientationJ's 'Gaussian' gradient
        mode.  The colour-survey image is synthesised locally when requested.
        """
        from scipy.ndimage import sobel

        sigma     = float(params.get("sigma", 2.0))
        min_coh   = float(params.get("min-coherency", 0.0))
        do_survey = bool(params.get("color-survey", False))

        img = normalise_image(image).astype(np.float64)

        gradient_method = params.get("gradient", "Gaussian").lower()
        if gradient_method == "finite difference":
            gy = np.gradient(img, axis=0)
            gx = np.gradient(img, axis=1)
        else:
            # Gaussian gradient: smooth then Sobel
            smooth = gaussian_filter(img, sigma=0.5)
            gx = sobel(smooth, axis=1).astype(np.float64)
            gy = sobel(smooth, axis=0).astype(np.float64)

        # Structure tensor J = G_sigma * (grad ⊗ grad)
        Jxx = gaussian_filter(gx * gx, sigma=sigma)
        Jxy = gaussian_filter(gx * gy, sigma=sigma)
        Jyy = gaussian_filter(gy * gy, sigma=sigma)

        # Orientation: dominant eigenvector angle → [−90°, +90°]
        orientation = (0.5 * np.degrees(
            np.arctan2(2.0 * Jxy, Jxx - Jyy)
        )).astype(np.float32)

        # Coherency: (λ_max − λ_min) / (λ_max + λ_min)
        diff  = Jxx - Jyy
        trace = Jxx + Jyy
        disc  = np.sqrt(np.maximum(diff**2 + 4.0 * Jxy**2, 0.0))
        denom = np.where(trace > 1e-12, trace, 1e-12)
        coherency = np.clip(disc / denom, 0.0, 1.0).astype(np.float32)

        # Energy: normalised trace
        t_max = trace.max()
        energy = (trace / (t_max + 1e-12)).astype(np.float32)

        # Apply coherency threshold — pixels below → NaN (matches OrientationJ)
        if min_coh > 0.0:
            orientation = orientation.copy()
            orientation[coherency < min_coh] = np.nan

        result: Dict[str, np.ndarray] = {
            "orientation": orientation,
            "coherency":   coherency,
            "energy":      energy,
        }
        if do_survey:
            result["color_survey"] = make_color_survey(orientation, coherency)

        return result



# ─────────────────────────────────────────────────────────────────────────────
# Ridge Detection bridge
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Macro template (Ridge Detection 2.0 API)
# ─────────────────────────────────────────────────────────────────────────────

_RD_MACRO = """\
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

            macro_text = _RD_MACRO.format(
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


__all__ = [
    "FijiBackendMixin",
    "FijiBridge",
    "OrientationJBridge",
    "RidgeDetectionBridge",
    "normalise_image",
    "contrast_value",
    "contrast_to_fraction",
    "make_color_survey",
    "order_by_nearest_neighbour",
]
