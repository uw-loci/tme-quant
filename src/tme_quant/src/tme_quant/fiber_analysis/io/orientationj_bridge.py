"""
OrientationJ plugin bridge.

Provides a self-contained interface to Fiji's OrientationJ plugin with three
backends selected automatically by :class:`OrientationJBridge`:

  1. **pyimagej** — full plugin via the ``imagej`` Python package.
  2. **subprocess** — Fiji headless macro runner called as a child process.
  3. **numpy** — structure-tensor re-implementation (always available).

The public API is a single method:

    result = bridge.run(image, params)
    # result keys: 'orientation', 'coherency', 'energy', 'color_survey'

Reference
---------
Rezakhaniha et al. (2012) OrientationJ. Biomech Model Mechanobiol.
Puespoki et al. (2016) Transforms and operators for directional bioimage
analysis. Adv Anat Embryol Cell Biol 219:69–93.
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from ._fiji_utils import (
    FijiBackendMixin,
    normalise_image,
    make_color_survey,
)


# ─────────────────────────────────────────────────────────────────────────────
# Macro templates (OrientationJ 2.0 API)
# ─────────────────────────────────────────────────────────────────────────────

_MACRO = """\
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

_MACRO_SURVEY = """\
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

            tmpl = _MACRO_SURVEY if do_survey else _MACRO
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


__all__ = ["OrientationJBridge"]