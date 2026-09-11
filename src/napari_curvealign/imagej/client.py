"""
ImageJ/Fiji client for the CurveAlign napari plugin.

Wraps napari-imagej for Bio-Formats I/O, Fiji plugins (Tubeness, Frangi, etc.),
ROI Manager, and TrackMate-related entry points. Keeps JVM details out of UI code.
"""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import napari_imagej

    HAS_IMAGEJ = True
except ImportError:
    HAS_IMAGEJ = False
    napari_imagej = None


class FijiBridge:
    """
    Bridge to Fiji/ImageJ via napari-imagej.

    Provides access to Fiji plugins and operations alongside napari workflows.
    """

    def __init__(self):
        self._ij = None
        self._initialized = False

    def initialize(self, mode: str = "headless") -> bool:
        """
        Initialize ImageJ/Fiji.

        Parameters
        ----------
        mode : str, default "headless"
            ``"headless"``, ``"gui"``, or ``"interactive"`` (passed to napari-imagej).
        """
        if not HAS_IMAGEJ:
            warnings.warn("napari-imagej not available. Install with: pip install napari-imagej")
            return False

        try:
            if mode == "headless":
                self._ij = napari_imagej.init(headless=True)
            elif mode == "gui":
                self._ij = napari_imagej.init(headless=False)
            else:
                self._ij = napari_imagej.init()

            self._initialized = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to initialize ImageJ: {e}")
            return False

    @property
    def ij(self):
        if not self._initialized:
            self.initialize()
        return self._ij

    def is_available(self) -> bool:
        return HAS_IMAGEJ and self._initialized

    def load_image_bioformats(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")

        try:
            dataset = self.ij.scifio().datasetIO().open(file_path)
            image_data = np.array(dataset.data())

            metadata = {
                "source": "bioformats",
                "shape": image_data.shape,
                "dims": str(dataset.dims()),
            }

            return image_data, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to load image with Bio-Formats: {e}") from e

    def apply_tubeness(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")

        try:
            ij_image = self.ij.py.to_java(image)
            self.ij.ui().show("input", ij_image)
            self.ij.command().run("Tubeness", True, f"sigma={sigma}")

            result_window = self.ij.WindowManager.getCurrentImage()
            if result_window:
                result = self.ij.py.from_java(result_window.getProcessor().getPixels())
                return result
            from skimage.filters import meijering

            return meijering(image, sigmas=sigma, black_ridges=False)
        except Exception as e:
            warnings.warn(f"Tubeness via Fiji failed: {e}, using Python fallback")
            from skimage.filters import meijering

            return meijering(image, sigmas=sigma, black_ridges=False)

    def apply_frangi(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")

        try:
            ij_image = self.ij.py.to_java(image)
            self.ij.ui().show("input", ij_image)
            self.ij.command().run("Frangi", True, **kwargs)

            result_window = self.ij.WindowManager.getCurrentImage()
            if result_window:
                result = self.ij.py.from_java(result_window.getProcessor().getPixels())
                return result
            from skimage.filters import frangi

            return frangi(image, **kwargs)
        except Exception as e:
            warnings.warn(f"Frangi via Fiji failed: {e}, using Python fallback")
            from skimage.filters import frangi

            return frangi(image, **kwargs)

    def get_roi_manager(self):
        if not self._initialized and HAS_IMAGEJ:
            self.initialize()

        if not self.is_available():
            return None

        try:
            return self.ij.roiManager()
        except Exception as e:
            warnings.warn(f"ROI Manager not available: {e}")
            return None

    def export_rois_to_fiji(self, rois: List[Any]) -> bool:
        roi_manager = self.get_roi_manager()
        if roi_manager is None:
            return False

        try:
            ij = self.ij

            for roi in rois:
                shape_type = roi.shape.value if hasattr(roi.shape, "value") else str(roi.shape)
                name = roi.name
                coords = roi.coordinates

                ij_roi = None

                if shape_type == "Rectangle":
                    x_min = float(np.min(coords[:, 0]))
                    y_min = float(np.min(coords[:, 1]))
                    width = float(np.max(coords[:, 0]) - x_min)
                    height = float(np.max(coords[:, 1]) - y_min)
                    ij_roi = ij.gui.Roi(x_min, y_min, width, height)

                elif shape_type == "Ellipse":
                    x_min = float(np.min(coords[:, 0]))
                    y_min = float(np.min(coords[:, 1]))
                    width = float(np.max(coords[:, 0]) - x_min)
                    height = float(np.max(coords[:, 1]) - y_min)
                    ij_roi = ij.gui.OvalRoi(x_min, y_min, width, height)

                elif shape_type in ["Polygon", "Freehand"]:
                    x_points = coords[:, 0].astype(float).tolist()
                    y_points = coords[:, 1].astype(float).tolist()
                    roi_type = 2 if shape_type == "Polygon" else 3
                    ij_roi = ij.gui.PolygonRoi(x_points, y_points, len(x_points), roi_type)

                if ij_roi:
                    ij_roi.setName(name)
                    roi_manager.addRoi(ij_roi)

            return True
        except Exception as e:
            warnings.warn(f"Failed to export ROIs to Fiji: {e}")
            return False

    def import_rois_from_fiji(self) -> List[Dict]:
        roi_manager = self.get_roi_manager()
        if roi_manager is None:
            return []

        try:
            rois_data = []
            rois_array = roi_manager.getRoisAsArray()

            for ij_roi in rois_array:
                name = ij_roi.getName()
                roi_type = ij_roi.getType()

                poly = ij_roi.getFloatPolygon()
                x_points = poly.xpoints
                y_points = poly.ypoints
                n_points = poly.npoints

                coords = []
                for i in range(n_points):
                    coords.append([x_points[i], y_points[i]])
                coords = np.array(coords, dtype=float)

                shape_type = "Polygon"
                if roi_type == 0:
                    shape_type = "Rectangle"
                elif roi_type == 1:
                    shape_type = "Ellipse"
                elif roi_type == 3 or roi_type == 4:
                    shape_type = "Freehand"

                rois_data.append(
                    {
                        "name": name,
                        "shape": shape_type,
                        "coordinates": coords,
                    }
                )

            return rois_data
        except Exception as e:
            warnings.warn(f"Failed to import ROIs from Fiji: {e}")
            return []

    def run_trackmate(self, image: np.ndarray, **params) -> Dict:
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")

        try:
            ij_image = self.ij.py.to_java(image)
            self.ij.ui().show("input", ij_image)
            self.ij.command().run("TrackMate", True, **params)
            return {"tracks": [], "spots": []}
        except Exception as e:
            warnings.warn(f"TrackMate failed: {e}")
            return {"tracks": [], "spots": []}

    def run_orientationj(self, image: np.ndarray, **kwargs):
        if not self.is_available():
            warnings.warn("Fiji not available")
            return None

        warnings.warn(
            "Automated OrientationJ execution not fully implemented. "
            "Please run in Fiji and export results."
        )
        return None

    def run_ridge_detection(self, image: np.ndarray, **kwargs):
        if not self.is_available():
            warnings.warn("Fiji not available")
            return None

        warnings.warn(
            "Automated Ridge Detection execution not fully implemented. "
            "Please run in Fiji and export results."
        )
        return None


_fiji_bridge = None


def get_fiji_bridge() -> FijiBridge:
    """Singleton used by the dock widget to share one ImageJ context."""
    global _fiji_bridge
    if _fiji_bridge is None:
        _fiji_bridge = FijiBridge()
    return _fiji_bridge
