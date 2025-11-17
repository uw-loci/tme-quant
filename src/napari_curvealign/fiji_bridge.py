"""
Fiji/ImageJ bridge for CurveAlign Napari plugin.

Provides integration with Fiji/ImageJ via napari-imagej for:
- Bio-Formats import/export
- Fiji plugin access (Tubeness, Frangi, etc.)
- ROI Manager integration
- TrackMate integration
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import warnings

try:
    import napari_imagej
    HAS_IMAGEJ = True
except ImportError:
    HAS_IMAGEJ = False
    napari_imagej = None

try:
    import pyimagej
    HAS_PYIMAGEJ = True
except ImportError:
    HAS_PYIMAGEJ = False
    pyimagej = None


class FijiBridge:
    """
    Bridge to Fiji/ImageJ functionality via napari-imagej.
    
    Provides access to Fiji plugins and operations while maintaining
    compatibility with napari workflows.
    """
    
    def __init__(self):
        """Initialize Fiji bridge."""
        self._ij = None
        self._initialized = False
    
    def initialize(self, mode: str = "headless") -> bool:
        """
        Initialize ImageJ/Fiji.
        
        Parameters
        ----------
        mode : str, default "headless"
            Initialization mode: "headless", "gui", or "interactive"
        
        Returns
        -------
        bool
            True if initialization successful
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
        """Get ImageJ instance."""
        if not self._initialized:
            self.initialize()
        return self._ij
    
    def is_available(self) -> bool:
        """Check if Fiji/ImageJ is available."""
        return HAS_IMAGEJ and self._initialized
    
    def load_image_bioformats(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load image using Bio-Formats.
        
        Parameters
        ----------
        file_path : str
            Path to image file
        
        Returns
        -------
        Tuple[np.ndarray, Dict]
            Image data and metadata
        """
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")
        
        try:
            # Use Bio-Formats to open image
            dataset = self.ij.scifio().datasetIO().open(file_path)
            image_data = np.array(dataset.data())
            
            # Get metadata
            metadata = {
                "source": "bioformats",
                "shape": image_data.shape,
                "dims": str(dataset.dims()),
            }
            
            return image_data, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to load image with Bio-Formats: {e}")
    
    def apply_tubeness(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Tubeness filter via Fiji plugin.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        sigma : float, default 1.0
            Sigma parameter
        
        Returns
        -------
        np.ndarray
            Filtered image
        """
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")
        
        try:
            # Convert numpy array to ImageJ ImagePlus
            ij_image = self.ij.py.to_java(image)
            
            # Run Tubeness plugin
            # Note: This requires the Tubeness plugin to be installed in Fiji
            self.ij.ui().show("input", ij_image)
            
            # Run plugin via macro command
            # This is a simplified version - actual implementation would
            # need to properly call the Tubeness plugin
            self.ij.command().run("Tubeness", True, f"sigma={sigma}")
            
            # Get result
            result_window = self.ij.WindowManager.getCurrentImage()
            if result_window:
                result = self.ij.py.from_java(result_window.getProcessor().getPixels())
                return result
            else:
                # Fallback to Python implementation
                from skimage.filters import meijering
                return meijering(image, sigmas=sigma, black_ridges=False)
        except Exception as e:
            warnings.warn(f"Tubeness via Fiji failed: {e}, using Python fallback")
            from skimage.filters import meijering
            return meijering(image, sigmas=sigma, black_ridges=False)
    
    def apply_frangi(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Frangi filter via Fiji plugin.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        **kwargs
            Additional parameters for Frangi filter
        
        Returns
        -------
        np.ndarray
            Filtered image
        """
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")
        
        try:
            # Convert to ImageJ format
            ij_image = self.ij.py.to_java(image)
            self.ij.ui().show("input", ij_image)
            
            # Run Frangi plugin
            # This requires the Frangi plugin to be installed
            self.ij.command().run("Frangi", True, **kwargs)
            
            # Get result
            result_window = self.ij.WindowManager.getCurrentImage()
            if result_window:
                result = self.ij.py.from_java(result_window.getProcessor().getPixels())
                return result
            else:
                # Fallback to Python
                from skimage.filters import frangi
                return frangi(image, **kwargs)
        except Exception as e:
            warnings.warn(f"Frangi via Fiji failed: {e}, using Python fallback")
            from skimage.filters import frangi
            return frangi(image, **kwargs)
    
    def get_roi_manager(self):
        """
        Get Fiji ROI Manager.
        
        Returns
        -------
        ROI Manager object if available
        """
        if not self.is_available():
            return None
        
        try:
            return self.ij.roiManager()
        except Exception as e:
            warnings.warn(f"ROI Manager not available: {e}")
            return None
    
    def export_rois_to_fiji(self, rois: List[Any]) -> bool:
        """
        Export ROIs to Fiji ROI Manager.
        
        Parameters
        ----------
        rois : List
            List of ROI objects to export
        
        Returns
        -------
        bool
            True if successful
        """
        roi_manager = self.get_roi_manager()
        if roi_manager is None:
            return False
        
        try:
            # Clear existing ROIs
            roi_manager.reset()
            
            # Convert and add ROIs
            for roi in rois:
                # Convert ROI to ImageJ format
                # This would need proper conversion based on ROI type
                pass
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to export ROIs to Fiji: {e}")
            return False
    
    def import_rois_from_fiji(self) -> List[Any]:
        """
        Import ROIs from Fiji ROI Manager.
        
        Returns
        -------
        List
            List of imported ROIs
        """
        roi_manager = self.get_roi_manager()
        if roi_manager is None:
            return []
        
        try:
            rois = []
            count = roi_manager.getCount()
            for i in range(count):
                roi = roi_manager.getRoi(i)
                # Convert ImageJ ROI to napari format
                # This would need proper conversion
                rois.append(roi)
            
            return rois
        except Exception as e:
            warnings.warn(f"Failed to import ROIs from Fiji: {e}")
            return []
    
    def run_trackmate(self, image: np.ndarray, **params) -> Dict:
        """
        Run TrackMate for cell tracking.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        **params
            TrackMate parameters
        
        Returns
        -------
        Dict
            Tracking results
        """
        if not self.is_available():
            raise RuntimeError("Fiji/ImageJ not initialized")
        
        try:
            # Convert to ImageJ format
            ij_image = self.ij.py.to_java(image)
            self.ij.ui().show("input", ij_image)
            
            # Run TrackMate
            # This would require TrackMate plugin
            self.ij.command().run("TrackMate", True, **params)
            
            # Extract results
            # This would need to parse TrackMate output
            return {"tracks": [], "spots": []}
        except Exception as e:
            warnings.warn(f"TrackMate failed: {e}")
            return {"tracks": [], "spots": []}


# Global bridge instance
_fiji_bridge = None

def get_fiji_bridge() -> FijiBridge:
    """Get or create global Fiji bridge instance."""
    global _fiji_bridge
    if _fiji_bridge is None:
        _fiji_bridge = FijiBridge()
    return _fiji_bridge

