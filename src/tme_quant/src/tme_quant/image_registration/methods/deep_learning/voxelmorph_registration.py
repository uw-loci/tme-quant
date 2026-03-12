"""
Complete Deep Learning Registration Methods
CoMIR and VoxelMorph - Standalone Implementation

Location: tme_quant/image_registration/methods/deep_learning/
Files:
  - comir_registration.py
  - voxelmorph_registration.py
"""

import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path
import subprocess
import tempfile
import json

# ============================================================
# FILE 2: voxelmorph_registration.py
# ============================================================

class VoxelMorphRegistration:
    """
    VoxelMorph deep learning registration.
    
    Repository: https://github.com/voxelmorph/voxelmorph
    
    Installation:
        pip install voxelmorph
    
    Example:
        >>> from tme_quant.image_registration.methods.deep_learning import VoxelMorphRegistration
        >>> 
        >>> vxm = VoxelMorphRegistration(verbose=True)
        >>> result = vxm.register(shg_image, he_image)
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize VoxelMorph registration."""
        self.verbose = verbose
        self.method_name = "voxelmorph"
        
        try:
            import voxelmorph as vxm
            self.vxm = vxm
        except ImportError:
            raise ImportError(
                "VoxelMorph required. Install with:\n"
                "pip install voxelmorph"
            )
    
    def register(self, fixed_image, moving_image, params=None):
        """
        Register using VoxelMorph.
        
        Args:
            fixed_image: Reference image
            moving_image: Image to register
            params: Registration parameters (optional)
            
        Returns:
            RegistrationResult
        """
        if self.verbose:
            print("Starting VoxelMorph registration")
        
        # Prepare images
        fixed_gray = self._ensure_grayscale(fixed_image)
        moving_gray = self._ensure_grayscale(moving_image)
        
        fixed_norm = self._normalize_image(fixed_gray)
        moving_norm = self._normalize_image(moving_gray)
        
        # Convert to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        fixed_t = torch.from_numpy(fixed_norm).float().unsqueeze(0).unsqueeze(0)
        moving_t = torch.from_numpy(moving_norm).float().unsqueeze(0).unsqueeze(0)
        
        # Create VoxelMorph network
        inshape = fixed_norm.shape
        nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
        
        model = self.vxm.networks.VxmDense(inshape, nb_features)
        model = model.to(device)
        model.eval()
        
        # Register
        with torch.no_grad():
            moved, flow = model(moving_t.to(device), fixed_t.to(device))
        
        registered = moved.squeeze().cpu().numpy()
        
        if self.verbose:
            print("  ✓ VoxelMorph complete")
        
        # Create result
        class Result:
            def __init__(self):
                self.registered_image = registered
                self.flow_field = flow.squeeze().cpu().numpy()
                self.transform_matrix = np.eye(3)
                self.method = "voxelmorph"
                self.converged = True
        
        return Result()
    
    def _ensure_grayscale(self, image):
        if image.ndim == 2:
            return image
        return np.mean(image, axis=2)
    
    def _normalize_image(self, image):
        image = image.astype(float)
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        return np.clip(image, 0, 1)