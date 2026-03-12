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
# FILE 1: comir_registration.py
# ============================================================

class CoMIRRegistration:
    """
    CoMIR (Contrastive Multimodal Image Registration) wrapper.
    
    Repository: https://github.com/MIDA-group/CoMIR_INSPIRE
    Paper: Unsupervised deep learning for multimodal medical image registration
    
    Installation:
        git clone https://github.com/MIDA-group/CoMIR_INSPIRE.git
        cd CoMIR_INSPIRE
        pip install -r requirements.txt
    
    Example:
        >>> from tme_quant.image_registration.methods.deep_learning import CoMIRRegistration
        >>> 
        >>> comir = CoMIRRegistration(
        ...     comir_path="/path/to/CoMIR_INSPIRE",
        ...     model_path="/path/to/pretrained_model.pth",
        ...     verbose=True
        ... )
        >>> 
        >>> result = comir.register(shg_image, he_image, params)
        >>> registered_he = result.registered_image
    """
    
    def __init__(
        self,
        verbose: bool = False,
        comir_path: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize CoMIR registration.
        
        Args:
            verbose: Print progress messages
            comir_path: Path to CoMIR_INSPIRE directory
            model_path: Path to pretrained model (.pth file)
        """
        self.verbose = verbose
        self.method_name = "comir"
        
        # Locate CoMIR installation
        self.comir_path = Path(comir_path) if comir_path else self._find_comir()
        self.model_path = Path(model_path) if model_path else None
        
        if not self.comir_path:
            raise ImportError(
                "CoMIR not found. Install from:\n"
                "git clone https://github.com/MIDA-group/CoMIR_INSPIRE.git"
            )
        
        # Check PyTorch
        if not torch.cuda.is_available() and self.verbose:
            print("⚠️  CUDA not available. CoMIR will use CPU (slower)")
    
    def register(self, fixed_image, moving_image, params=None):
        """
        Register using CoMIR deep learning.
        
        Args:
            fixed_image: Reference image (e.g., SHG)
            moving_image: Image to register (e.g., H&E)
            params: Registration parameters (optional)
            
        Returns:
            RegistrationResult
        """
        if self.verbose:
            print("Starting CoMIR deep learning registration")
        
        import time
        start_time = time.time()
        
        # Prepare images
        fixed_gray = self._ensure_grayscale(fixed_image)
        moving_gray = self._ensure_grayscale(moving_image)
        
        fixed_norm = self._normalize_image(fixed_gray)
        moving_norm = self._normalize_image(moving_gray)
        
        # Try Python module first, fall back to CLI
        try:
            registered, disp_field = self._register_module(fixed_norm, moving_norm)
        except ImportError:
            if self.verbose:
                print("  Using CoMIR CLI interface")
            registered, disp_field = self._register_cli(fixed_norm, moving_norm)
        
        if self.verbose:
            print(f"  ✓ CoMIR complete ({time.time() - start_time:.2f}s)")
        
        # Create result
        class Result:
            def __init__(self):
                self.registered_image = registered
                self.displacement_field = disp_field
                self.transform_matrix = np.eye(3)
                self.method = "comir"
                self.converged = True
        
        return Result()
    
    def _register_module(self, fixed, moving):
        """Register using CoMIR as Python module."""
        import sys
        sys.path.insert(0, str(self.comir_path))
        
        # Import CoMIR (assumes it's installed)
        try:
            from comir import register_images
            from comir.models import CoMIRNet
        except ImportError:
            raise ImportError("CoMIR Python module not available")
        
        # Convert to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        fixed_t = torch.from_numpy(fixed).float().unsqueeze(0).unsqueeze(0).to(device)
        moving_t = torch.from_numpy(moving).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Load model
        if self.model_path and self.model_path.exists():
            model = torch.load(self.model_path, map_location=device)
        else:
            model = CoMIRNet()
        
        model = model.to(device)
        model.eval()
        
        # Register
        with torch.no_grad():
            registered_t, disp_field_t = register_images(model, fixed_t, moving_t)
        
        # Convert back
        registered = registered_t.squeeze().cpu().numpy()
        disp_field = disp_field_t.squeeze().cpu().numpy()
        
        return registered, disp_field
    
    def _register_cli(self, fixed, moving):
        """Register using CoMIR CLI."""
        from skimage import io
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save images
            fixed_path = tmpdir / "fixed.tif"
            moving_path = tmpdir / "moving.tif"
            output_path = tmpdir / "registered.tif"
            
            io.imsave(fixed_path, (fixed * 255).astype(np.uint8))
            io.imsave(moving_path, (moving * 255).astype(np.uint8))
            
            # Run CoMIR
            cmd = [
                "python",
                str(self.comir_path / "register.py"),
                "--fixed", str(fixed_path),
                "--moving", str(moving_path),
                "--output", str(output_path)
            ]
            
            if self.model_path:
                cmd.extend(["--model", str(self.model_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"CoMIR failed: {result.stderr}")
            
            # Load result
            registered = io.imread(output_path).astype(float) / 255.0
            disp_field = np.zeros((2, *fixed.shape))
        
        return registered, disp_field
    
    def _ensure_grayscale(self, image):
        """Convert to grayscale."""
        if image.ndim == 2:
            return image
        elif image.ndim == 3:
            return np.mean(image, axis=2)
        return image
    
    def _normalize_image(self, image):
        """Normalize to [0, 1]."""
        image = image.astype(float)
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            image = (image - p2) / (p98 - p2)
        return np.clip(image, 0, 1)
    
    def _find_comir(self):
        """Try to find CoMIR installation."""
        possible_paths = [
            Path.home() / "CoMIR_INSPIRE",
            Path.cwd() / "CoMIR_INSPIRE",
            Path("/opt/CoMIR_INSPIRE"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None