"""
Load pre-trained models for cell segmentation.

Handles model downloading, caching and validation for StarDist and Cellpose.
"""

from pathlib import Path
from typing import Optional
import warnings


class ModelLoader:
    """Load and manage pre-trained segmentation models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for caching models (default: ~/.tme_quant/models)
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.tme_quant' / 'models'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_stardist_model(
        self,
        model_name: str = "2D_versatile_fluo",
        dimension: str = "2D",
    ):
        """
        Load a pre-trained StarDist model.

        Args:
            model_name: e.g. "2D_versatile_fluo", "2D_versatile_he"
            dimension: "2D" or "3D"

        Returns:
            Loaded StarDist model, or None on failure.
        """
        try:
            if dimension == "2D":
                from stardist.models import StarDist2D
                return StarDist2D.from_pretrained(model_name)
            else:
                from stardist.models import StarDist3D
                return StarDist3D.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "StarDist not installed. Install with: pip install stardist"
            )
        except Exception as e:
            warnings.warn(f"Failed to load StarDist model {model_name!r}: {e}")
            return None

    def load_cellpose_model(
        self,
        model_type: str = "nuclei",
        gpu: bool = False,
    ):
        """
        Load a pre-trained Cellpose model.

        Args:
            model_type: "nuclei", "cyto", or "cyto2"
            gpu: Use GPU if available.

        Returns:
            Loaded Cellpose model, or None on failure.
        """
        try:
            from cellpose import models
            import torch
            if gpu and not torch.cuda.is_available():
                warnings.warn("GPU requested but not available, using CPU")
                gpu = False
            return models.Cellpose(gpu=gpu, model_type=model_type)
        except ImportError:
            raise ImportError(
                "Cellpose not installed. Install with: pip install cellpose"
            )
        except Exception as e:
            warnings.warn(f"Failed to load Cellpose model {model_type!r}: {e}")
            return None

    def list_available_models(self) -> dict:
        """Return a dictionary of available pre-trained models by framework."""
        return {
            'stardist_2d': ['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'],
            'stardist_3d': ['3D_demo'],
            'cellpose': ['nuclei', 'cyto', 'cyto2'],
        }
