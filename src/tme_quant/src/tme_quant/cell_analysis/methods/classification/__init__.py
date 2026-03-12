"""
Cell classification methods.
"""

from .base_classification import BaseClassificationMethod

__all__ = [
    'BaseClassificationMethod',
]

# Optional: additional classification methods (not yet created)
try:
    from .morphology_classifier import MorphologyClassifier
    __all__.append('MorphologyClassifier')
except ImportError:
    pass

try:
    from .marker_classifier import MarkerClassifier
    __all__.append('MarkerClassifier')
except ImportError:
    pass