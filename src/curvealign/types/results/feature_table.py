"""
Feature table data structure.

This module defines the structure for storing computed feature data.
"""

from typing import Dict
import numpy as np


# Feature table - structured array or DataFrame-like structure
FeatureTable = Dict[str, np.ndarray]
