"""
Curvelet coefficient structures.

This module defines data structures for representing curvelet transform
coefficients and related data.
"""

from typing import List
import numpy as np


# Curvelet coefficient structure - list of scales, each containing wedges
CtCoeffs = List[List[np.ndarray]]
