import math


import math
import numpy as np


def round_mlab(num):
    """
    Rounding function specifically made to follow the MATLAB standard rather than
    Python's, where MATLAB rounds 0.5 up, while Python rounds 0.5 down.
    """
    if isinstance(num, np.ndarray):
        return np.floor(num + 0.5).astype(int)
    elif isinstance(num, (list, tuple)):
        return [math.floor(x + 0.5) for x in num]
    else:
        return math.floor(num + 0.5)
