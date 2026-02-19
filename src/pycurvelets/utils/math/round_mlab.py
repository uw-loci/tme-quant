import math


def round_mlab(num):
    """
    Rounding function specifically made to follow the MATLAB standard rather than
    Python's, where MATLAB rounds 0.5 up, while Python rounds 0.5 down.
    """
    if isinstance(num, (list, tuple)):
        return [math.floor(x + 0.5) for x in num]
    elif hasattr(num, "__iter__"):  # e.g., numpy array
        return [math.floor(float(x) + 0.5) for x in num]
    else:
        return math.floor(num + 0.5)
