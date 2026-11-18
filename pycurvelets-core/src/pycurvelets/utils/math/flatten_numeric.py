import numpy as np


def flatten_numeric(series):
    """
    Convert a pandas Series of numbers or 1-element arrays into a flat float array.
    Mainly used for testing.
    """
    arr = series.to_numpy()

    # If it's object dtype (arrays of arrays), flatten manually
    if arr.dtype == object:
        flat = []
        for x in arr:
            if isinstance(x, (list, np.ndarray)):
                # unwrap single-element arrays safely
                if np.ndim(x) == 0:
                    flat.append(float(x))
                elif len(np.ravel(x)) == 1:
                    flat.append(float(np.ravel(x)[0]))
                else:
                    # multi-element arrays (shouldn't happen) — extend instead
                    flat.extend(np.ravel(x).astype(float))
            else:
                flat.append(float(x))
        return np.array(flat, dtype=float)
    else:
        return arr.astype(float).ravel()
