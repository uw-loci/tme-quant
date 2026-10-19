"""Complex wrapping transform used by CT-FIRE's MATLAB reconstruction.

Adapted from Laurent Demanet's 2004 CurveLab MATLAB fdct_wrapping,
ifdct_wrapping, and fdct_wrapping_window routines. This implements the
complex transform with a finest wavelet (is_real=0, finest=2) used by CTrec_1;
coefficient layouts intentionally differ from CurveLab's C++ transform.
"""

import math

import numpy as np


def _round(x):
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def _fft(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x))) / math.sqrt(x.size)


def _ifft(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x))) * math.sqrt(x.size)


def _window(x):
    x = np.array(x, dtype=float, copy=True)
    x[np.abs(x) < 2**-52] = 0
    left = np.zeros_like(x)
    right = np.zeros_like(x)
    inside = (x > 0) & (x < 1)
    t = x[inside]
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        right[inside] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / t)))
        left[inside] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / (1 - t))))
    right[x <= 0] = 1
    left[x >= 1] = 1
    norm = np.sqrt(left**2 + right**2)
    return left / norm, right / norm


def _lowpass(m1, m2):
    ramps = []
    for m in (m1, m2):
        length = math.floor(2 * m) - math.floor(m) - 1
        if length < 0:
            raise ValueError("Image or scale count is too small for MATLAB wrapping")
        # MATLAB 0:(1/length):1 produces [0] when length is zero.
        t = np.arange(length + 1) / length if length else np.array([0.0])
        left, right = _window(t)
        ramps.append(np.concatenate((left, np.ones(2 * math.floor(m) + 1), right)))
    return ramps[0][:, None] * ramps[1][None, :]


def _center_indices(shape, subshape):
    return np.ix_(*[np.arange(n) + (s // 2 - n // 2) for s, n in zip(shape, subshape)])


def _wedges(m1, m2, angles):
    """Yield MATLAB-order wrapping maps and windows for each quadrant/wedge.

    Maps are stored in wrapped array order. Coordinates remain one-based while
    constructing geometry, matching the source's corner formulas.
    """
    perquad = angles // 4
    for q in range(1, 5):
        mh, mv = (m2, m1) if q % 2 else (m1, m2)
        h, v = math.floor(4 * mh), math.floor(4 * mv)
        ticks_left = _round(np.arange(perquad + 1) / (2 * perquad) * 2 * h + 1)
        ticks_right = 2 * h + 2 - ticks_left
        ticks = np.concatenate(
            (ticks_left, ticks_right[::-1] if perquad % 2 else ticks_right[-2::-1])
        )
        ends = ticks[1:-1:2]
        mids = (ends[:-1] + ends[1:]) / 2
        endpoint_v = int(_round(np.asarray(2 * v / (2 * perquad) + 1)))
        corner_length = v - math.floor(mv) + math.ceil(endpoint_v / 4)
        for sub in range(perquad):
            corner = sub in (0, perquad - 1)
            length = corner_length if corner else v - math.floor(mv)
            y = np.arange(1, length + 1)
            if sub == 0:
                width = int(ends[1] + ends[0] - 1)
                line = _round(2 - ends[0] + (h + 1 - ends[0]) / v * (y - 1))
            elif sub == perquad - 1:
                width = int(4 * h + 3 - ends[-1] - ends[-2])
                line = _round(ends[-2] + (h + 1 - ends[-1]) / v * (y - 1))
            else:
                width = int(ends[sub + 1] - ends[sub - 1] + 1)
                line = _round(ends[sub - 1] + (h + 1 - ends[sub]) / v * (y - 1))
            first_row = (
                v
                + 2
                - math.ceil((length + 1) / 2)
                + (length + 1) % 2 * (q - 2 == (q - 2) % 2)
            )
            first_col = (
                h
                + 2
                - math.ceil((width + 1) / 2)
                + (width + 1) % 2 * (q - 3 == (q - 3) % 2)
            )
            cols = (
                line[:, None]
                + (np.arange(width)[None, :] - (line[:, None] - first_col)) % width
            )
            new_rows = (y - first_row) % length
            admissible = np.clip(cols, 1, 2 * h + 1)
            xx = np.empty((length, width), dtype=float)
            yy = np.empty_like(xx)
            xx[new_rows] = admissible
            yy[new_rows] = y[:, None]

            def coord(mid, span):
                midline = mid + (h + 1 - mid) / v * (yy - 1)
                return 0.5 + v / span * (xx - midline) / (v + 1 - yy)

            if sub == 0:
                right_coord = coord(mids[0], ends[1] - ends[0])
                c2 = 1 / (
                    1 / (2 * h / (ends[0] - 1) - 1) + 1 / (2 * v / (endpoint_v - 1) - 1)
                )
                c1 = c2 / (2 * v / (endpoint_v - 1) - 1)
                xx[(xx - 1) / h + (yy - 1) / v == 2] += 1
                left_coord = c1 + c2 * ((xx - 1) / h - (yy - 1) / v) / (
                    2 - ((xx - 1) / h + (yy - 1) / v)
                )
            elif sub == perquad - 1:
                left_coord = coord(mids[-1], ends[-1] - ends[-2])
                c2 = -1 / (
                    2 * h / (ends[-1] - 1) - 1 + 1 / (2 * v / (endpoint_v - 1) - 1)
                )
                c1 = -c2 * (2 * h / (ends[-1] - 1) - 1)
                xx[(xx - 1) / h == (yy - 1) / v] -= 1
                right_coord = c1 + c2 * (2 - ((xx - 1) / h + (yy - 1) / v)) / (
                    (xx - 1) / h - (yy - 1) / v
                )
            else:
                left_coord = coord(mids[sub - 1], ends[sub] - ends[sub - 1])
                right_coord = coord(mids[sub], ends[sub + 1] - ends[sub])
            weight = _window(left_coord)[0] * _window(right_coord)[1]
            valid = (cols > 0) & (cols <= 2 * h + 1)
            yield q, sub, y - 1, admissible - 1, new_rows, valid, weight


def _angles(scales, coarse):
    return (
        [1] + [coarse * 2 ** math.ceil((j - 1) / 2) for j in range(1, scales - 1)] + [1]
    )


def forward(image, scales=None, coarse_angles=16):
    """Return complex MATLAB-layout coefficients, with a finest wavelet."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("MATLAB wrapping requires a two-dimensional image")
    if scales is None:
        scales = math.ceil(math.log2(min(image.shape)) - 3)
    if scales < 2 or coarse_angles < 8 or coarse_angles % 4:
        raise ValueError(
            "Need at least two scales and coarse angles a multiple of four >= 8"
        )
    angles = _angles(scales, coarse_angles)
    coefficients = [[] for _ in range(scales)]
    m1, m2 = np.array(image.shape) / 6
    low = _lowpass(m1, m2)
    spectrum = _fft(image)
    ix = _center_indices(spectrum.shape, low.shape)
    xlow = spectrum[ix] * low
    xhi = spectrum.copy()
    xhi[ix] *= np.sqrt(1 - low**2)
    coefficients[-1] = [_ifft(xhi)]
    for j in range(scales - 2, 0, -1):
        m1, m2 = m1 / 2, m2 / 2
        low = _lowpass(m1, m2)
        ix = _center_indices(xlow.shape, low.shape)
        xhi = xlow.copy()
        xhi[ix] = xlow[ix] * np.sqrt(1 - low**2)
        xlow = xlow[ix] * low
        quadrant = 1
        for q, sub, rows, cols, new_rows, valid, weight in _wedges(m1, m2, angles[j]):
            if q != quadrant:
                xhi = np.rot90(xhi)
                quadrant = q
            wrapped = np.empty(weight.shape, dtype=complex)
            wrapped[new_rows] = xhi[rows[:, None], cols] * valid
            coefficients[j].append(_ifft(np.rot90(wrapped * weight, -(q - 1))))
    coefficients[0] = [_ifft(xlow)]
    return coefficients


def inverse(coefficients):
    """Invert complex MATLAB-layout coefficients with a finest wavelet."""
    scales = len(coefficients)
    shape = coefficients[-1][0].shape
    angles = _angles(scales, len(coefficients[1]))
    m1, m2 = np.array(shape) / 6
    low = _lowpass(m1, m2)
    hipass_finest = np.sqrt(1 - low**2)
    spectrum = np.zeros(low.shape, dtype=complex)
    top = np.array([0, 0])
    for j in range(scales - 2, 0, -1):
        m1, m2 = m1 / 2, m2 / 2
        next_low = _lowpass(m1, m2)
        xj = np.zeros(
            (2 * math.floor(4 * m1) + 1, 2 * math.floor(4 * m2) + 1), dtype=complex
        )
        quadrant = 1
        for k, (q, sub, rows, cols, new_rows, valid, weight) in enumerate(
            _wedges(m1, m2, angles[j])
        ):
            if q != quadrant:
                xj = np.rot90(xj)
                quadrant = q
            wrapped = np.rot90(_fft(coefficients[j][k]), q - 1) * weight
            # MATLAB indexed assignment keeps the last repeated column. Right
            # corners reverse assignment order to retain the opposite edge.
            for row, columns, new_row in zip(rows, cols, new_rows):
                values = wrapped[new_row]
                if sub == angles[j] // 4 - 1:
                    columns, values = columns[::-1], values[::-1]
                xj[row, columns] = xj[row, columns] + values
        xj = np.rot90(xj)
        xj *= low
        ix = _center_indices(xj.shape, next_low.shape)
        xj[ix] *= np.sqrt(1 - next_low**2)
        loc = tuple(slice(int(t), int(t + n)) for t, n in zip(top, xj.shape))
        spectrum[loc] += xj
        top += [
            math.floor(4 * m1) - math.floor(2 * m1),
            math.floor(4 * m2) - math.floor(2 * m2),
        ]
        low = next_low
    coarsest = _fft(coefficients[0][0])
    loc = tuple(slice(int(t), int(t + n)) for t, n in zip(top, coarsest.shape))
    spectrum[loc] += coarsest * low
    finest = _fft(coefficients[-1][0])
    ix = _center_indices(shape, spectrum.shape)
    finest[ix] = finest[ix] * hipass_finest + spectrum
    return _ifft(finest)
