"""
Frequency-domain gain curves.

Everything here answers one question: given a frame of N FFT bins taken at
some sample rate, what should each bin be multiplied by?

Nothing in this module performs a transform. It only builds gain arrays.
"""

import math

# Band edges in Hz. Adjust to taste.
BANDS = {
    "bass": (20, 250),
    "mid": (250, 4000),
    "treble": (4000, 20000),
}


def bin_frequencies(n, sample_rate):
    """Centre frequency of each of the n FFT bins, in Hz.

    Bins above n // 2 correspond to negative frequencies.
    """
    freqs = []
    for k in range(n):
        if k <= n // 2:
            freqs.append(k * sample_rate / n)
        else:
            freqs.append((k - n) * sample_rate / n)
    return freqs


def db_to_linear(db):
    # Convert a decibel gain to a linear multiplier.
    return 10.0 ** (db / 20.0)


def band_centres(band_gains_db, sample_rate=None):
    """Geometric centre of each band paired with its requested gain.

    Geometric rather than arithmetic because pitch is logarithmic: the
    midpoint of 20-250 Hz sounds like ~70 Hz, not 135 Hz.

    When a sample rate is given, bands are clipped to the Nyquist frequency
    first. Without that, a band lying wholly or partly above Nyquist has a
    centre no bin can reach, so its slider silently under-delivers: at an
    8 kHz sample rate the treble centre sits at 8944 Hz and asking for
    +12 dB gets you +7.6 dB at best.

    Returns:
        list of (frequency, gain_db) sorted by frequency. Bands entirely
        above Nyquist are dropped.
    """
    nyquist = sample_rate / 2.0 if sample_rate else None
    centres = []
    for name, (low, high) in BANDS.items():
        if nyquist is not None:
            if low >= nyquist:
                continue
            high = min(high, nyquist)
        centres.append((math.sqrt(low * high), band_gains_db.get(name, 0.0)))
    centres.sort()
    return centres


def _gain_db_at(freq, centres):
    # Piecewise-linear interpolation between band centres, in log-frequency
    # space. Outside the outermost centres the nearest gain is held flat.
    if freq <= centres[0][0]:
        return centres[0][1]
    if freq >= centres[-1][0]:
        return centres[-1][1]

    for i in range(len(centres) - 1):
        low_freq, low_gain = centres[i]
        high_freq, high_gain = centres[i + 1]
        if low_freq <= freq <= high_freq:
            span = math.log2(high_freq) - math.log2(low_freq)
            position = (math.log2(freq) - math.log2(low_freq)) / span
            return low_gain + position * (high_gain - low_gain)

    return 0.0


def graphic_eq_curve(n, sample_rate, band_gains_db):
    """Build a per-bin gain array for a multi-band graphic equaliser.

    Gains are interpolated smoothly between band centres rather than
    switching abruptly at band edges. A step change in the gain curve is a
    brick-wall filter, whose impulse response is long enough to wrap around
    the frame and produce audible clicks.

    Args:
        n: FFT size (number of bins).
        sample_rate: samples per second.
        band_gains_db: dict mapping band name (see BANDS) to gain in dB.

    Returns:
        list of n real multipliers. Symmetric about n // 2, so the filtered
        spectrum stays conjugate-symmetric and the inverse transform comes
        back real.
    """
    centres = band_centres(band_gains_db, sample_rate)
    if not centres:
        return [1.0] * n

    # Taking the magnitude here is what makes the curve symmetric: bin k and
    # bin n - k differ only in the sign of their frequency.
    curve = [db_to_linear(_gain_db_at(abs(f), centres))
             for f in bin_frequencies(n, sample_rate)]

    # Bin 0 is 0 Hz: not a frequency anyone is equalising, but the constant
    # offset of the frame. It sits below every band so it would otherwise
    # inherit the lowest band's gain, and a recording with a DC offset would
    # have that offset amplified along with the bass, eating headroom for no
    # audible benefit.
    curve[0] = 1.0
    return curve


def apply_gain(spectrum, gain_curve):
    # Multiply each bin of spectrum by the matching entry of gain_curve.
    if len(spectrum) != len(gain_curve):
        raise ValueError(
            f"spectrum has {len(spectrum)} bins but gain curve has {len(gain_curve)}"
        )
    return [z * g for z, g in zip(spectrum, gain_curve)]
