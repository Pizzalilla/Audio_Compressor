"""
Matplotlib figures for the Streamlit front end.

Each function returns a Figure so the caller decides whether to show it,
save it, or hand it to Streamlit.

These use the project's own FFT rather than a library one, which means the
number of frames analysed has to be capped to keep plotting responsive.
Averaging a few dozen frames is enough for a readable spectrum anyway.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft import fft
from filters import BANDS, band_centres
from stft import hann_window

FLOOR_DB = -120.0
# The magnitude that FLOOR_DB corresponds to. Comparing against an unrelated
# epsilon lets values far below the floor through and stretches the axis.
FLOOR_MAGNITUDE = 10.0 ** (FLOOR_DB / 20.0)


def _to_db(magnitude):
    return 20.0 * math.log10(max(magnitude, FLOOR_MAGNITUDE))


def average_spectrum(samples, sample_rate, frame_size=2048, max_frames=24):
    """Magnitude spectrum averaged over frames spread across the signal.

    Returns:
        (frequencies, magnitudes_db) covering 0 Hz up to Nyquist.
    """
    usable = len(samples) - frame_size
    if usable <= 0:
        frame_starts = [0]
        padded = list(samples) + [0.0] * (frame_size - len(samples))
        samples = padded
    else:
        count = min(max_frames, max(1, usable // frame_size))
        step = usable / count
        frame_starts = [int(i * step) for i in range(count)]

    window = hann_window(frame_size)
    half = frame_size // 2
    totals = [0.0] * half

    for start in frame_starts:
        frame = [samples[start + i] * window[i] for i in range(frame_size)]
        spectrum = fft(frame)
        for k in range(half):
            totals[k] += abs(spectrum[k])

    scale = len(frame_starts) * frame_size / 2.0
    frequencies = [k * sample_rate / frame_size for k in range(half)]
    magnitudes = [_to_db(total / scale) for total in totals]
    return frequencies, magnitudes


def _style_frequency_axis(axis, sample_rate, show_bands=True):
    axis.set_xscale("log")
    axis.set_xlim(20, min(20000, sample_rate / 2))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.grid(True, alpha=0.3, which="both")
    if show_bands:
        for low, high in BANDS.values():
            axis.axvline(low, color="grey", alpha=0.25, linestyle="--", linewidth=0.8)


def plot_waveform(samples, sample_rate, title="Waveform"):
    # Amplitude against time.
    figure, axis = plt.subplots(figsize=(10, 2.5))
    # Drawing every sample of a long file is pointless at screen resolution.
    stride = max(1, len(samples) // 4000)
    reduced = samples[::stride]
    times = [i * stride / sample_rate for i in range(len(reduced))]
    axis.plot(times, reduced, linewidth=0.6)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_spectrum_comparison(before, after, sample_rate, band_gains_db=None):
    # Original and processed spectra stacked for comparison.
    before_freqs, before_db = average_spectrum(before, sample_rate)
    after_freqs, after_db = average_spectrum(after, sample_rate)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    top.plot(before_freqs, before_db, linewidth=0.8, color="#3b7dd8")
    top.set_title("Original")
    _style_frequency_axis(top, sample_rate)

    bottom.plot(after_freqs, after_db, linewidth=0.8, color="#d8663b")
    bottom.set_title("Equalised")
    _style_frequency_axis(bottom, sample_rate)

    if band_gains_db:
        # Label each band where its gain is actually applied. band_centres
        # clamps to the file's Nyquist frequency and drops bands that fall
        # above it, so a label only appears if the control did something.
        for centre, gain, name in band_centres(band_gains_db, sample_rate):
            if gain == 0.0:
                continue
            bottom.annotate(
                f"{name} {gain:+.0f} dB",
                xy=(centre, 0.93), xycoords=("data", "axes fraction"),
                ha="center", fontsize=8, alpha=0.75,
            )

    figure.tight_layout()
    return figure


def plot_spectrogram(samples, sample_rate, frame_size=1024, max_columns=160):
    # Time on the x-axis, frequency on the y-axis, magnitude as colour.
    # Hop size is chosen from max_columns rather than fixed, so a long file
    # costs the same to draw as a short one.
    half = frame_size // 2
    usable = max(1, len(samples) - frame_size)
    hop = max(1, usable // max_columns)

    window = hann_window(frame_size)
    columns = []
    position = 0
    while position + frame_size <= len(samples):
        frame = [samples[position + i] * window[i] for i in range(frame_size)]
        spectrum = fft(frame)
        columns.append([_to_db(abs(spectrum[k]) * 2.0 / frame_size)
                        for k in range(half)])
        position += hop

    if not columns:
        columns = [[FLOOR_DB] * half]

    # Transpose so rows are frequency and columns are time.
    grid = [[columns[t][k] for t in range(len(columns))] for k in range(half)]

    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(
        grid, origin="lower", aspect="auto", cmap="magma", vmin=-100, vmax=0,
        extent=[0, len(samples) / sample_rate, 0, sample_rate / 2],
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0, min(16000, sample_rate / 2))
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    return figure
