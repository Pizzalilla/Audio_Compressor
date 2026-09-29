"""
Short-time Fourier transform: framing, windowing, and overlap-add synthesis.

This is the layer that turns a one-shot transform into something that can
process a whole song. It depends on fft.py and nothing else.
"""

import math

from fft import fft, ifft, is_power_of_two
from filters import apply_gain


def hann_window(n):
    # Periodic Hann window of length n.
    # Dividing by n rather than n - 1 is what makes it periodic; the
    # symmetric variant does not satisfy the overlap-add condition exactly.
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / n)) for i in range(n)]


# Frames of silence added before and after the signal so that every real
# sample sits under a full set of overlapping windows.
LEAD_FRAMES = 1
TAIL_FRAMES = 2

# The window is applied on both analysis and synthesis, so the quantity that
# has to sum to a constant is the SQUARED window, not the window. Squared
# Hann satisfies that at a hop of a quarter frame and finer, but not at a
# half frame: sin^4 + cos^4 = 1 - sin^2(2x)/2, which swings between 0.5 and
# 1. The usual "Hann is COLA at 50% overlap" result is about the unsquared
# window and does not apply here.
MIN_OVERLAP_FACTOR = 4


def frame_positions(sample_count, frame_size, hop_size):
    # Where each frame starts, relative to the padded signal. process()
    # iterates over this; callers use len() on it to size a progress bar.
    padded_length = sample_count + (LEAD_FRAMES + TAIL_FRAMES) * frame_size
    return range(0, padded_length - frame_size + 1, hop_size)


def window_energy_sum(window, hop_size, total_length):
    """Accumulated sum of squared window values at each output index.

    process() divides by this to restore unity gain, and the tests check
    the overlap condition against it. Both use this one function so that
    what is tested is what actually runs.
    """
    frame_size = len(window)
    energy = [0.0] * total_length
    squared = [w * w for w in window]
    for position in range(0, total_length - frame_size + 1, hop_size):
        for i in range(frame_size):
            energy[position + i] += squared[i]
    return energy


def analyse(samples, position, window):
    # Cut one frame starting at position, taper it with the window, and
    # transform it. Returns the frame's spectrum.
    return fft([samples[position + i] * w for i, w in enumerate(window)])


def synthesise(spectrum, gain_curve, window):
    # Apply the gain curve, transform back, and taper again on the way out.
    #
    # The second taper is what makes spectral modification safe: changing
    # bins introduces discontinuities at the frame edges, and windowing
    # again suppresses them before the frames are summed. The imaginary
    # part is discarded, which is only sound because the gain curve is
    # symmetric and so preserves conjugate symmetry.
    restored = ifft(apply_gain(spectrum, gain_curve))
    return [restored[i].real * w for i, w in enumerate(window)]


def process(samples, sample_rate, gain_curve_fn, frame_size=2048,
            hop_size=None, progress_fn=None):
    """Run the full analysis-modify-synthesis loop over a signal.

    For each frame: window it, FFT it, multiply by the gain curve,
    inverse FFT, and accumulate into the output via overlap-add.

    The window is applied twice, once before the transform and once after.
    Windowing on the way out suppresses the discontinuities that spectral
    modification introduces at frame edges; dividing by the accumulated
    sum of squared windows then restores unity gain. With a flat curve this
    reproduces the input to floating-point precision.

    Args:
        samples: input signal as a list of floats.
        sample_rate: samples per second.
        gain_curve_fn: callable (n, sample_rate) -> list of n multipliers.
        frame_size: FFT size. Power of 2.
        hop_size: samples between consecutive frames. Defaults to 75% overlap.
        progress_fn: optional callable (frames_done, frames_total).

    Returns:
        list of floats, same length as samples.
    """
    if not is_power_of_two(frame_size):
        raise ValueError(f"frame_size must be a power of 2, got {frame_size}")
    if hop_size is None:
        hop_size = frame_size // MIN_OVERLAP_FACTOR
    if hop_size <= 0 or hop_size > frame_size // MIN_OVERLAP_FACTOR:
        raise ValueError(
            f"hop_size must be in 1..{frame_size // MIN_OVERLAP_FACTOR} for a "
            f"frame of {frame_size}; got {hop_size}. Coarser hops do not "
            f"satisfy the overlap condition for a squared Hann window, so the "
            f"result is amplitude-modulated once the spectrum is altered."
        )

    window = hann_window(frame_size)
    gain_curve = gain_curve_fn(frame_size, sample_rate)

    # Pad both ends so that every original sample sits under a full set of
    # overlapping windows. Without this the first and last frame_size samples
    # would be attenuated by the window's taper.
    original_length = len(samples)
    padded = ([0.0] * (LEAD_FRAMES * frame_size) + list(samples)
              + [0.0] * (TAIL_FRAMES * frame_size))

    positions = frame_positions(original_length, frame_size, hop_size)
    total_frames = len(positions)

    accumulated = [0.0] * len(padded)
    window_energy = window_energy_sum(window, hop_size, len(padded))

    for frame_index, position in enumerate(positions):
        spectrum = analyse(padded, position, window)
        frame = synthesise(spectrum, gain_curve, window)

        for i in range(frame_size):
            accumulated[position + i] += frame[i]

        if progress_fn is not None and frame_index % 16 == 0:
            progress_fn(frame_index, total_frames)

    if progress_fn is not None:
        progress_fn(total_frames, total_frames)

    output = []
    start = LEAD_FRAMES * frame_size
    for i in range(start, start + original_length):
        energy = window_energy[i]
        output.append(accumulated[i] / energy if energy > 1e-8 else 0.0)
    return output


def limit_peak(channels, ceiling=1.0):
    """Scale channels down if they would clip, leave them alone otherwise.

    Boosting a band adds energy, so the equalised signal routinely exceeds
    the [-1, 1] range a WAV file can store. Writing it anyway truncates
    every offending sample, which is audible as harsh distortion rather
    than as the boost the user asked for.

    Every channel is scaled by the same factor, derived from the loudest
    peak across all of them. Scaling channels independently would move the
    stereo image whenever one side happened to peak higher than the other.
    Within a channel the attenuation is uniform too, so this changes how
    loud the result is, not its spectrum.

    The ceiling is 1.0 because that is exactly what write_wav can store.
    Setting it lower means a file already mastered close to full scale gets
    attenuated, and reported as attenuated, for no reason.

    Args:
        channels: list of channels, each a list of floats.

    Returns:
        (scaled_channels, attenuation_db, original_peak). attenuation_db is
        0.0 when nothing needed doing.
    """
    peak = max((max(abs(v) for v in c) for c in channels if c), default=0.0)

    # Only ever turn things down. Scaling quiet signals up would change the
    # output for cases that were already correct, and would break the
    # guarantee that a flat curve reproduces the input exactly.
    if peak <= ceiling:
        return [list(c) for c in channels], 0.0, peak

    scale = ceiling / peak
    return ([[v * scale for v in c] for c in channels],
            -20.0 * math.log10(scale),
            peak)


