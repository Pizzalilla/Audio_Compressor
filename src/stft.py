"""
Short-time Fourier transform: framing, windowing, and overlap-add synthesis.

This is the layer that turns a one-shot transform into something that can
process a whole song. It depends on fft.py and nothing else.
"""

import math

from fft import fft, ifft, is_power_of_two


def hann_window(n):
    # Periodic Hann window of length n.
    # Dividing by n rather than n - 1 is what makes it periodic; the
    # symmetric variant does not satisfy the overlap-add condition exactly.
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / n)) for i in range(n)]


# Frames of silence added before and after the signal so that every real
# sample sits under a full set of overlapping windows.
LEAD_FRAMES = 1
TAIL_FRAMES = 2


def frame_positions(sample_count, frame_size, hop_size):
    # Where each frame starts, relative to the padded signal. process()
    # iterates over this; callers use len() on it to size a progress bar.
    padded_length = sample_count + (LEAD_FRAMES + TAIL_FRAMES) * frame_size
    return range(0, padded_length - frame_size + 1, hop_size)


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
    restored = ifft([z * g for z, g in zip(spectrum, gain_curve)])
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
        hop_size = frame_size // 4
    if hop_size <= 0 or hop_size > frame_size:
        raise ValueError(f"hop_size must be in 1..{frame_size}, got {hop_size}")

    window = hann_window(frame_size)
    gain_curve = gain_curve_fn(frame_size, sample_rate)

    # Pad both ends so that every original sample sits under a full set of
    # overlapping windows. Without this the first and last frame_size samples
    # would be attenuated by the window's taper.
    original_length = len(samples)
    padded = ([0.0] * (LEAD_FRAMES * frame_size) + list(samples)
              + [0.0] * (TAIL_FRAMES * frame_size))

    accumulated = [0.0] * len(padded)
    window_energy = [0.0] * len(padded)

    positions = frame_positions(original_length, frame_size, hop_size)
    total_frames = len(positions)

    for frame_index, position in enumerate(positions):
        spectrum = analyse(padded, position, window)
        frame = synthesise(spectrum, gain_curve, window)

        for i in range(frame_size):
            accumulated[position + i] += frame[i]
            window_energy[position + i] += window[i] * window[i]

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


def limit_peak(samples, ceiling=0.999):
    """Scale a signal down if it would clip, leave it alone otherwise.

    Boosting a band adds energy, so the equalised signal routinely exceeds
    the [-1, 1] range a WAV file can store. Writing it anyway truncates
    every offending sample, which is audible as harsh distortion rather
    than as the boost the user asked for.

    Attenuation is uniform, so the relative balance between frequencies is
    untouched: this only changes how loud the result is, not its spectrum.

    Returns:
        (scaled_samples, attenuation_db, original_peak). attenuation_db is
        0.0 when nothing needed doing.
    """
    if not samples:
        return [], 0.0, 0.0

    peak = max(abs(v) for v in samples)
    # Only ever turn things down. Scaling quiet signals up would change the
    # output for cases that were already correct, and would break the
    # guarantee that a flat curve reproduces the input exactly.
    if peak <= ceiling:
        return list(samples), 0.0, peak

    scale = ceiling / peak
    return ([v * scale for v in samples],
            -20.0 * math.log10(scale),
            peak)


def cola_sum(window, hop_size, num_frames=16):
    # Sum of the squared window shifted by hop_size, sampled in the middle
    # where coverage is complete. Used by the tests to check the COLA
    # condition holds for a given window and hop.
    frame_size = len(window)
    total_length = frame_size * (num_frames + 2)
    accumulator = [0.0] * total_length
    position = 0
    while position + frame_size <= total_length:
        for i in range(frame_size):
            accumulator[position + i] += window[i] * window[i]
        position += hop_size
    return accumulator[frame_size * 2:frame_size * num_frames]
