"""
Tests for framing, windowing, and overlap-add reconstruction.
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fft import fft, ifft
from filters import apply_gain, graphic_eq_curve
from stft import (analyse, frame_positions, hann_window, limit_peak, process,
                  synthesise, window_energy_sum)


def cola_sum(window, hop_size, num_frames=16):
    # The fully-covered middle of window_energy_sum, where every index sits
    # under a complete set of overlapping frames. Constant here means the
    # overlap condition holds. Reads the same function process() uses, so
    # what is checked is what actually runs.
    frame_size = len(window)
    energy = window_energy_sum(window, hop_size,
                               frame_size * (num_frames + 2))
    return energy[frame_size * 2:frame_size * num_frames]


def music_like_signal(n, sample_rate=8000, seed=0):
    # A few tones plus noise: enough spectral content to exercise the bands.
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i / sample_rate
        value = (0.4 * math.sin(2 * math.pi * 110 * t)
                 + 0.3 * math.sin(2 * math.pi * 880 * t)
                 + 0.2 * math.sin(2 * math.pi * 3000 * t)
                 + 0.05 * rng.uniform(-1, 1))
        out.append(value)
    return out


def test_hann_window_shape():
    window = hann_window(8)
    assert window[0] == pytest.approx(0.0)
    assert window[4] == pytest.approx(1.0)
    # Periodic, so the window is symmetric about its peak but does not
    # return to zero at the final sample.
    assert window[1] == pytest.approx(window[7])
    assert window[3] == pytest.approx(window[5])


def test_cola_hann_75_percent():
    # process() windows twice, once before the transform and once after, so
    # the quantity that has to come out constant is the sum of SQUARED
    # windows, not the sum of windows. That distinction matters: the usual
    # "Hann is COLA at 50% overlap" result is about the unsquared window.
    window = hann_window(64)

    at_75 = cola_sum(window, 16)
    assert max(at_75) - min(at_75) < 1e-9
    assert at_75[0] == pytest.approx(1.5)

    # Squared Hann at 50% is not constant, because
    # sin^4 + cos^4 = 1 - sin^2(2x)/2, which swings between 0.5 and 1.
    at_50 = cola_sum(window, 32)
    assert max(at_50) == pytest.approx(1.0)
    assert min(at_50) == pytest.approx(0.5)

    # A hop that does not divide the window evenly is worse still.
    ragged = cola_sum(window, 25)
    assert max(ragged) - min(ragged) > 1e-3


def test_rejects_hops_that_cannot_reconstruct():
    # Dividing by the accumulated window energy makes a flat curve
    # reconstruct at any hop, which is misleading: once the spectrum is
    # modified the compensation no longer holds and the output is amplitude
    # modulated at the frame rate. Anything coarser than a quarter frame is
    # refused rather than silently degrading.
    for hop in [128, 192, 256]:
        with pytest.raises(ValueError, match="overlap"):
            process([0.0] * 1000, 8000, lambda n, sr: [1.0] * n,
                    frame_size=256, hop_size=hop)


def test_accepted_hops_reconstruct_exactly():
    signal = music_like_signal(4000, seed=7)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    for hop in [64, 32, 16]:
        output = process(signal, 8000,
                         lambda n, sr: graphic_eq_curve(n, sr, flat),
                         frame_size=256, hop_size=hop)
        for original, restored in zip(signal, output):
            assert abs(original - restored) < 1e-9, f"hop {hop}"


def test_modified_spectrum_introduces_little_stray_content():
    # The real test of the overlap condition: feed a pure tone, cut a band
    # it does not occupy, and check nothing new appears at the frame rate.
    # A hop that fails the condition shows up here as sidebands.
    sample_rate, frame_size = 8000, 256
    signal = [math.sin(2 * math.pi * 1000 * i / sample_rate)
              for i in range(4000)]
    output = process(signal, sample_rate,
                     lambda n, sr: graphic_eq_curve(n, sr, {"bass": -18.0}),
                     frame_size=frame_size)

    spectrum = fft(output[1000:1000 + 1024])
    tone_bin = round(1000 * 1024 / sample_rate)
    tone = abs(spectrum[tone_bin])
    stray = max(abs(spectrum[k]) for k in range(1, 512)
                if abs(k - tone_bin) > 3)
    assert 20 * math.log10(stray / tone) < -40


def test_analyse_applies_window_and_transform():
    # A frame of constant 1.0 windowed by Hann has DC equal to the sum of
    # the window, which for a periodic Hann is exactly n / 2.
    signal = [1.0] * 128
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)
    assert len(spectrum) == 64
    assert spectrum[0].real == pytest.approx(32.0)
    assert spectrum[0].imag == pytest.approx(0.0)


def test_analyse_reads_from_the_given_offset():
    signal = [0.0] * 64 + [1.0] * 64
    window = hann_window(64)
    assert abs(analyse(signal, 0, window)[0]) == pytest.approx(0.0)
    assert analyse(signal, 64, window)[0].real == pytest.approx(32.0)


def test_synthesise_is_windowed_identity_under_flat_gain():
    # With a flat curve, synthesise undoes analyse except for the second
    # taper, so the result is the input multiplied by the window squared.
    signal = music_like_signal(128)
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)
    frame = synthesise(spectrum, [1.0] * 64, window)
    for i in range(64):
        assert frame[i] == pytest.approx(signal[i] * window[i] ** 2, abs=1e-12)


def test_synthesise_discards_nothing_meaningful():
    # synthesise() takes .real and drops the imaginary part. Asserting the
    # result is a float proves nothing -- .real is always a float. What
    # matters is that the discarded part is numerical noise, which holds
    # only because the gain curve is symmetric.
    signal = music_like_signal(128, seed=9)
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)

    curve = graphic_eq_curve(64, 8000, {"bass": 6.0, "treble": -6.0})
    discarded = max(abs(z.imag) for z in ifft(apply_gain(spectrum, curve)))
    kept = max(abs(z.real) for z in ifft(apply_gain(spectrum, curve)))
    assert discarded < 1e-12 * max(kept, 1.0)


def test_asymmetric_curve_would_discard_real_signal():
    # Guards the reasoning above: if the curve were not symmetric, taking
    # .real would throw away actual signal, and the old isinstance-based
    # test could not tell the difference.
    signal = music_like_signal(128, seed=9)
    window = hann_window(64)
    spectrum = analyse(signal, 0, window)

    lopsided = [1.0] * 32 + [0.0] * 32
    discarded = max(abs(z.imag) for z in ifft(apply_gain(spectrum, lopsided)))
    assert discarded > 1e-6


def test_flat_eq_is_identity():
    # With all gains at 0 dB, output should match input. End-to-end version
    # of the Parseval check: if the whole chain is correct, flat EQ is a no-op.
    signal = music_like_signal(4000)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    output = process(
        signal, 8000,
        lambda n, sr: graphic_eq_curve(n, sr, flat),
        frame_size=256,
    )
    assert len(output) == len(signal)
    for original, restored in zip(signal, output):
        assert abs(original - restored) < 1e-9


def test_energy_is_preserved_under_flat_eq():
    signal = music_like_signal(4000, seed=1)
    flat = {"bass": 0.0, "mid": 0.0, "treble": 0.0}
    output = process(signal, 8000, lambda n, sr: graphic_eq_curve(n, sr, flat),
                     frame_size=256)
    before = sum(v * v for v in signal)
    after = sum(v * v for v in output)
    assert after == pytest.approx(before, rel=1e-9)


def test_boost_and_cut_change_energy_in_the_right_direction():
    signal = music_like_signal(4000, seed=2)
    baseline = sum(v * v for v in signal)

    boosted = process(signal, 8000,
                      lambda n, sr: graphic_eq_curve(n, sr, {"bass": 12.0}),
                      frame_size=256)
    assert sum(v * v for v in boosted) > baseline

    cut = process(signal, 8000,
                  lambda n, sr: graphic_eq_curve(n, sr, {"bass": -12.0}),
                  frame_size=256)
    assert sum(v * v for v in cut) < baseline


def test_output_stays_bounded():
    signal = music_like_signal(2000, seed=3)
    output = process(signal, 8000,
                     lambda n, sr: graphic_eq_curve(n, sr, {"treble": 9.0}),
                     frame_size=256)
    # +9 dB is a factor of 2.8, so anything beyond a few times the input
    # peak means the overlap-add normalisation has gone wrong.
    assert max(abs(v) for v in output) < 4 * max(abs(v) for v in signal)
    assert all(abs(v) < 10.0 for v in output)


def test_limit_peak_leaves_quiet_signals_alone():
    # Anything already within range must pass through untouched, or the
    # flat-EQ identity would no longer hold. A full-scale master sits at
    # exactly 1.0 and must not be attenuated.
    for signal in ([0.5, -0.25, 0.75, -0.9], [1.0, -1.0, 0.3]):
        scaled, attenuation, peak = limit_peak([signal])
        assert scaled == [signal]
        assert attenuation == 0.0


def test_limit_peak_scales_down_when_clipping():
    signal = [1.5, -0.75, 3.0, -1.2]
    scaled, attenuation, peak = limit_peak([signal])
    assert peak == pytest.approx(3.0)
    assert max(abs(v) for v in scaled[0]) <= 1.0
    assert attenuation == pytest.approx(20 * math.log10(3.0), abs=1e-6)


def test_limit_peak_preserves_relative_balance():
    # Uniform attenuation, so every ratio between samples survives.
    signal = [2.0, -1.0, 0.5, 4.0]
    scaled, _, _ = limit_peak([signal])
    for i in range(len(signal) - 1):
        assert scaled[0][i] / scaled[0][i + 1] == pytest.approx(
            signal[i] / signal[i + 1])


def test_limit_peak_scales_channels_together():
    # One shared factor, derived from the loudest channel. Scaling each
    # channel to its own peak would move the stereo image.
    left, right = [2.0, -2.0], [0.5, -0.5]
    scaled, _, peak = limit_peak([left, right])
    assert peak == pytest.approx(2.0)
    assert scaled[0] == pytest.approx([1.0, -1.0])
    assert scaled[1] == pytest.approx([0.25, -0.25])


def test_limit_peak_handles_empty_and_silent_input():
    assert limit_peak([]) == ([], 0.0, 0.0)
    assert limit_peak([[]]) == ([[]], 0.0, 0.0)
    scaled, attenuation, peak = limit_peak([[0.0, 0.0]])
    assert scaled == [[0.0, 0.0]] and attenuation == 0.0 and peak == 0.0


def test_boosted_output_no_longer_clips():
    # The end-to-end version: the bass-boost preset used to clip 45% of
    # samples on a normal-level track.
    signal = [0.8 * math.sin(2 * math.pi * 150 * i / 8000) for i in range(4000)]
    raw = process(signal, 8000,
                  lambda n, sr: graphic_eq_curve(n, sr, {"bass": 8.0}),
                  frame_size=256)
    assert max(abs(v) for v in raw) > 1.0, "expected the raw boost to overshoot"

    limited, attenuation, _ = limit_peak([raw])
    assert max(abs(v) for v in limited[0]) <= 1.0
    assert attenuation > 0


def test_frame_positions_matches_what_process_iterates():
    # The app sizes its progress bar from this, so it has to agree with the
    # loop exactly. It used to be recomputed by hand and drifted by one.
    for length, frame_size, hop in [(4000, 256, 64), (66150, 2048, 512),
                                    (1, 256, 64), (0, 256, 32)]:
        positions = frame_positions(length, frame_size, hop)
        seen = []
        process(list(music_like_signal(length)), 8000,
                lambda n, sr: [1.0] * n,
                frame_size=frame_size, hop_size=hop,
                progress_fn=lambda done, total: seen.append(total))
        assert seen and seen[-1] == len(positions)


def test_every_sample_is_fully_covered_by_windows():
    # The padding exists so no real sample sits under a partial window.
    length, frame_size, hop = 2000, 256, 64
    window = hann_window(frame_size)
    padded_length = length + 3 * frame_size
    energy = [0.0] * padded_length
    for position in frame_positions(length, frame_size, hop):
        for i in range(frame_size):
            energy[position + i] += window[i] ** 2
    covered = energy[frame_size:frame_size + length]
    assert min(covered) == pytest.approx(max(covered))


def test_rejects_bad_frame_size():
    with pytest.raises(ValueError):
        process([0.0] * 100, 8000, lambda n, sr: [1.0] * n, frame_size=300)
