"""
Tests for the equaliser gain curves.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from filters import (BANDS, _gain_db_at, apply_gain, band_centres,
                     bin_frequencies, db_to_linear, graphic_eq_curve)


def test_db_to_linear():
    assert db_to_linear(0) == pytest.approx(1.0)
    assert db_to_linear(6) == pytest.approx(1.995, abs=1e-3)
    assert db_to_linear(-6) == pytest.approx(0.501, abs=1e-3)
    assert db_to_linear(20) == pytest.approx(10.0)


def test_bin_frequencies_are_symmetric():
    freqs = bin_frequencies(8, 8000)
    assert freqs[0] == 0.0
    assert freqs[4] == 4000.0
    # Bins past the midpoint are the negative frequencies.
    assert freqs[5] == -3000.0
    assert freqs[1] == -freqs[7]


def test_curve_is_symmetric():
    # Conjugate symmetry is what keeps the inverse transform real.
    curve = graphic_eq_curve(64, 8000, {"bass": 9.0, "treble": -9.0})
    for k in range(1, 32):
        assert curve[k] == pytest.approx(curve[64 - k])


def test_gains_are_exact_at_band_centres():
    # The claim is about the interpolation, not about any particular bin
    # landing on a centre. At 44.1 kHz with a 4096-point frame the bins are
    # 10.8 Hz apart, so the nearest one to the 70.7 Hz bass centre is 4.7 Hz
    # away, which on a log scale is far enough to read a few tenths low.
    gains = {"bass": 9.0, "mid": -6.0, "treble": 3.0}
    centres = band_centres(gains, 44100)
    for centre, expected, _ in centres:
        assert _gain_db_at(centre, centres) == pytest.approx(expected)


def test_curve_gets_close_to_the_requested_gain_near_each_centre():
    gains = {"bass": 9.0, "mid": -6.0, "treble": 3.0}
    curve = graphic_eq_curve(4096, 44100, gains)
    freqs = bin_frequencies(4096, 44100)
    for centre, expected, _ in band_centres(gains, 44100):
        k = min(range(2048), key=lambda i: abs(freqs[i] - centre))
        assert 20 * math.log10(curve[k]) == pytest.approx(expected, abs=0.5)


def test_flat_request_is_unity():
    assert graphic_eq_curve(64, 8000, {}) == [1.0] * 64
    assert graphic_eq_curve(64, 8000, {"bass": 0.0}) == pytest.approx([1.0] * 64)


def test_dc_bin_follows_the_lowest_band():
    # Bin 0 spans 0 Hz up to half a bin width, which at a 512-sample frame
    # is 43 Hz of genuine bass. Exempting it from the bass control leaves
    # that content unequalised and puts a step in the curve, so it gets the
    # same gain as the rest of the band.
    for gain in [-24.0, -6.0, 6.0, 24.0]:
        curve = graphic_eq_curve(512, 44100, {"bass": gain})
        assert 20 * math.log10(curve[0]) == pytest.approx(gain, abs=0.01)


def test_curve_has_no_step_at_the_bottom():
    # A discontinuity between adjacent bins is a brick-wall filter, which
    # rings. Neighbouring bins should differ by a fraction of a dB.
    curve = graphic_eq_curve(2048, 44100, {"bass": 12.0})
    for k in range(4):
        step = abs(20 * math.log10(curve[k + 1] / curve[k]))
        assert step < 1.0, f"step of {step:.1f} dB between bins {k} and {k+1}"


def test_band_centres_are_clamped_to_nyquist():
    # Without clamping, the treble centre sits at 8944 Hz regardless of
    # sample rate. At 22.05 kHz no bin reaches it, so asking for +12 dB
    # would quietly deliver less.
    for sample_rate in [16000, 22050, 44100]:
        for centre, _, _ in band_centres({}, sample_rate):
            assert centre <= sample_rate / 2


def test_band_centres_report_their_names():
    # The app uses these names to say which controls are inert, and the
    # plots use them to label only the bands that were actually applied.
    names = [name for _, _, name in band_centres({}, 44100)]
    assert names == sorted(BANDS, key=lambda n: BANDS[n][0])
    assert "treble" not in [name for _, _, name in band_centres({}, 8000)]


def test_bands_entirely_above_nyquist_are_dropped():
    # At 8 kHz the treble band starts exactly at Nyquist, so there are no
    # bins for it to act on and it is left out rather than half-applied.
    assert len(band_centres({}, 8000)) == 2
    assert len(band_centres({}, 16000)) == len(BANDS)
    assert len(band_centres({}, 44100)) == len(BANDS)


def test_each_band_reaches_its_full_gain_when_it_fits():
    for sample_rate in [16000, 22050, 44100]:
        for name in BANDS:
            curve = graphic_eq_curve(4096, sample_rate, {name: 12.0})
            top = max(20 * math.log10(g) for g in curve)
            assert top == pytest.approx(12.0, abs=0.5), \
                f"{name} at {sample_rate} Hz"


def test_apply_gain_multiplies_elementwise():
    spectrum = [1 + 1j, 2 + 0j, 0 + 3j]
    assert apply_gain(spectrum, [2.0, 0.5, 1.0]) == [2 + 2j, 1 + 0j, 0 + 3j]


def test_apply_gain_rejects_length_mismatch():
    with pytest.raises(ValueError):
        apply_gain([1 + 0j, 2 + 0j], [1.0])
