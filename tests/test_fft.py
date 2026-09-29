"""
Correctness tests for the FFT core.

Run with:  python -m pytest tests/ -v
"""

import cmath
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fft import dft_naive, fft, ifft, is_power_of_two


def random_signal(n, seed=0):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def assert_close(a, b, tolerance=1e-9):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tolerance, f"index {i}: {x} vs {y}"


def test_matches_naive_dft():
    # fft() should agree with the O(n^2) definition.
    for n in [1, 2, 4, 8, 16, 32, 64]:
        signal = random_signal(n, seed=n)
        assert_close(fft(signal), dft_naive(signal), tolerance=1e-9)


def test_roundtrip():
    # ifft(fft(x)) should recover x.
    for n in [1, 2, 4, 16, 256]:
        signal = random_signal(n, seed=n + 1)
        recovered = ifft(fft(signal))
        assert_close([complex(v) for v in signal], recovered, tolerance=1e-9)


def test_known_values():
    # Hand-computed case: [1, 2, 3, 4] -> [10, -2+2j, -2, -2-2j]
    assert_close(fft([1, 2, 3, 4]), [10 + 0j, -2 + 2j, -2 + 0j, -2 - 2j])

    # An impulse contains every frequency in equal measure.
    assert_close(fft([1, 0, 0, 0]), [1 + 0j] * 4)

    # A constant signal is pure DC.
    assert_close(fft([1, 1, 1, 1]), [4 + 0j, 0j, 0j, 0j])

    # Alternating +1/-1 is exactly the Nyquist frequency.
    assert_close(fft([1, -1, 1, -1]), [0j, 0j, 4 + 0j, 0j])


def test_single_bin_for_exact_frequency():
    # A sinusoid completing a whole number of cycles in the window lands in
    # one bin with nothing leaking into the others.
    n, cycles = 64, 7
    signal = [math.cos(2 * math.pi * cycles * i / n) for i in range(n)]
    spectrum = fft(signal)
    for k, value in enumerate(spectrum):
        expected = n / 2 if k in (cycles, n - cycles) else 0.0
        assert abs(abs(value) - expected) < 1e-9


def test_parseval():
    # sum(|x[n]|^2) == sum(|X[k]|^2) / N
    for n in [8, 64, 512]:
        signal = random_signal(n, seed=n + 2)
        spectrum = fft(signal)
        time_energy = sum(abs(v) ** 2 for v in signal)
        frequency_energy = sum(abs(v) ** 2 for v in spectrum) / n
        assert abs(time_energy - frequency_energy) < 1e-9


def test_linearity():
    n = 32
    a, b = random_signal(n, seed=3), random_signal(n, seed=4)
    combined = [2.0 * x + 3.0 * y for x, y in zip(a, b)]
    expected = [2.0 * x + 3.0 * y for x, y in zip(fft(a), fft(b))]
    assert_close(fft(combined), expected)


def test_real_input_is_conjugate_symmetric():
    # This symmetry is what lets the equaliser mirror its gain curve and
    # still get a real signal back out.
    n = 64
    spectrum = fft(random_signal(n, seed=5))
    for k in range(1, n // 2):
        assert abs(spectrum[k] - spectrum[n - k].conjugate()) < 1e-9


def test_rejects_non_power_of_two():
    # fft() should raise ValueError on lengths that are not powers of 2.
    for n in [3, 5, 6, 7, 9, 100]:
        with pytest.raises(ValueError):
            fft([0.0] * n)


def test_empty_input():
    assert fft([]) == []
    assert ifft([]) == []


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(1024)
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(-4)


def test_matches_numpy():
    # External check against a reference implementation.
    numpy = pytest.importorskip("numpy")
    for n in [4, 32, 256, 1024]:
        signal = random_signal(n, seed=n + 6)
        assert_close(fft(signal), numpy.fft.fft(signal).tolist(), tolerance=1e-9)
        assert_close(ifft(signal), numpy.fft.ifft(signal).tolist(), tolerance=1e-9)