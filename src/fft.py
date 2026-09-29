"""
Core Fourier transform implementations.

This module is deliberately dependency-free: it knows nothing about audio,
files, or plotting. It operates on plain sequences of numbers.
"""

import cmath


def dft_naive(x):
    """Direct O(n^2) evaluation of the DFT. Correctness reference for fft().

    Args:
        x: sequence of real or complex samples, any length.

    Returns:
        list of complex DFT coefficients, same length as x.
    """
    n = len(x)
    coefficients = []
    for k in range(n):
        total = 0j
        for j in range(n):
            total += x[j] * cmath.exp(-2j * cmath.pi * k * j / n)
        coefficients.append(total)
    return coefficients


def fft(x):
    """Recursive radix-2 Cooley-Tukey FFT.

    Args:
        x: sequence of real or complex samples. Length must be a power of 2.

    Returns:
        list of complex DFT coefficients, same length as x.

    Raises:
        ValueError: if len(x) is not a power of 2.
    """
    n = len(x)
    if n == 0:
        return []
    if not is_power_of_two(n):
        raise ValueError(f"length must be a power of 2, got {n}")

    # A single sample is its own transform: the sum has one term, e^0 = 1.
    if n == 1:
        return [complex(x[0])]

    even = fft(x[0::2])
    odd = fft(x[1::2])

    half = n // 2
    out = [0j] * n
    for k in range(half):
        # Computing this product once is the whole trick: it feeds both
        # output bins, because w^(k + n/2) == -w^k.
        twiddled = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        out[k] = even[k] + twiddled
        out[k + half] = even[k] - twiddled
    return out


def ifft(X):
    """Inverse FFT.

    Args:
        X: sequence of complex DFT coefficients. Length must be a power of 2.

    Returns:
        list of complex samples, same length as X.
    """
    n = len(X)
    if n == 0:
        return []

    # The inverse matrix is the conjugate transpose over n, so conjugating
    # on the way in and out turns the forward transform into the inverse.
    conjugated = [complex(z).conjugate() for z in X]
    transformed = fft(conjugated)
    return [z.conjugate() / n for z in transformed]


def is_power_of_two(n):
    # True if n is a positive power of 2.
    return n > 0 and n & (n - 1) == 0
