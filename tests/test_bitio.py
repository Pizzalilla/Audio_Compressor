"""
Tests for BitWriter and BitReader.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitio import BitReader, BitWriter


def test_msb_first_packing():
    w = BitWriter()
    w.write(0b1, 1)
    w.write(0b01, 2)
    w.write(0b10110, 5)
    w.write(0b1, 1)
    assert w.bit_length == 9
    assert w.getvalue() == bytes([0b10110110, 0b10000000])


def test_random_roundtrip():
    rng = random.Random(0)
    fields = [(rng.getrandbits(n), n) for n in
              (rng.randint(0, 33) for _ in range(2000))]
    w = BitWriter()
    for value, width in fields:
        w.write(value, width)
    r = BitReader(w.getvalue())
    assert [r.read(width) for _, width in fields] == [v for v, _ in fields]


@pytest.mark.parametrize("value", [1, 2, 3, 4, 7, 8, 255, 256, 10 ** 6])
def test_gamma_roundtrip(value):
    w = BitWriter()
    w.write_gamma(value)
    w.write(0b101, 3)  # something after, to check nothing is over-read
    assert w.bit_length == 2 * value.bit_length() - 1 + 3
    r = BitReader(w.getvalue())
    assert r.read_gamma() == value
    assert r.read(3) == 0b101


def test_gamma_rejects_zero():
    with pytest.raises(ValueError):
        BitWriter().write_gamma(0)


def test_value_too_wide_rejected():
    with pytest.raises(ValueError):
        BitWriter().write(4, 2)
    with pytest.raises(ValueError):
        BitWriter().write(-1, 8)


def test_peek_does_not_consume():
    r = BitReader(bytes([0b11010000]))
    assert r.peek(3) == 0b110
    assert r.peek(3) == 0b110
    assert r.read(4) == 0b1101


def test_peek_past_end_pads_but_read_past_end_raises():
    r = BitReader(bytes([0xFF]))
    assert r.peek(12) == 0xFF0
    assert r.read(8) == 0xFF
    with pytest.raises(EOFError):
        r.read(1)
