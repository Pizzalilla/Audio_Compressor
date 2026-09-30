"""
Tests for the Huffman tree, canonical codes, and decoder.
"""

import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitio import BitReader, BitWriter
from huffman import (Decoder, MinHeap, build_codes, build_tree,
                     canonical_codes, code_lengths, decode, encode,
                     limited_code_lengths)

EXAMPLE = {"A": 5, "B": 2, "C": 1, "D": 1}


def roundtrip(symbols, codes):
    writer = encode(symbols, codes, BitWriter())
    lengths = {s: length for s, (length, _) in codes.items()}
    return decode(BitReader(writer.getvalue()), Decoder(lengths), len(symbols))


def cost(frequencies, lengths):
    return sum(frequencies[s] * lengths[s] for s in frequencies)


def is_prefix_free(codes):
    strings = [format(v, f"0{n}b") for n, v in codes.values()]
    return not any(a != b and b.startswith(a)
                   for a, b in itertools.permutations(strings, 2))


# --- MinHeap ---------------------------------------------------------------

def test_heap_pops_in_key_order():
    rng = random.Random(1)
    keys = [rng.randint(0, 100) for _ in range(500)]
    heap = MinHeap()
    for k in keys:
        heap.push(k, str(k))
    popped = [heap.pop()[0] for _ in range(len(keys))]
    assert popped == sorted(keys)
    assert len(heap) == 0


def test_heap_pop_empty_raises():
    with pytest.raises(IndexError):
        MinHeap().pop()


# --- tree and codes -------------------------------------------------------

def test_worked_example():
    root = build_tree(EXAMPLE)
    assert root.weight == 9
    lengths = code_lengths(root)
    assert lengths == {"A": 1, "B": 2, "C": 3, "D": 3}
    assert cost(EXAMPLE, lengths) == 15

    codes = build_codes(root)
    assert codes == {"A": (1, 0b0), "B": (2, 0b10),
                     "C": (3, 0b110), "D": (3, 0b111)}


def test_single_symbol_gets_one_bit():
    codes = build_codes(build_tree({7: 100}))
    assert codes == {7: (1, 0)}
    assert roundtrip([7] * 100, codes) == [7] * 100


def test_empty_frequencies_rejected():
    with pytest.raises(ValueError):
        build_tree({})
    with pytest.raises(ValueError):
        build_tree({1: 0})


def brute_force_optimum(frequencies):
    # Try every length assignment satisfying Kraft; the cheapest is optimal.
    symbols = list(frequencies)
    n = len(symbols)
    best = None
    for lengths in itertools.product(range(1, n), repeat=n):
        if sum(2.0 ** -l for l in lengths) <= 1:
            c = sum(frequencies[s] * l for s, l in zip(symbols, lengths))
            best = c if best is None else min(best, c)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force_optimum(seed):
    rng = random.Random(seed)
    frequencies = {s: rng.randint(1, 50) for s in range(rng.randint(2, 6))}
    lengths = code_lengths(build_tree(frequencies))
    assert cost(frequencies, lengths) == brute_force_optimum(frequencies)


@pytest.mark.parametrize("seed", range(10))
def test_codes_are_prefix_free_and_roundtrip(seed):
    rng = random.Random(seed)
    symbols = [int(rng.gauss(0, 30)) for _ in range(3000)]
    frequencies = {s: symbols.count(s) for s in set(symbols)}
    codes = build_codes(build_tree(frequencies))
    assert is_prefix_free(codes)
    assert roundtrip(symbols, codes) == symbols


def test_canonical_codes_depend_only_on_lengths():
    # Ties in the example (B(2) vs the merged C+D node) can be broken either
    # way; canonical codes come out the same regardless of dict order.
    lengths = {"A": 1, "B": 2, "C": 3, "D": 3}
    shuffled = dict(reversed(list(lengths.items())))
    assert canonical_codes(lengths) == canonical_codes(shuffled)


def test_long_codes_use_slow_decoder_path():
    # Fibonacci weights give the most lopsided tree possible: depth n-1.
    fib = [1, 1]
    while len(fib) < 18:
        fib.append(fib[-1] + fib[-2])
    frequencies = {i: f for i, f in enumerate(fib)}
    codes = build_codes(build_tree(frequencies))
    assert max(n for n, _ in codes.values()) > Decoder.LOOKUP_BITS
    symbols = list(range(18)) * 3
    assert roundtrip(symbols, codes) == symbols


def test_length_limit_is_enforced():
    fib = [1, 1]
    while len(fib) < 30:
        fib.append(fib[-1] + fib[-2])
    frequencies = {i: f for i, f in enumerate(fib)}
    root = build_tree(frequencies)
    assert max(code_lengths(root).values()) == 29

    lengths = limited_code_lengths(root, max_length=12)
    assert max(lengths.values()) <= 12
    assert sum(2 ** -n for n in lengths.values()) <= 1

    codes = canonical_codes(lengths)
    symbols = list(range(30))
    assert roundtrip(symbols, codes) == symbols


def test_decoder_rejects_oversubscribed_lengths():
    with pytest.raises(ValueError):
        Decoder({"a": 1, "b": 1, "c": 1})


def test_encode_unknown_symbol_raises():
    with pytest.raises(ValueError):
        encode(["Z"], build_codes(build_tree(EXAMPLE)), BitWriter())
