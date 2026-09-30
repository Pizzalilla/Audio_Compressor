"""
Tests for the block codec and container format.
"""

import io
import math
import os
import random
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_io import read_wav, write_wav
from bitio import BitReader, BitWriter
from codec import (HEADER, MAX_ORDER, CodecError, _BitCounter,
                   _read_code_table, _symbol_list_bits, _write_code_table,
                   analyse,
                   candidate_plans, compress, compress_with_report,
                   decompress, residuals, restore, unzigzag, zigzag)
from huffman import build_codes, build_tree


def tone(n, freq, rate=44100, amplitude=0.5, width=2):
    peak = (1 << (width * 8 - 1)) - 1
    return [round(amplitude * peak * math.sin(2 * math.pi * freq * i / rate))
            for i in range(n)]


def noise(n, width=2, seed=0):
    rng = random.Random(seed)
    half = 1 << (width * 8 - 1)
    return [rng.randrange(-half, half) for _ in range(n)]


def roundtrip(channels, width=2, rate=44100, block_size=4096,
              min_block_size=512):
    data = compress(channels, rate, width, block_size, min_block_size)
    assert decompress(data) == (channels, rate, width)
    return data


# --- transforms ------------------------------------------------------------

def test_zigzag():
    assert [zigzag(v) for v in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]
    for v in range(-1000, 1000):
        assert unzigzag(zigzag(v)) == v


@pytest.mark.parametrize("order", range(MAX_ORDER + 1))
def test_residuals_roundtrip(order):
    values = noise(500)
    assert restore(residuals(values, order), order) == values


def test_residuals_match_predictor_formulas():
    x = [3, 7, 20, 44, 91, 100]
    n = 5
    assert residuals(x, 1)[n] == x[n] - x[n - 1]
    assert residuals(x, 2)[n] == x[n] - 2 * x[n - 1] + x[n - 2]
    assert residuals(x, 3)[n] == x[n] - 3 * x[n - 1] + 3 * x[n - 2] - x[n - 3]


# --- code table ------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_code_table_roundtrip_and_size(seed):
    rng = random.Random(seed)
    symbols = [zigzag(int(rng.gauss(0, 10 ** rng.randint(0, 4))))
               for _ in range(5000)]
    codes = build_codes(build_tree(Counter(symbols)))
    writer = BitWriter()
    _write_code_table(writer, codes)
    counter = _BitCounter()
    _write_code_table(counter, codes)
    assert counter.bit_length == writer.bit_length
    # The pruning bound's table term must never exceed the real table.
    assert _symbol_list_bits(codes) + len(codes) <= counter.bit_length

    lengths = _read_code_table(BitReader(writer.getvalue()), len(symbols))
    assert lengths == {s: n for s, (n, _) in codes.items()}


def test_code_table_costs_under_four_bits_per_entry_for_audio():
    symbols = [zigzag(e) for e in residuals(tone(16384, 440), 1)]
    codes = build_codes(build_tree(Counter(symbols)))
    counter = _BitCounter()
    _write_code_table(counter, codes)
    assert counter.bit_length / len(codes) < 4


# --- round trips -----------------------------------------------------------

@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_roundtrip_every_width(width):
    roundtrip([tone(3000, 440, width=width)], width=width)
    roundtrip([noise(3000, width=width)], width=width)


def test_roundtrip_stereo_with_partial_last_block():
    roundtrip([tone(10000, 220), tone(10000, 330)], block_size=4096)


def test_roundtrip_extremes():
    # Full-scale square waves give the largest possible order-3 residuals.
    samples = ([-32768, 32767] * 600 + [32767, 32767, -32768, -32768] * 300
               + [0] * 100)
    roundtrip([samples])


def test_roundtrip_edge_sizes():
    roundtrip([[]])
    roundtrip([[5]])
    roundtrip([[1, 2, 3]], block_size=1, min_block_size=1)
    roundtrip([[0] * 5000])
    roundtrip([tone(777, 440)], block_size=4096, min_block_size=4096)


@pytest.mark.parametrize("block_size,min_block_size",
                         [(1, 1), (7, 3), (1000, 1000), (65536, 1024)])
def test_roundtrip_block_size_combinations(block_size, min_block_size):
    samples = tone(3000, 300) + noise(1000) + [9] * 500
    roundtrip([samples], block_size=block_size, min_block_size=min_block_size)


def test_wav_file_roundtrip(tmp_path):
    channels = [tone(5000, 440), tone(5000, 660)]
    path = tmp_path / "in.wav"
    write_wav(str(path), channels, 44100, 2)
    read_back, rate, width = read_wav(str(path))
    assert read_back == channels
    out = io.BytesIO()
    write_wav(out, *decompress(compress(read_back, rate, width)))
    out.seek(0)
    assert read_wav(out) == (channels, 44100, 2)


# --- mode and block choices ------------------------------------------------

def modes(blocks):
    return {b["mode"] for b in blocks}


def test_silence_compresses_to_almost_nothing():
    data = roundtrip([[0] * 441000], block_size=65536)
    assert len(data) < HEADER.size + 8 * 4


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_constant_blocks_store_one_sample(width):
    low = -(1 << (width * 8 - 1))
    samples = [low] * 1000 + [7] * 1000 + [-3] * 500
    blocks = analyse([samples], width, 500, 500)
    assert modes(blocks) == {"constant"}
    roundtrip([samples], width=width, block_size=500, min_block_size=500)


def test_tone_uses_a_predictor_and_compresses_well():
    samples = tone(44100, 440)
    blocks = analyse([samples], 2)
    assert all(b["mode"] in ("order2", "order3") for b in blocks)
    data = roundtrip([samples], block_size=65536, min_block_size=1024)
    assert len(data) < 0.5 * len(samples) * 2


def test_noise_is_stored_verbatim_without_growing():
    samples = noise(65536)
    assert modes(analyse([samples], 2)) == {"verbatim"}
    data = roundtrip([samples], block_size=65536)
    # One split flag and one mode for the whole superblock.
    assert len(data) <= HEADER.size + len(samples) * 2 + 1


def test_partition_adapts_to_the_signal():
    third = 32768
    samples = tone(third, 220) + [0] * third + noise(third)
    data, blocks = compress_with_report([samples], 44100, 2, 65536, 1024)
    assert decompress(data)[0] == [samples]

    by_mode = Counter()
    for b in blocks:
        by_mode[b["mode"]] += b["samples"]
    assert by_mode["constant"] >= third - 2048
    assert by_mode["verbatim"] >= third - 2048

    fixed = compress([samples], 44100, 2, 65536, 65536)
    assert len(data) < len(fixed)


@pytest.mark.parametrize("seed", range(8))
def test_pruning_never_changes_the_winner(seed):
    rng = random.Random(seed)
    kind = seed % 4
    n = rng.choice([16, 300, 2048])
    if kind == 0:
        samples = noise(n, seed=seed)
    elif kind == 1:
        samples = tone(n, rng.randint(20, 8000))
    elif kind == 2:
        samples = [int(rng.gauss(0, 2 ** rng.randint(1, 14))) for _ in range(n)]
    else:
        samples = [rng.choice([-1, 0, 0, 0, 1]) for _ in range(n)]
    pruned = min(p.bits for p in candidate_plans(samples, 2, prune=True))
    full = min(p.bits for p in candidate_plans(samples, 2, prune=False))
    assert pruned == full


def test_reported_bits_match_output_size():
    samples = tone(20000, 300) + noise(5000) + [4] * 3000
    data, blocks = compress_with_report([samples], 44100, 2, 8192, 512)
    body_bits = (len(data) - HEADER.size) * 8
    split_flags = body_bits - sum(b["bits"] for b in blocks)
    assert 0 <= split_flags < 8 + len(blocks) * 2


# --- bad input -------------------------------------------------------------

def test_bad_magic_rejected():
    data = bytearray(compress([tone(100, 440)], 44100, 2))
    data[0:4] = b"NOPE"
    with pytest.raises(CodecError):
        decompress(bytes(data))


def test_unknown_version_rejected():
    data = bytearray(compress([tone(100, 440)], 44100, 2))
    data[4] = 99
    with pytest.raises(CodecError, match="version"):
        decompress(bytes(data))


def test_truncated_data_rejected():
    data = compress([tone(5000, 440)], 44100, 2)
    for cut in (5, HEADER.size, len(data) // 2, len(data) - 1):
        with pytest.raises(CodecError):
            decompress(data[:cut])


def test_corrupted_data_is_always_detected():
    # Every kind of damage must raise CodecError: never IndexError, a hang,
    # or, thanks to the checksum, a silent decode into the wrong audio.
    original = compress([tone(3000, 440) + noise(500) + [0] * 500],
                        44100, 2, 1024, 256)
    rng = random.Random(0)
    for _ in range(300):
        data = bytearray(original)
        for _ in range(rng.randint(1, 4)):
            index = rng.randrange(HEADER.size, len(data))
            data[index] ^= 1 << rng.randrange(8)
        with pytest.raises(CodecError):
            decompress(bytes(data))


@pytest.mark.parametrize("zeroed", [2, 3, 10, 200])
def test_zeroed_tail_is_detected(zeroed):
    original = compress([tone(20000, 440)], 44100, 2)
    data = bytearray(original)
    data[-zeroed:] = bytes(zeroed)
    assert data != original
    with pytest.raises(CodecError):
        decompress(bytes(data))


def test_checksum_catches_damage_that_still_parses():
    # A flipped bit inside verbatim samples leaves the structure intact:
    # every read succeeds and every sample is in range. Only the checksum
    # can tell.
    data = bytearray(compress([noise(4096)], 44100, 2))
    data[HEADER.size + 1000] ^= 0x10
    with pytest.raises(CodecError, match="checksum"):
        decompress(bytes(data))


def test_damaged_checksum_is_detected():
    data = bytearray(compress([tone(1000, 440)], 44100, 2))
    data[HEADER.size - 1] ^= 1
    with pytest.raises(CodecError, match="checksum"):
        decompress(bytes(data))


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        compress([[40000]], 44100, 2)
    with pytest.raises(ValueError):
        compress([[1, 2], [1]], 44100, 2)
    with pytest.raises(ValueError):
        compress([[1]], 44100, 2, block_size=100, min_block_size=200)
    with pytest.raises(ValueError):
        compress([[1]], 44100, 2, block_size=0, min_block_size=0)
