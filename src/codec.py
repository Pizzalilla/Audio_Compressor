"""
Lossless audio compression: prediction, adaptive blocks, and the container
format.

Block encodings
---------------
Each block is stored in whichever of these is smallest, recorded with a
3-bit mode:

  VERBATIM  plain fixed-width samples. The fallback that guarantees a block
            never grows by more than a few bits of framing.
  CONSTANT  every sample is the same (usually silence): store it once.
            Huffman can't help here, because no code is shorter than 1 bit.
  FIXED k   (k = 0..3) Huffman codes for the residual of a fixed polynomial
            predictor of order k, the same predictors FLAC uses:
                k=0  e = x[n]                                (raw samples)
                k=1  e = x[n] - x[n-1]                       (delta)
                k=2  e = x[n] - 2x[n-1] + x[n-2]
                k=3  e = x[n] - 3x[n-1] + 3x[n-2] - x[n-3]
            Order k is just "take differences k times". Higher orders track
            smooth signals more closely and shrink their residuals; on noise
            every extra difference widens the distribution instead, which is
            why the choice is made per block from exact sizes.

Adaptive block sizes
--------------------
A code table is a fixed cost per block, so big blocks amortise it better.
But a block only gets one mode, so small blocks follow changes in the
signal (tone -> silence -> noise) more closely. Rather than choosing one
size, each superblock of `block_size` samples is split in half recursively,
down to `min_block_size`, keeping a split only where it is smaller in total.

Code tables
-----------
A Huffman table is {symbol: code length}. Symbols are zigzag-mapped
(0, -1, 1, -2, ... -> 0, 1, 2, 3, ...), sorted, and stored as gamma-coded
gaps, which are mostly 1 (1 bit each) because residuals cluster around zero.
Neighbouring symbols have similar code lengths, so the lengths are stored as
changes from the previous one, themselves Huffman coded with a tiny second
table. That costs one or two bits per entry instead of a fixed five.

Container layout
----------------
Header (bytes, little-endian):
    magic "AHUF" | version u8 | channels u16 | sample_width u8 |
    sample_rate u32 | frame_count u32 | block_size u32 | min_block_size u32 |
    md5 (16 bytes)

The MD5 covers the decoded samples, channel by channel, each as a
little-endian signed 64-bit integer. Structural damage mostly shows up as an
impossible code or a read past the end, but not always: a zeroed tail, say,
can decode "successfully" into wrong audio. The checksum catches that. FLAC
stores an MD5 in its header for the same reason.

Then one bitstream. For each superblock position, for each channel, a block
tree in preorder. A block of n samples with n >= 2 * min_block_size starts
with a split bit; if it is 1, the block is two blocks of n//2 and n - n//2.
A leaf block is:
    mode (3 bits)
    VERBATIM: n samples, each sample_width*8 bits, two's complement
    CONSTANT: one sample, sample_width*8 bits, two's complement
    FIXED k:  code table, then n Huffman codes

Code table:
    symbol count (gamma)
    symbol gaps (gamma each), ascending, starting from -1
    length-change table: count (gamma), then per entry
        gap (gamma), code length (4 bits)
    one length change per symbol, Huffman coded with that table

Prediction restarts at each block (samples before it are taken as zero) so
every block can be decoded on its own.
"""

import hashlib
import math
import struct
import sys
from array import array
from collections import Counter

from bitio import BitReader, BitWriter
from huffman import (MAX_CODE_LENGTH, Decoder, build_codes, build_tree,
                     decode, encode)

MAGIC = b"AHUF"
VERSION = 3
HEADER = struct.Struct("<4sBHBIIII16s")

DEFAULT_BLOCK_SIZE = 65536
DEFAULT_MIN_BLOCK_SIZE = 4096
MAX_BLOCK_SIZE = 1 << 20

MAX_ORDER = 3

MODE_BITS = 3
MODE_VERBATIM = 0
MODE_CONSTANT = 1
MODE_FIXED = 2          # modes 2..2+MAX_ORDER are FIXED order 0..MAX_ORDER

META_MAX_LENGTH = 15    # code lengths in the length-change table
META_LENGTH_BITS = 4
MAX_LENGTH_CHANGE_SYMBOL = 2 * MAX_CODE_LENGTH   # zigzag(-MAX_CODE_LENGTH)
assert META_MAX_LENGTH < 1 << META_LENGTH_BITS


class CodecError(ValueError):
    """Compressed data that is malformed or not in this format."""


def mode_name(mode):
    if mode == MODE_VERBATIM:
        return "verbatim"
    if mode == MODE_CONSTANT:
        return "constant"
    return f"order{mode - MODE_FIXED}"


# --- transforms ------------------------------------------------------------

def zigzag(value):
    return value * 2 if value >= 0 else -value * 2 - 1


def unzigzag(value):
    return value >> 1 if value % 2 == 0 else -((value + 1) >> 1)


def delta(samples):
    previous = 0
    out = []
    for sample in samples:
        out.append(sample - previous)
        previous = sample
    return out


def undelta(differences):
    total = 0
    out = []
    for difference in differences:
        total += difference
        out.append(total)
    return out


def residuals(samples, order):
    """Residual of the order-k fixed predictor: k rounds of delta()."""
    for _ in range(order):
        samples = delta(samples)
    return samples


def restore(values, order):
    """Inverse of residuals()."""
    for _ in range(order):
        values = undelta(values)
    return values


def entropy(values):
    """Shannon entropy in bits per value: the floor for any symbol code."""
    if not values:
        return 0.0
    return _entropy(Counter(values), len(values))


def _entropy(frequencies, n):
    return -sum(c / n * math.log2(c / n) for c in frequencies.values())


def checksum(channels):
    """MD5 of the samples, as described in the module docstring."""
    digest = hashlib.md5()
    for channel in channels:
        values = array("q", channel)
        if sys.byteorder == "big":
            values.byteswap()
        digest.update(values.tobytes())
    return digest.digest()


# --- code tables -----------------------------------------------------------

class _BitCounter:
    """Stands in for a BitWriter when only the size is wanted.

    Sizing a table by "writing" it through this, rather than with a separate
    formula, means the size used to choose a mode can't drift from what is
    actually written.
    """

    def __init__(self):
        self.bit_length = 0

    def write(self, value, width):
        self.bit_length += width

    def write_gamma(self, value):
        self.bit_length += 2 * value.bit_length() - 1


def _symbol_list_bits(symbols):
    """Exact size of the count and gap part of a code table."""
    bits = 2 * len(symbols).bit_length() - 1
    previous = -1
    for symbol in sorted(symbols):
        bits += 2 * (symbol - previous).bit_length() - 1
        previous = symbol
    return bits


def _write_code_table(writer, codes):
    symbols = sorted(codes)
    writer.write_gamma(len(symbols))
    previous = -1
    for symbol in symbols:
        writer.write_gamma(symbol - previous)
        previous = symbol

    changes = []
    previous = 0
    for symbol in symbols:
        length = codes[symbol][0]
        changes.append(zigzag(length - previous))
        previous = length

    meta = build_codes(build_tree(Counter(changes)), META_MAX_LENGTH)
    writer.write_gamma(len(meta))
    previous = -1
    for change in sorted(meta):
        writer.write_gamma(change - previous)
        writer.write(meta[change][0], META_LENGTH_BITS)
        previous = change
    encode(changes, meta, writer)


def _read_code_table(reader, max_symbols):
    count = reader.read_gamma()
    if count > max_symbols:
        raise CodecError(f"code table claims {count} symbols for a block "
                         f"of {max_symbols} samples")
    symbols = []
    symbol = -1
    for _ in range(count):
        symbol += reader.read_gamma()
        symbols.append(symbol)

    meta_count = reader.read_gamma()
    if meta_count > MAX_LENGTH_CHANGE_SYMBOL + 1:
        raise CodecError("length-change table is too large")
    meta_lengths = {}
    change = -1
    for _ in range(meta_count):
        change += reader.read_gamma()
        length = reader.read(META_LENGTH_BITS)
        if change > MAX_LENGTH_CHANGE_SYMBOL or not 1 <= length <= META_MAX_LENGTH:
            raise CodecError("invalid length-change table")
        meta_lengths[change] = length

    changes = decode(reader, Decoder(meta_lengths), count)
    lengths = {}
    length = 0
    for symbol, change in zip(symbols, changes):
        length += unzigzag(change)
        if not 1 <= length <= MAX_CODE_LENGTH:
            raise CodecError(f"invalid code length {length}")
        lengths[symbol] = length
    return lengths


# --- per-block plans -------------------------------------------------------

class _VerbatimPlan:
    mode = MODE_VERBATIM
    table_bits = 0

    def __init__(self, samples, sample_width):
        self.samples = samples
        self.width = sample_width * 8
        self.bits = MODE_BITS + self.width * len(samples)

    def write(self, writer):
        writer.write(self.mode, MODE_BITS)
        mask = (1 << self.width) - 1
        for sample in self.samples:
            writer.write(sample & mask, self.width)


class _ConstantPlan:
    mode = MODE_CONSTANT
    table_bits = 0

    def __init__(self, value, sample_width):
        self.value = value
        self.width = sample_width * 8
        self.bits = MODE_BITS + self.width

    def write(self, writer):
        writer.write(self.mode, MODE_BITS)
        writer.write(self.value & ((1 << self.width) - 1), self.width)


class _HuffmanPlan:
    """A block's residual symbols with their codes and exact encoded size."""

    def __init__(self, order, symbols, frequencies):
        self.mode = MODE_FIXED + order
        self.symbols = symbols
        self.codes = build_codes(build_tree(frequencies))
        counter = _BitCounter()
        _write_code_table(counter, self.codes)
        self.table_bits = counter.bit_length
        payload = sum(c * self.codes[s][0] for s, c in frequencies.items())
        self.bits = MODE_BITS + self.table_bits + payload

    def write(self, writer):
        writer.write(self.mode, MODE_BITS)
        _write_code_table(writer, self.codes)
        encode(self.symbols, self.codes, writer)


def candidate_plans(samples, sample_width, prune=True):
    """Encodings of one block, as plans with an exact `bits` size.

    Building a Huffman plan is the expensive part, so with prune=True a
    predictor order is skipped when a lower bound on its size already loses
    to the best plan so far. The bound is safe: the payload can't beat the
    Shannon entropy, the symbol gaps in the table depend only on which
    symbols occur (so their cost is exact), and every length change costs at
    least 1 bit. So pruning never changes which plan wins; it only avoids
    building ones that can't.

    The gap term matters. In a short block of noise almost every sample is
    distinct, so the entropy is capped near log2(block length) and looks
    compressible; it's the table, with a large gap per entry, that isn't.
    """
    if samples and min(samples) == max(samples):
        # Nothing else can beat this.
        return [_ConstantPlan(samples[0], sample_width)]

    plans = [_VerbatimPlan(samples, sample_width)]
    best = plans[0].bits
    n = len(samples)

    candidates = []
    for order in range(MAX_ORDER + 1):
        symbols = [zigzag(e) for e in residuals(samples, order)]
        frequencies = Counter(symbols)
        bound = (MODE_BITS + n * _entropy(frequencies, n)
                 + _symbol_list_bits(frequencies) + len(frequencies))
        candidates.append((bound, order, symbols, frequencies))

    # Most promising first, so the bound prunes as much as possible.
    for bound, order, symbols, frequencies in sorted(candidates,
                                                     key=lambda c: c[:2]):
        # The 1-bit margin absorbs floating-point error in the entropy.
        if prune and bound - 1 >= best:
            continue
        plan = _HuffmanPlan(order, symbols, frequencies)
        plans.append(plan)
        best = min(best, plan.bits)
    return plans


def plan_block(samples, sample_width):
    """The smallest encoding for one block.

    On a tie, the earlier candidate wins: verbatim before any Huffman plan,
    since it is cheaper to decode.
    """
    return min(candidate_plans(samples, sample_width), key=lambda p: p.bits)


# --- block partitioning ----------------------------------------------------

def _can_split(n, min_block_size):
    return n >= 2 * min_block_size


def _best_partition(samples, sample_width, min_block_size):
    """(bits, tree) for the cheapest way to split `samples` into blocks.

    A tree is either a plan (a leaf block) or a (left, right) pair. `bits`
    includes the split flag when one is written.
    """
    leaf = plan_block(samples, sample_width)
    n = len(samples)
    if not _can_split(n, min_block_size):
        return leaf.bits, leaf
    if leaf.mode == MODE_CONSTANT:
        # Two blocks would need at least two of everything this one has.
        return 1 + leaf.bits, leaf

    half = n // 2
    left_bits, left = _best_partition(samples[:half], sample_width,
                                      min_block_size)
    right_bits, right = _best_partition(samples[half:], sample_width,
                                        min_block_size)
    if left_bits + right_bits < leaf.bits:
        return 1 + left_bits + right_bits, (left, right)
    return 1 + leaf.bits, leaf


def _write_tree(writer, tree, n, min_block_size):
    is_split = isinstance(tree, tuple)
    if _can_split(n, min_block_size):
        writer.write(1 if is_split else 0, 1)
    if is_split:
        half = n // 2
        _write_tree(writer, tree[0], half, min_block_size)
        _write_tree(writer, tree[1], n - half, min_block_size)
    else:
        tree.write(writer)


def _leaves(tree, start, n):
    """(start, length, plan) for each leaf block, in order."""
    if isinstance(tree, tuple):
        half = n // 2
        yield from _leaves(tree[0], start, half)
        yield from _leaves(tree[1], start + half, n - half)
    else:
        yield start, n, tree


def _superblocks(frame_count, block_size):
    for start in range(0, frame_count, block_size):
        yield start, min(start + block_size, frame_count)


def _validate(channels, sample_width, block_size, min_block_size):
    if not channels:
        raise ValueError("no channels to compress")
    if len(channels) > 0xFFFF:
        raise ValueError("too many channels")
    if len({len(c) for c in channels}) != 1:
        raise ValueError("channels must all be the same length")
    if len(channels[0]) > 0xFFFFFFFF:
        raise ValueError("too many samples for this format")
    if not 1 <= sample_width <= 4:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")
    if not 1 <= min_block_size <= block_size <= MAX_BLOCK_SIZE:
        raise ValueError(
            f"need 1 <= min block size <= block size <= {MAX_BLOCK_SIZE}, "
            f"got {min_block_size} and {block_size}")

    low = -(1 << (sample_width * 8 - 1))
    high = -low - 1
    for channel in channels:
        if channel and not low <= min(channel) <= max(channel) <= high:
            raise ValueError(f"samples do not fit in {sample_width} bytes")


def _plan(channels, sample_width, block_size, min_block_size):
    """trees[superblock][channel] for the whole signal."""
    frame_count = len(channels[0])
    return [
        [_best_partition(channel[start:end], sample_width, min_block_size)[1]
         for channel in channels]
        for start, end in _superblocks(frame_count, block_size)
    ]


def _describe(trees, frame_count, block_size):
    blocks = []
    for (start, end), row in zip(_superblocks(frame_count, block_size), trees):
        for index, tree in enumerate(row):
            for leaf_start, n, plan in _leaves(tree, start, end - start):
                blocks.append({
                    "channel": index,
                    "start": leaf_start,
                    "samples": n,
                    "mode": mode_name(plan.mode),
                    "bits": plan.bits,
                    "table_bits": plan.table_bits,
                })
    return blocks


def compress_with_report(channels, sample_rate, sample_width,
                         block_size=DEFAULT_BLOCK_SIZE,
                         min_block_size=DEFAULT_MIN_BLOCK_SIZE):
    """Like compress(), but also returns a description of every block."""
    _validate(channels, sample_width, block_size, min_block_size)
    frame_count = len(channels[0])
    trees = _plan(channels, sample_width, block_size, min_block_size)

    writer = BitWriter()
    for (start, end), row in zip(_superblocks(frame_count, block_size), trees):
        for tree in row:
            _write_tree(writer, tree, end - start, min_block_size)

    header = HEADER.pack(MAGIC, VERSION, len(channels), sample_width,
                         sample_rate, frame_count, block_size, min_block_size,
                         checksum(channels))
    return header + writer.getvalue(), _describe(trees, frame_count, block_size)


def compress(channels, sample_rate, sample_width,
             block_size=DEFAULT_BLOCK_SIZE,
             min_block_size=DEFAULT_MIN_BLOCK_SIZE):
    """Compress channels of signed integer samples to bytes."""
    return compress_with_report(channels, sample_rate, sample_width,
                                block_size, min_block_size)[0]


def analyse(channels, sample_width, block_size=DEFAULT_BLOCK_SIZE,
            min_block_size=DEFAULT_MIN_BLOCK_SIZE):
    """The blocks compress() would produce, with residual entropy per order.

    Each dict has: channel, start, samples, mode, bits, table_bits, and
    entropy (a list indexed by predictor order).
    """
    _validate(channels, sample_width, block_size, min_block_size)
    trees = _plan(channels, sample_width, block_size, min_block_size)
    blocks = _describe(trees, len(channels[0]), block_size)
    for block in blocks:
        samples = channels[block["channel"]][
            block["start"]:block["start"] + block["samples"]]
        block["entropy"] = [entropy(residuals(samples, k))
                            for k in range(MAX_ORDER + 1)]
    return blocks


# --- decoding --------------------------------------------------------------

def _read_block(reader, n, width, low, high):
    sign_bit = 1 << (width - 1)
    mode = reader.read(MODE_BITS)
    if mode == MODE_CONSTANT:
        value = reader.read(width)
        return [(value ^ sign_bit) - sign_bit] * n
    if mode == MODE_VERBATIM:
        return [(reader.read(width) ^ sign_bit) - sign_bit for _ in range(n)]
    if MODE_FIXED <= mode <= MODE_FIXED + MAX_ORDER:
        decoder = Decoder(_read_code_table(reader, n))
        values = [unzigzag(s) for s in decode(reader, decoder, n)]
        samples = restore(values, mode - MODE_FIXED)
        if samples and not low <= min(samples) <= max(samples) <= high:
            raise CodecError("decoded samples are out of range")
        return samples
    raise CodecError(f"unknown block mode {mode}")


def _read_tree(reader, n, min_block_size, width, low, high, out):
    if _can_split(n, min_block_size) and reader.read_bit():
        half = n // 2
        _read_tree(reader, half, min_block_size, width, low, high, out)
        _read_tree(reader, n - half, min_block_size, width, low, high, out)
    else:
        out.extend(_read_block(reader, n, width, low, high))


def decompress(data):
    """Inverse of compress(). Returns (channels, sample_rate, sample_width)."""
    if len(data) < HEADER.size:
        raise CodecError("data is too short to contain a header")
    magic, version = data[:4], data[4]
    if magic != MAGIC:
        raise CodecError("not a compressed audio file (bad magic number)")
    if version != VERSION:
        raise CodecError(f"unsupported format version {version}")
    (_, _, channel_count, sample_width, sample_rate, frame_count,
     block_size, min_block_size, expected) = HEADER.unpack_from(data)
    if (channel_count < 1 or not 1 <= sample_width <= 4
            or not 1 <= min_block_size <= block_size <= MAX_BLOCK_SIZE):
        raise CodecError("header contains invalid values")

    width = sample_width * 8
    low = -(1 << (width - 1))
    high = -low - 1
    channels = [[] for _ in range(channel_count)]
    reader = BitReader(data[HEADER.size:])

    try:
        for start, end in _superblocks(frame_count, block_size):
            for channel in channels:
                _read_tree(reader, end - start, min_block_size,
                           width, low, high, channel)
    except EOFError as error:
        raise CodecError(f"compressed data is truncated: {error}") from error
    except CodecError:
        raise
    except ValueError as error:
        raise CodecError(f"compressed data is corrupt: {error}") from error

    if checksum(channels) != expected:
        raise CodecError("checksum mismatch: the compressed data is corrupt")
    return channels, sample_rate, sample_width
