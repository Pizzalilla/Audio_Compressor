"""
Bit-level reading and writing, most significant bit first.

Huffman codes don't line up with byte boundaries, so everything in the
compressed stream after the fixed header goes through these two classes.
"""


class BitWriter:
    """Accumulates bits and packs them into bytes."""

    def __init__(self):
        self._buffer = bytearray()
        self._acc = 0      # bits not yet flushed to _buffer
        self._count = 0    # how many bits are in _acc (always < 8 between calls)

    def write(self, value, width):
        """Write the low `width` bits of a non-negative integer."""
        if width == 0:
            return
        if value < 0 or value >> width:
            raise ValueError(f"{value} does not fit in {width} bits")
        self._acc = (self._acc << width) | value
        self._count += width
        while self._count >= 8:
            self._count -= 8
            self._buffer.append((self._acc >> self._count) & 0xFF)
        self._acc &= (1 << self._count) - 1

    def write_gamma(self, value):
        """Elias gamma code for value >= 1: (n-1) zeros, then value in n bits.

        Small numbers get short codes, and no width has to be agreed in
        advance, which suits the gaps between sorted symbols in a table.
        """
        if value < 1:
            raise ValueError("gamma codes only represent integers >= 1")
        n = value.bit_length()
        self.write(0, n - 1)
        self.write(value, n)

    @property
    def bit_length(self):
        return len(self._buffer) * 8 + self._count

    def getvalue(self):
        """The bytes written so far, with the last byte zero-padded."""
        if self._count == 0:
            return bytes(self._buffer)
        tail = self._acc << (8 - self._count)
        return bytes(self._buffer) + bytes([tail])


class BitReader:
    """Reads bits back out of a byte string written by BitWriter."""

    def __init__(self, data):
        self._data = data
        self._pos = 0        # next byte to pull into _acc
        self._acc = 0
        self._count = 0
        self._consumed = 0   # bits handed out so far
        self._limit = len(data) * 8

    def _fill(self, width):
        # Past the end of the data we shift in zeros so that peek() works
        # near the end of the stream. Actually consuming those bits is an
        # error, which skip() checks.
        while self._count < width:
            byte = self._data[self._pos] if self._pos < len(self._data) else 0
            self._pos += 1
            self._acc = (self._acc << 8) | byte
            self._count += 8

    def peek(self, width):
        """The next `width` bits as an integer, without consuming them."""
        self._fill(width)
        return (self._acc >> (self._count - width)) & ((1 << width) - 1)

    def skip(self, width):
        self._consumed += width
        if self._consumed > self._limit:
            raise EOFError("read past the end of the compressed data")
        self._fill(width)
        self._count -= width
        self._acc &= (1 << self._count) - 1

    def read(self, width):
        if width == 0:
            return 0
        value = self.peek(width)
        self.skip(width)
        return value

    def read_bit(self):
        return self.read(1)

    def read_gamma(self):
        zeros = 0
        while self.read_bit() == 0:
            zeros += 1
            if zeros > 64:
                raise ValueError("malformed gamma code")
        # The leading 1 has been consumed; the remaining bits follow it.
        return (1 << zeros) | self.read(zeros)
