"""
Huffman coding, with canonical code assignment.

The tree is built greedily: repeatedly merge the two lightest nodes until
one remains. Only the depth of each leaf is kept from it. Codes are then
assigned canonically from those depths alone, so:

  * the decoder never needs the tree, only {symbol: length}, which is far
    cheaper to store;
  * how ties were broken while building the tree stops mattering. Different
    tie-breaks can give different depths, but whichever depths the encoder
    writes down, the decoder derives exactly the same codes from them.
"""

MAX_CODE_LENGTH = 20


class Node:
    __slots__ = ("weight", "symbol", "left", "right")

    def __init__(self, weight, symbol=None, left=None, right=None):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None


class MinHeap:
    """Binary min-heap of (key, item) pairs, ordered by key only.

    Stored as a list where the children of index i are 2i+1 and 2i+2.
    Push and pop are O(log n): an item moves along one root-to-leaf path.
    """

    def __init__(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def push(self, key, item):
        items = self._items
        items.append((key, item))
        i = len(items) - 1
        # Sift up: swap with the parent while smaller than it.
        while i > 0:
            parent = (i - 1) // 2
            if items[parent][0] <= items[i][0]:
                break
            items[parent], items[i] = items[i], items[parent]
            i = parent

    def pop(self):
        """Remove and return the (key, item) pair with the smallest key."""
        items = self._items
        if not items:
            raise IndexError("pop from an empty heap")
        top = items[0]
        last = items.pop()
        if items:
            items[0] = last
            # Sift down: swap with the smaller child while larger than it.
            i, n = 0, len(items)
            while True:
                smallest = i
                for child in (2 * i + 1, 2 * i + 2):
                    if child < n and items[child][0] < items[smallest][0]:
                        smallest = child
                if smallest == i:
                    break
                items[i], items[smallest] = items[smallest], items[i]
                i = smallest
        return top


def build_tree(frequencies):
    """Build a Huffman tree from {symbol: count}. Returns the root Node.

    Keys are (weight, sequence number). The sequence number makes ties pop
    in insertion order, so the tree is deterministic, and means the heap
    never has to compare two Nodes.
    """
    counts = {s: c for s, c in frequencies.items() if c > 0}
    if not counts:
        raise ValueError("cannot build a Huffman tree with no symbols")

    heap = MinHeap()
    sequence = 0
    for symbol in sorted(counts):
        heap.push((counts[symbol], sequence), Node(counts[symbol], symbol))
        sequence += 1

    while len(heap) > 1:
        _, a = heap.pop()
        _, b = heap.pop()
        parent = Node(a.weight + b.weight, left=a, right=b)
        heap.push((parent.weight, sequence), parent)
        sequence += 1

    return heap.pop()[1]


def code_lengths(root):
    """{symbol: depth} for every leaf under root."""
    if root.is_leaf:
        # A single distinct symbol would sit at depth 0 and get an empty
        # code, which the decoder can't count. Give it one bit instead.
        return {root.symbol: 1}

    lengths = {}
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.is_leaf:
            lengths[node.symbol] = depth
        else:
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
    return lengths


def _leaf_weights(root):
    weights = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            weights[node.symbol] = node.weight
        else:
            stack.append(node.left)
            stack.append(node.right)
    return weights


def limited_code_lengths(root, max_length=MAX_CODE_LENGTH):
    """Code lengths from the tree, capped at max_length.

    Very skewed frequencies can produce codes deeper than the stream format
    allows. When that happens, halve every weight (keeping each at least 1)
    and rebuild. Flattening the distribution makes the tree more balanced,
    so this terminates once the weights are close enough to equal. The
    result is slightly suboptimal, but only in cases that were at the limit.
    """
    lengths = code_lengths(root)
    if len(lengths) > 1 << max_length:
        raise ValueError(
            f"{len(lengths)} symbols cannot fit in codes of "
            f"{max_length} bits or fewer")

    weights = _leaf_weights(root)
    while max(lengths.values()) > max_length:
        weights = {s: (w + 1) // 2 for s, w in weights.items()}
        lengths = code_lengths(build_tree(weights))
    return lengths


def canonical_codes(lengths):
    """Assign canonical codes from {symbol: length}.

    Returns {symbol: (length, code)}. Within a length, codes are consecutive
    integers in symbol order; each length starts where the previous one
    ended, shifted left by one. This is the same construction DEFLATE uses.
    """
    if not lengths:
        return {}
    max_length = max(lengths.values())
    counts = [0] * (max_length + 1)
    for length in lengths.values():
        counts[length] += 1

    next_code = [0] * (max_length + 1)
    code = 0
    for length in range(1, max_length + 1):
        code = (code + counts[length - 1]) << 1
        next_code[length] = code

    codes = {}
    for symbol in sorted(lengths, key=lambda s: (lengths[s], s)):
        length = lengths[symbol]
        codes[symbol] = (length, next_code[length])
        next_code[length] += 1
    return codes


def build_codes(root, max_length=MAX_CODE_LENGTH):
    """{symbol: (bit_length, value)} canonical codes for the tree's leaves."""
    return canonical_codes(limited_code_lengths(root, max_length))


def encode(symbols, codes, writer):
    """Write each symbol's code to a BitWriter."""
    write = writer.write
    try:
        for symbol in symbols:
            length, value = codes[symbol]
            write(value, length)
    except KeyError as error:
        raise ValueError(f"symbol {error.args[0]!r} has no code") from None
    return writer


class Decoder:
    """Decodes canonical Huffman codes, given only the code lengths.

    Short codes (the common ones, by construction) are decoded with a single
    table lookup on the next LOOKUP_BITS bits. Longer ones fall back to the
    canonical bit-by-bit walk, which needs just three small arrays.
    """

    LOOKUP_BITS = 10

    def __init__(self, lengths):
        if not lengths:
            raise ValueError("no code lengths")
        if min(lengths.values()) < 1:
            raise ValueError("code lengths must be at least 1")
        max_length = max(lengths.values())

        # Kraft inequality. If it fails, the lengths describe codes that
        # overlap, which only happens with corrupted input.
        if sum(1 << (max_length - n) for n in lengths.values()) > 1 << max_length:
            raise ValueError("code lengths do not form a prefix code")

        codes = canonical_codes(lengths)
        self.max_length = max_length

        # Tables for the slow path. At each length, codes run consecutively
        # from first_code[length]; symbols lists them in the same order.
        self.symbols = sorted(lengths, key=lambda s: (lengths[s], s))
        self.count = [0] * (max_length + 1)
        self.first_code = [0] * (max_length + 1)
        self.first_index = [0] * (max_length + 1)
        # symbols is sorted by length, so each length's run starts at the
        # first symbol of that length. One pass fills all three arrays.
        for index, symbol in enumerate(self.symbols):
            length = lengths[symbol]
            if self.count[length] == 0:
                self.first_index[length] = index
                self.first_code[length] = codes[symbol][1]
            self.count[length] += 1

        # Fast path: every LOOKUP_BITS-bit pattern that starts with a short
        # code maps straight to (symbol, length).
        self.lookup_bits = min(self.LOOKUP_BITS, max_length)
        self.table = [None] * (1 << self.lookup_bits)
        for symbol, (length, value) in codes.items():
            if length <= self.lookup_bits:
                shift = self.lookup_bits - length
                start = value << shift
                entry = (symbol, length)
                for pattern in range(start, start + (1 << shift)):
                    self.table[pattern] = entry

    def decode_one(self, reader):
        entry = self.table[reader.peek(self.lookup_bits)]
        if entry is not None:
            reader.skip(entry[1])
            return entry[0]

        code = 0
        for length in range(1, self.max_length + 1):
            code = (code << 1) | reader.read_bit()
            offset = code - self.first_code[length]
            if 0 <= offset < self.count[length]:
                return self.symbols[self.first_index[length] + offset]
        raise ValueError("bit pattern matches no code")


def decode(reader, decoder, count):
    """Read `count` symbols from a BitReader using a Decoder."""
    decode_one = decoder.decode_one
    return [decode_one(reader) for _ in range(count)]
