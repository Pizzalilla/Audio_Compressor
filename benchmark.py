"""
Compare this codec against general-purpose compressors on WAV files.

    python benchmark.py samples/*.wav

For each file: compressed size as a percentage of the WAV, for this codec
and for zlib (gzip), bz2 and lzma (xz) at their strongest settings, plus
this codec's encode and decode speed. Every round trip is checked.
"""

import bz2
import lzma
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from audio_io import read_wav  # noqa: E402
from codec import compress, decompress  # noqa: E402

GENERAL = {
    "gzip -9": lambda b: zlib.compress(b, 9),
    "bz2 -9": lambda b: bz2.compress(b, 9),
    "xz -9": lambda b: lzma.compress(b, preset=9),
}


def main(paths):
    if not paths:
        print(__doc__.strip())
        return 1

    names = ["this codec", *GENERAL]
    print(f"{'file':<22}{'seconds':>8}" + "".join(f"{n:>12}" for n in names)
          + f"{'encode':>14}{'decode':>14}")
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        channels, rate, width = read_wav(path)
        samples = sum(len(c) for c in channels)

        start = time.perf_counter()
        packed = compress(channels, rate, width)
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        restored = decompress(packed)
        decode_time = time.perf_counter() - start
        if restored != (channels, rate, width):
            raise SystemExit(f"{path}: round trip FAILED")

        sizes = [len(packed)] + [len(fn(raw)) for fn in GENERAL.values()]
        print(f"{os.path.basename(path)[:21]:<22}"
              f"{len(channels[0]) / rate:>8.1f}"
              + "".join(f"{s / len(raw):>12.1%}" for s in sizes)
              + f"{samples / encode_time / 1000:>10.0f}k/s"
              + f"{samples / decode_time / 1000:>10.0f}k/s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
