"""
Command-line interface.

    python cli.py compress   input.wav  output.ahuf [--block-size N] [--min-block-size N]
    python cli.py decompress input.ahuf output.wav
    python cli.py stats      input.wav  [--block-size N] [--min-block-size N] [--blocks]
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from audio_io import UnsupportedAudioError, read_wav, write_wav  # noqa: E402
from codec import (DEFAULT_BLOCK_SIZE, DEFAULT_MIN_BLOCK_SIZE,  # noqa: E402
                   MAX_ORDER, CodecError, analyse, compress_with_report,
                   decompress, entropy, residuals)

# A verbatim block whose best residual entropy is at least this many bits
# per sample below the sample width could have compressed, if its code table
# had had more samples to pay for itself over.
WASTE_MARGIN_BITS = 1.0
WASTE_WARN_FRACTION = 0.1


def table_limited_samples(channels, width, blocks):
    """How many samples went out verbatim only because blocks were small."""
    wasted = 0
    for block in blocks:
        if block["mode"] != "verbatim":
            continue
        samples = channels[block["channel"]][
            block["start"]:block["start"] + block["samples"]]
        best = min(entropy(residuals(samples, k)) for k in range(MAX_ORDER + 1))
        if best <= width * 8 - WASTE_MARGIN_BITS:
            wasted += block["samples"]
    return wasted


def cmd_compress(args):
    channels, rate, width = read_wav(args.input)
    data, blocks = compress_with_report(channels, rate, width,
                                        args.block_size, args.min_block_size)
    with open(args.output, "wb") as f:
        f.write(data)
    before = os.path.getsize(args.input)
    print(f"{before} -> {len(data)} bytes "
          f"({len(data) / before:.1%} of original)")

    total = sum(b["samples"] for b in blocks)
    wasted = table_limited_samples(channels, width, blocks)
    if total and wasted / total >= WASTE_WARN_FRACTION:
        print(f"warning: {wasted / total:.0%} of the audio was stored "
              f"uncompressed because its blocks were too short for a code "
              f"table to pay for itself. Try a larger --block-size "
              f"(currently {args.block_size}).", file=sys.stderr)


def cmd_decompress(args):
    with open(args.input, "rb") as f:
        channels, rate, width = decompress(f.read())
    write_wav(args.output, channels, rate, width)
    print(f"wrote {len(channels)} channel(s), {len(channels[0])} frames, "
          f"{width * 8}-bit, {rate} Hz")


def cmd_stats(args):
    channels, rate, width = read_wav(args.input)
    blocks = analyse(channels, width, args.block_size, args.min_block_size)
    total = sum(b["samples"] for b in blocks)
    if total == 0:
        print("file contains no samples")
        return

    print(f"{len(channels)} channel(s), {len(channels[0])} frames, "
          f"{width * 8}-bit, {rate} Hz, {len(blocks)} blocks")
    for order in range(MAX_ORDER + 1):
        weighted = sum(b["entropy"][order] * b["samples"] for b in blocks) / total
        print(f"entropy, order {order} residual: {weighted:6.2f} bits/sample")
    chosen_bits = sum(b["bits"] for b in blocks)
    table_bits = sum(b["table_bits"] for b in blocks)
    print(f"achieved (incl tables):     {chosen_bits / total:6.2f} bits/sample "
          f"vs {width * 8} uncompressed")
    print(f"code tables:                {table_bits / total:6.2f} bits/sample")

    samples_by_mode = Counter()
    for b in blocks:
        samples_by_mode[b["mode"]] += b["samples"]
    print("audio by mode:              " + ", ".join(
        f"{mode} {count / total:.0%}"
        for mode, count in samples_by_mode.most_common()))

    if args.blocks:
        print()
        print(f"{'ch':>2} {'start':>9} {'samples':>8} {'mode':>8} "
              f"{'bits':>9} {'table':>7} {'bits/sample':>11}")
        for b in blocks:
            print(f"{b['channel']:>2} {b['start']:>9} {b['samples']:>8} "
                  f"{b['mode']:>8} {b['bits']:>9} {b['table_bits']:>7} "
                  f"{b['bits'] / b['samples']:>11.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lossless Huffman audio codec")
    sub = parser.add_subparsers(dest="command", required=True)

    def block_options(p):
        p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                       help="largest block, in samples (default %(default)s)")
        p.add_argument("--min-block-size", type=int,
                       default=DEFAULT_MIN_BLOCK_SIZE,
                       help="smallest block the encoder may split down to "
                            "(default %(default)s)")

    p = sub.add_parser("compress", help="WAV -> compressed file")
    p.add_argument("input")
    p.add_argument("output")
    block_options(p)
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("decompress", help="compressed file -> WAV")
    p.add_argument("input")
    p.add_argument("output")
    p.set_defaults(func=cmd_decompress)

    p = sub.add_parser("stats", help="show how a WAV would compress")
    p.add_argument("input")
    block_options(p)
    p.add_argument("--blocks", action="store_true", help="list every block")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    if getattr(args, "block_size", None) is not None:
        # Clamp the minimum rather than rejecting "--block-size 512" just
        # because it is below the default minimum.
        args.min_block_size = min(args.min_block_size, args.block_size)
    try:
        args.func(args)
    except (OSError, UnsupportedAudioError, CodecError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
