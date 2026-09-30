"""
WAV reading and writing, using only the standard library `wave` module.

Adapted from the Audio_Equalizer project, with one deliberate difference:
samples stay as signed integers rather than being scaled to floats. A
lossless codec has to give back exactly the integers it was given, and a
round trip through floats is one more place for that to go wrong.
"""

import struct
import wave


class UnsupportedAudioError(Exception):
    """A WAV file this module cannot read."""


def _decode(raw, sample_width, count):
    # Turn packed bytes into signed integers. 8-bit WAV is the odd one out:
    # it is stored unsigned with an offset of 128.
    #
    # A file whose data chunk is shorter than its header claims is truncated.
    # Refuse it rather than silently padding the end with silence.
    expected = count * sample_width
    if len(raw) < expected:
        raise UnsupportedAudioError(
            f"file is truncated: header declares {expected} bytes of audio "
            f"but only {len(raw)} are present"
        )

    if sample_width == 1:
        return [b - 128 for b in raw[:count]]
    if sample_width == 2:
        return list(struct.unpack(f"<{count}h", raw[:count * 2]))
    if sample_width == 4:
        return list(struct.unpack(f"<{count}i", raw[:count * 4]))
    if sample_width == 3:
        return [
            int.from_bytes(raw[i:i + 3], "little", signed=True)
            for i in range(0, count * 3, 3)
        ]
    raise UnsupportedAudioError(
        f"unsupported sample width: {sample_width} bytes")


def _encode(values, sample_width):
    # Inverse of _decode.
    if sample_width == 1:
        return bytes((v + 128) & 0xFF for v in values)
    if sample_width == 2:
        return struct.pack(f"<{len(values)}h", *values)
    if sample_width == 4:
        return struct.pack(f"<{len(values)}i", *values)
    if sample_width == 3:
        out = bytearray()
        for v in values:
            out.extend(v.to_bytes(3, "little", signed=True))
        return bytes(out)
    raise ValueError(f"unsupported sample width: {sample_width} bytes")


def read_wav(source):
    """Read a PCM WAV file.

    Args:
        source: a path, or any file-like object `wave` can open.

    Returns:
        (channels, sample_rate, sample_width) where channels is a list of
        channels, each a list of signed integer samples.

    Raises:
        UnsupportedAudioError: for formats this module cannot read, and for
            files whose header disagrees with their contents.
    """
    try:
        with wave.open(source, "rb") as wf:
            channel_count = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            raw = wf.readframes(frame_count)
    except wave.Error as error:
        if "unknown format: 3" in str(error):
            raise UnsupportedAudioError(
                "this is a 32-bit floating-point WAV, which this reader does "
                "not support. Re-export it as 16- or 24-bit PCM."
            ) from error
        raise UnsupportedAudioError(f"could not read as WAV: {error}") from error

    if sample_rate <= 0:
        raise UnsupportedAudioError(
            f"file declares a sample rate of {sample_rate} Hz")
    if channel_count <= 0:
        raise UnsupportedAudioError(
            f"file declares {channel_count} channels")

    values = _decode(raw, sample_width, frame_count * channel_count)
    channels = [values[c::channel_count] for c in range(channel_count)]
    return channels, sample_rate, sample_width


def write_wav(destination, channels, sample_rate, sample_width=2):
    """Write integer channels out as a PCM WAV file."""
    if not channels:
        raise ValueError("no channels to write")
    if len({len(c) for c in channels}) != 1:
        raise ValueError("channels must all be the same length")

    low = -(1 << (sample_width * 8 - 1))
    high = -low - 1
    interleaved = [sample for frame in zip(*channels) for sample in frame]
    for sample in interleaved:
        if not low <= sample <= high:
            raise ValueError(
                f"sample {sample} does not fit in {sample_width} bytes")

    with wave.open(destination, "wb") as wf:
        wf.setnchannels(len(channels))
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(_encode(interleaved, sample_width))
