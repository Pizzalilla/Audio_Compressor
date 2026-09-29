"""
Tests for WAV reading and writing.
"""

import io
import math
import os
import struct
import sys
import wave

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_io import UnsupportedAudioError, read_wav, to_mono, write_wav


def tone(n, freq=440, sample_rate=8000, amplitude=0.5):
    return [amplitude * math.sin(2 * math.pi * freq * i / sample_rate)
            for i in range(n)]


def roundtrip(channels, sample_rate=8000, sample_width=2):
    buffer = io.BytesIO()
    write_wav(buffer, channels, sample_rate, sample_width)
    buffer.seek(0)
    return read_wav(buffer)


def test_roundtrip_preserves_shape_and_rate():
    for width in [1, 2, 3, 4]:
        for channel_count in [1, 2]:
            original = [tone(500, freq=300 * (c + 1)) for c in range(channel_count)]
            channels, rate, read_width = roundtrip(original, 8000, width)
            assert len(channels) == channel_count
            assert len(channels[0]) == 500
            assert rate == 8000
            assert read_width == width


def test_roundtrip_accuracy_improves_with_bit_depth():
    original = [tone(500)]
    errors = {}
    for width in [1, 2, 3]:
        channels, _, _ = roundtrip(original, 8000, width)
        errors[width] = max(abs(a - b) for a, b in zip(original[0], channels[0]))
    assert errors[1] > errors[2] > errors[3]
    assert errors[2] < 1e-4


def test_writing_rounds_rather_than_truncating():
    # Truncating biases every sample toward zero by up to one step, which
    # shows up as most samples changing even when nothing should have.
    original = [tone(2000, amplitude=0.3)]
    channels, _, _ = roundtrip(original, 8000, 2)
    errors = [b - a for a, b in zip(original[0], channels[0])]
    # With rounding the error is centred on zero; with truncation every
    # error points toward zero and the mean is strongly biased.
    assert abs(sum(errors) / len(errors)) < 1e-5


def test_channels_stay_separate():
    left = [0.5] * 100
    right = [-0.25] * 100
    channels, _, _ = roundtrip([left, right])
    assert channels[0][0] == pytest.approx(0.5, abs=1e-4)
    assert channels[1][0] == pytest.approx(-0.25, abs=1e-4)


def test_values_outside_range_are_clipped():
    channels, _, _ = roundtrip([[2.0, -2.0, 0.0]])
    assert max(channels[0]) <= 1.0
    assert min(channels[0]) >= -1.0


def test_to_mono_averages():
    assert to_mono([[1.0, 0.0]]) == [1.0, 0.0]
    assert to_mono([[1.0, 0.0], [0.0, 1.0]]) == [0.5, 0.5]
    # Anti-phase channels cancel, which is why the app processes channels
    # separately rather than mixing first.
    assert to_mono([[0.5], [-0.5]]) == [0.0]


def test_rejects_float_wav_with_a_useful_message():
    # 32-bit float (format tag 3) is a common export from audio editors and
    # the stdlib wave module cannot read it.
    buffer = io.BytesIO()
    data = struct.pack("<4f", 0.0, 0.5, -0.5, 0.0)
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 3, 1, 8000, 32000, 4, 32)
              + b"data" + struct.pack("<I", len(data)) + data)
    buffer.write(header)
    buffer.seek(0)
    with pytest.raises(UnsupportedAudioError, match="floating-point"):
        read_wav(buffer)


def test_rejects_truncated_file_rather_than_padding_with_silence():
    buffer = io.BytesIO()
    write_wav(buffer, [tone(1000)], 8000, 2)
    raw = bytearray(buffer.getvalue())
    del raw[-600:]  # lop off the tail of the data chunk
    with pytest.raises(UnsupportedAudioError, match="truncated"):
        read_wav(io.BytesIO(bytes(raw)))


def test_rejects_zero_sample_rate():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1)
        wf.writeframes(b"\x00\x00" * 10)
    raw = bytearray(buffer.getvalue())
    rate_offset = raw.find(b"fmt ") + 12
    raw[rate_offset:rate_offset + 4] = struct.pack("<I", 0)
    with pytest.raises(UnsupportedAudioError, match="sample rate"):
        read_wav(io.BytesIO(bytes(raw)))


def test_rejects_ragged_channels():
    with pytest.raises(ValueError, match="same length"):
        write_wav(io.BytesIO(), [[0.0, 0.0], [0.0]], 8000)


def test_rejects_empty_channel_list():
    with pytest.raises(ValueError, match="no channels"):
        write_wav(io.BytesIO(), [], 8000)


def test_handles_empty_and_single_sample_files():
    for length in [0, 1]:
        channels, rate, _ = roundtrip([[0.5] * length])
        assert len(channels[0]) == length
