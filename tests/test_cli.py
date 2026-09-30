"""
Tests for the command-line interface.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_io import write_wav
from cli import main


def write_tone(path, n=20000):
    # A tone with a little noise on it, like a real recording. A perfectly
    # clean sine is predicted almost exactly, so even tiny blocks compress.
    rng = random.Random(0)
    samples = [round(8000 * math.sin(2 * math.pi * 440 * i / 44100)
                     + rng.gauss(0, 200))
               for i in range(n)]
    write_wav(str(path), [samples, samples[::-1]], 44100, 2)


def test_compress_decompress_is_byte_identical(tmp_path, capsys):
    wav, packed, out = tmp_path / "a.wav", tmp_path / "a.ahuf", tmp_path / "b.wav"
    write_tone(wav)
    assert main(["compress", str(wav), str(packed)]) == 0
    assert main(["decompress", str(packed), str(out)]) == 0
    assert wav.read_bytes() == out.read_bytes()
    assert "warning" not in capsys.readouterr().err


def test_small_blocks_warn(tmp_path, capsys):
    wav, packed = tmp_path / "a.wav", tmp_path / "a.ahuf"
    write_tone(wav)
    assert main(["compress", str(wav), str(packed), "--block-size", "64"]) == 0
    assert "--block-size" in capsys.readouterr().err


def test_stats_runs(tmp_path, capsys):
    wav = tmp_path / "a.wav"
    write_tone(wav)
    assert main(["stats", str(wav), "--blocks"]) == 0
    output = capsys.readouterr().out
    assert "bits/sample" in output and "order" in output


def test_errors_are_reported_not_raised(tmp_path, capsys):
    bogus = tmp_path / "bogus.ahuf"
    bogus.write_bytes(b"not a compressed file at all")
    assert main(["decompress", str(bogus), str(tmp_path / "x.wav")]) == 1
    assert main(["compress", str(tmp_path / "missing.wav"),
                 str(tmp_path / "x.ahuf")]) == 1
    assert "error:" in capsys.readouterr().err
