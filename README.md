# Audio Compressor

A lossless audio codec built on Huffman coding, in pure Python with no
runtime dependencies.

```
python cli.py compress   input.wav  output.ahuf [--block-size N] [--min-block-size N]
python cli.py decompress input.ahuf output.wav
python cli.py stats      input.wav  [--blocks]
python benchmark.py      samples/*.wav
```

## How it works

- `src/huffman.py`: greedy tree construction on a hand-written binary heap,
  then **canonical** code assignment. Only code lengths are stored, so
  tie-breaking during tree construction can't desynchronise the decoder.
  Codes are capped at 20 bits. Decoding uses a 10-bit lookup table with a
  canonical bit-by-bit fallback for longer codes.
- `src/bitio.py`: MSB-first `BitWriter` / `BitReader`, plus Elias gamma codes.
- `src/codec.py`:
  - **Prediction.** Every block is stored as whichever is smallest:
    `verbatim`, `constant` (e.g. silence), or Huffman codes for the residual
    of a fixed predictor of order 0–3 (the predictors FLAC uses; order 1 is
    plain delta encoding). The choice uses exact sizes, tables included.
  - **Adaptive blocks.** Each 65,536-sample superblock is split in half
    recursively, down to 4,096 samples, wherever splitting makes the output
    smaller. Steady audio gets long blocks to spread the table's cost;
    changing audio gets short blocks that each use their own mode.
  - **Compact tables.** Symbols are stored as gamma-coded gaps; code lengths
    as changes from the previous length, Huffman coded with a second small
    table. That's under 4 bits per entry on typical audio.
  - **Integrity.** The header holds an MD5 of the decoded audio, as FLAC's
    does, so corruption that still parses is caught too.

  The container format is documented at the top of `codec.py`.
- `src/audio_io.py`: PCM WAV I/O (8/16/24/32-bit), keeping samples as
  integers.

## Results

From `python benchmark.py samples/*.wav`. Sizes are a percentage of the WAV
file. Every round trip is checked to be exact.

| file | length | this codec | gzip -9 | bz2 -9 | xz -9 |
|---|---|---|---|---|---|
| `speech.wav` (16-bit, 22.05 kHz) | 17.0 s | **69.2%** | 84.5% | 73.6% | 77.0% |
| `tones.wav` (16-bit, 22.05 kHz) | 3.0 s | 71.7% | 20.1% | 25.3% | **19.0%** |
| `stereo_tones.wav` (24-bit, 22.05 kHz) | 3.0 s | 52.8% | 20.5% | 25.7% | **19.8%** |

`speech.wav` was made with macOS text-to-speech (`say`, then `afconvert`).
It isn't a microphone recording, but unlike the tone files it isn't
periodic.

The tone files are exactly periodic synthetic sines, so the same bytes
repeat every cycle. Dictionary compressors (gzip, xz) find those repeats and
win easily. A sample-by-sample predictor can't exploit exact repetition, and
real recordings don't have it, so these files understate the codec. Use a
real recording for any comparison that matters.

## Performance and limitations

- **Encoding is slow: about 100k samples/s**, so roughly 4–5 s per 10 s of
  44.1 kHz mono. The encoder tries four predictors at every level of the
  block-split tree. A lower bound on each candidate's size skips the ones
  that can't win (exactly, never changing the result), but on correlated
  audio most candidates survive. `--min-block-size 65536` turns splitting
  off and is several times faster, for a small loss on audio that changes.
- **Decoding is about 450k samples/s**, over 10× real time at 44.1 kHz.
- Channels are coded independently. Mid/side stereo would help recordings
  where the channels are similar.
- Predictors are fixed polynomials. Fitting per-block linear predictors
  (LPC, as FLAC does) and coding residuals with Rice codes, which need no
  table, are the next steps toward FLAC's ratios.
- The format stores sample counts in 32 bits: up to about 27 hours per
  channel at 44.1 kHz.
- Small `--block-size` values leave blocks too short for a code table to
  pay for itself, so they fall back to verbatim; `compress` warns when that
  affects at least 10% of the audio.

## Tests

```
pip install -r requirements-dev.txt
python -m pytest
```
