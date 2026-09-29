# Audio Equalizer

A graphic equaliser and spectrum analyser built on a from-scratch implementation
of the Fast Fourier Transform. No FFT library is used - the transform is written
directly from the Cooley-Tukey recurrence.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests/ -v
```

The FFT is checked against a naive O(n²) transform written straight from the
definition, and against `numpy.fft` as an independent reference. numpy is used
only for that check, never by the equaliser itself.

## Layout

```
src/
  fft.py        Cooley-Tukey FFT, inverse FFT, naive DFT reference
  filters.py    Per-bin gain curves for the equaliser bands
  stft.py       Framing, Hann windowing, overlap-add synthesis
  audio_io.py   WAV reading and writing
  plots.py      Waveform, spectrum, and spectrogram figures
app.py          Streamlit interface
tests/          Correctness tests
```

Dependencies point one way only: `fft.py` knows nothing about audio, and the
audio layer knows nothing about the interface.

## How it works

1. Read the WAV file and convert samples to floats
2. Split each channel into overlapping frames and apply a Hann window
3. Transform each frame to the frequency domain with the FFT
4. Multiply each frequency bin by the gain for its band
5. Transform back with the inverse FFT
6. Window again and reassemble the frames with overlap-add
7. Turn the result down if it would clip, then write it out

Channels are processed independently rather than mixed to mono, so a stereo
file stays stereo. With every band at 0 dB the whole chain reproduces the
input exactly, which is the main correctness check.

## Known limitations

- **Speed.** The FFT is pure Python, roughly 0.2x real time, so the interface
  caps how much audio it will process at once. An iterative FFT with a
  bit-reversal permutation would help; the recursive form was kept because it
  maps directly onto the Cooley-Tukey recurrence.
- **Band gains are exact at band centres only.** Between centres the gain is
  interpolated in log-frequency space, so asking for +12 dB of bass gives
  +12 dB at 71 Hz and less toward 250 Hz. Abrupt band edges would be a
  brick-wall filter, which rings.
- **PCM WAV only.** 32-bit floating-point WAV, MP3, and other formats are
  rejected with a message rather than supported.
- **Bands above half the sample rate are dropped.** A file recorded at 8 kHz
  carries nothing above 4 kHz, so the treble control has no bins to act on and
  the interface says so.
