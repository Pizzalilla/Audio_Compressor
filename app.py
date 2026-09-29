"""
Streamlit front end for the equaliser.

Run with:  streamlit run app.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from audio_io import UnsupportedAudioError, read_wav, to_mono, write_wav
from filters import BANDS, band_centres, graphic_eq_curve
from plots import plot_spectrogram, plot_spectrum_comparison, plot_waveform
from stft import frame_positions, limit_peak, process

# Attenuation below this is inaudible and not worth telling the user about.
AUDIBLE_ATTENUATION_DB = 0.1

PRESETS = {
    "Flat": {"bass": 0, "mid": 0, "treble": 0},
    "Bass boost": {"bass": 8, "mid": 0, "treble": 0},
    "Treble boost": {"bass": 0, "mid": 0, "treble": 8},
    "Scooped": {"bass": 5, "mid": -6, "treble": 5},
    "Telephone": {"bass": -18, "mid": 6, "treble": -18},
}


def to_wav_bytes(channels, sample_rate, sample_width):
    buffer = io.BytesIO()
    write_wav(buffer, channels, sample_rate, sample_width)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_audio(raw_bytes):
    # Keyed on the file's bytes, so changing a slider no longer re-decodes
    # the whole upload. Returns floats, which are what everything downstream
    # works in.
    return read_wav(io.BytesIO(raw_bytes))


@st.cache_data(show_spinner=False)
def equalise(raw_bytes, gains, frame_size, hop_size, max_seconds):
    # Cached on everything that affects the result, so a rerun triggered by
    # the download button or a layout change does not reprocess the audio.
    channels, sample_rate, sample_width = load_audio(raw_bytes)
    limit = int(max_seconds * sample_rate)
    trimmed = [channel[:limit] for channel in channels]

    outputs = [
        process(channel, sample_rate,
                lambda n, sr: graphic_eq_curve(n, sr, dict(gains)),
                frame_size=frame_size, hop_size=hop_size)
        for channel in trimmed
    ]

    outputs, attenuation, peak = limit_peak(outputs)
    return trimmed, outputs, sample_rate, sample_width, attenuation, peak


def main():
    st.set_page_config(page_title="Audio Equalizer", layout="wide")
    st.title("Audio Equalizer")
    st.caption("Frequency-domain equalisation built on a from-scratch FFT.")

    uploaded = st.file_uploader("WAV file", type=["wav"])

    with st.sidebar:
        st.header("Equaliser")

        preset = st.selectbox("Preset", list(PRESETS.keys()))
        defaults = PRESETS[preset]

        gains = {}
        for name, (low, high) in BANDS.items():
            gains[name] = st.slider(
                f"{name.title()}  ({low:,}-{high:,} Hz)",
                min_value=-24.0, max_value=24.0,
                value=float(defaults.get(name, 0)),
                step=0.5, format="%.1f dB",
                key=f"{preset}_{name}",
            )

        st.header("Analysis")
        frame_size = st.select_slider(
            "Frame size", options=[512, 1024, 2048, 4096], value=2048,
            help="Larger frames resolve frequency better but smear time.",
        )
        max_seconds = st.slider(
            "Seconds to process", 1, 30, 5,
            help="The FFT here is pure Python, so long files are slow.",
        )

    if uploaded is None:
        st.info("Upload a WAV file to begin.")
        return

    raw_bytes = uploaded.getvalue()
    try:
        channels, sample_rate, sample_width = load_audio(raw_bytes)
    except UnsupportedAudioError as error:
        st.error(f"Could not read this file: {error}")
        return

    # A quarter-frame hop is the coarsest that reconstructs cleanly once the
    # spectrum has been altered; see MIN_OVERLAP_FACTOR in stft.py.
    hop_size = frame_size // 4

    limit = int(max_seconds * sample_rate)
    total_samples = len(channels[0])
    truncated = total_samples > limit
    shown = min(total_samples, limit)

    columns = st.columns(4)
    columns[0].metric("Sample rate", f"{sample_rate:,} Hz")
    columns[1].metric("Channels", len(channels))
    columns[2].metric("Bit depth", f"{sample_width * 8}-bit")
    columns[3].metric("Duration", f"{shown / sample_rate:.1f} s")
    if truncated:
        st.warning(f"Only the first {max_seconds}s are being processed.")

    frames = len(frame_positions(shown, frame_size, hop_size)) * len(channels)
    st.caption(
        f"{frames:,} frames of {frame_size} samples, hop {hop_size}, "
        f"across {len(channels)} channel{'s' if len(channels) > 1 else ''}."
    )

    # A band lying above Nyquist has no bins to act on, so its slider would
    # do nothing at all. Say so rather than letting it look broken.
    active = {name for _, _, name in band_centres(gains, sample_rate)}
    inert = [name for name in BANDS if name not in active]
    if inert:
        st.warning(
            f"This file only carries audio up to {sample_rate // 2:,} Hz, so "
            f"these controls have nothing to act on: {', '.join(inert)}."
        )

    settings = (uploaded.name, len(raw_bytes), tuple(sorted(gains.items())),
                frame_size, hop_size, max_seconds)

    # Remember that the user asked for this, rather than reading the button
    # directly. Streamlit reruns the whole script on every interaction, and
    # a button reads False on any rerun it did not itself cause -- including
    # the one the download button triggers, which used to wipe the results.
    if st.button("Apply equaliser", type="primary"):
        st.session_state["applied"] = settings

    applied = st.session_state.get("applied")
    if applied is None:
        st.pyplot(plot_waveform(to_mono(channels)[:limit], sample_rate,
                                "Input waveform"))
        return

    if applied != settings:
        st.caption("Settings have changed since this was rendered. "
                   "Press Apply equaliser to update.")

    _, _, frozen_gains, frame_size, hop_size, max_seconds = applied
    gains = dict(frozen_gains)

    with st.spinner("Processing..."):
        (original, processed, sample_rate, sample_width,
         attenuation_db, peak) = equalise(
            raw_bytes, frozen_gains, frame_size, hop_size, max_seconds)

    # Reconstruction can overshoot a full-scale input by a fraction of a
    # decibel purely through floating-point error, and limit_peak dutifully
    # scales that away. Reporting it would tell the user something happened
    # when the 16-bit output is bit-identical either way.
    if attenuation_db >= AUDIBLE_ATTENUATION_DB:
        st.info(
            f"Boosting pushed the peak to {peak:.2f}, above what a WAV file "
            f"can store. Output turned down by {attenuation_db:.1f} dB so it "
            f"fits. The balance between bands is unchanged."
        )

    original_bytes = to_wav_bytes(original, sample_rate, sample_width)
    processed_bytes = to_wav_bytes(processed, sample_rate, sample_width)

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.audio(original_bytes, format="audio/wav")
    with right:
        st.subheader("Equalised")
        st.audio(processed_bytes, format="audio/wav")
        st.download_button(
            "Download WAV", processed_bytes,
            file_name=f"eq_{uploaded.name}", mime="audio/wav",
        )

    before, after = to_mono(original), to_mono(processed)
    st.pyplot(plot_spectrum_comparison(before, after, sample_rate, gains))

    with st.expander("Spectrograms"):
        st.pyplot(plot_spectrogram(before, sample_rate))
        st.pyplot(plot_spectrogram(after, sample_rate))


if __name__ == "__main__":
    main()
