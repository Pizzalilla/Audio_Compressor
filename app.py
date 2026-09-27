"""
Streamlit front end for the equaliser.

Run with:  streamlit run app.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from audio_io import read_wav, write_wav, to_mono
from filters import BANDS, band_centres, graphic_eq_curve
from plots import plot_spectrum_comparison, plot_spectrogram, plot_waveform
from stft import frame_positions, limit_peak, process

PRESETS = {
    "Flat": {"bass": 0, "mid": 0, "treble": 0},
    "Bass boost": {"bass": 8, "mid": 0, "treble": 0},
    "Treble boost": {"bass": 0, "mid": 0, "treble": 8},
    "Scooped": {"bass": 5, "mid": -6, "treble": 5},
    "Telephone": {"bass": -18, "mid": 6, "treble": -18},
}


def to_wav_bytes(channels, sample_rate):
    buffer = io.BytesIO()
    write_wav(buffer, channels, sample_rate)
    return buffer.getvalue()


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
        overlap = st.select_slider(
            "Overlap", options=["50%", "75%"], value="75%",
            help="75% is more robust when the spectrum is modified heavily.",
        )
        max_seconds = st.slider(
            "Seconds to process", 1, 30, 5,
            help="The FFT here is pure Python, so long files are slow.",
        )

    if uploaded is None:
        st.info("Upload a WAV file to begin.")
        return

    channels, sample_rate = read_wav(uploaded)
    mono = to_mono(channels)

    limit = int(max_seconds * sample_rate)
    truncated = len(mono) > limit
    if truncated:
        mono = mono[:limit]

    duration = len(mono) / sample_rate
    columns = st.columns(4)
    columns[0].metric("Sample rate", f"{sample_rate:,} Hz")
    columns[1].metric("Channels", len(channels))
    columns[2].metric("Duration", f"{duration:.1f} s")
    columns[3].metric("Samples", f"{len(mono):,}")
    if truncated:
        st.warning(f"Only the first {max_seconds}s are being processed.")

    hop_size = frame_size // (2 if overlap == "50%" else 4)
    frames = len(frame_positions(len(mono), frame_size, hop_size))
    st.caption(f"{frames:,} frames of {frame_size} samples, hop {hop_size}.")

    # A band lying above Nyquist has no bins to act on, so its slider would
    # do nothing at all. Say so rather than letting it look broken.
    reachable = len(band_centres(gains, sample_rate))
    if reachable < len(BANDS):
        unreachable = list(BANDS)[reachable:]
        st.warning(
            f"This file's sample rate only carries audio up to "
            f"{sample_rate // 2:,} Hz, so these controls have nothing to "
            f"act on: {', '.join(unreachable)}."
        )

    if not st.button("Apply equaliser", type="primary"):
        st.pyplot(plot_waveform(mono, sample_rate, "Input waveform"))
        return

    bar = st.progress(0.0, text="Processing...")

    def report(done, total):
        bar.progress(min(1.0, done / total), text=f"Frame {done:,} of {total:,}")

    processed = process(
        mono, sample_rate,
        lambda n, sr: graphic_eq_curve(n, sr, gains),
        frame_size=frame_size,
        hop_size=hop_size,
        progress_fn=report,
    )
    bar.empty()

    processed, attenuation_db, peak = limit_peak(processed)
    if attenuation_db > 0:
        st.info(
            f"Boosting pushed the peak to {peak:.2f}, above what a WAV file "
            f"can store. Output turned down by {attenuation_db:.1f} dB so it "
            f"fits. The balance between bands is unchanged."
        )

    original_bytes = to_wav_bytes([mono], sample_rate)
    processed_bytes = to_wav_bytes([processed], sample_rate)

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

    st.pyplot(plot_spectrum_comparison(mono, processed, sample_rate, gains))

    with st.expander("Spectrograms"):
        st.pyplot(plot_spectrogram(mono, sample_rate))
        st.pyplot(plot_spectrogram(processed, sample_rate))


if __name__ == "__main__":
    main()
