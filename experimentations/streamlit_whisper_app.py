"""Quick Streamlit UI to record speech and transcribe it word by word."""

from __future__ import annotations

from typing import Iterator

import streamlit as st

from memento.audio import (
    FasterWhisperBackend,
    WhisperConfig,
    WhisperTranscriber,
    speech_segment_from_wav_bytes,
)


@st.cache_resource(show_spinner=False)
def get_transcriber(
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    beam_size: int,
) -> WhisperTranscriber:
    config = WhisperConfig(
        model_name=model_name,
        language=language,
        min_segment_duration_ms=0,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    return WhisperTranscriber(backend=FasterWhisperBackend(config=config), config=config)


def stream_words(words: tuple) -> Iterator[str]:
    for word in words:
        token = getattr(word, "word", "").strip()
        if token:
            yield token + " "


def main() -> None:
    st.set_page_config(page_title="Memento Whisper Mic", page_icon="microphone", layout="centered")
    st.title("Memento Whisper Microphone")
    st.caption("Record a short sentence, then transcribe it word by word with a real faster-whisper backend.")

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Whisper model",
            options=("tiny", "base", "small", "medium"),
            index=2,
            help="Smaller models start faster; larger models are usually more accurate.",
        )
        language = st.text_input("Language", value="fr")
        device = st.selectbox("Device", options=("cpu", "cuda"), index=0)
        compute_type = st.selectbox("Compute type", options=("int8", "float32", "float16"), index=0)
        beam_size = st.slider("Beam size", min_value=1, max_value=8, value=5)

    audio_value = st.audio_input("Record a sentence", sample_rate=16_000)

    if audio_value is None:
        st.info("Use the microphone button above to record audio.")
        return

    st.audio(audio_value)

    if st.button("Transcribe", type="primary"):
        transcriber = get_transcriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
        )
        segment = speech_segment_from_wav_bytes(audio_value.getvalue())

        with st.spinner("Transcribing with faster-whisper..."):
            transcription = transcriber.transcribe_segment(segment)

        if transcription is None or not transcription.text:
            st.warning("No speech was transcribed from this recording.")
            return

        st.subheader("Word by word")
        if transcription.words:
            st.write_stream(stream_words(transcription.words))
        else:
            st.write(transcription.text)

        st.subheader("Full transcription")
        st.write(transcription.text)
        st.caption(
            f"Latency: {transcription.latency_ms:.0f} ms | "
            f"Duration: {transcription.duration_ms:.0f} ms | "
            f"Model: {transcription.model_name}"
        )

        if transcription.words:
            st.subheader("Timestamps")
            st.dataframe(
                [
                    {
                        "word": word.word,
                        "start_ms": word.start_ms,
                        "end_ms": word.end_ms,
                        "probability": word.probability,
                    }
                    for word in transcription.words
                ],
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
