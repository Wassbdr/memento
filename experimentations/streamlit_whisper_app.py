"""Interactive Streamlit lab for the `memento.audio` package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import streamlit as st

from memento.audio import (
    AudioFrame,
    EnergyVAD,
    MicrophoneConfig,
    OpenAIWhisperBackend,
    RealTimeMicrophone,
    SegmentTranscription,
    SpeechSegment,
    SoundDeviceInput,
    StreamingSpeechSegmenter,
    VoiceActivityConfig,
    WhisperConfig,
    WhisperTranscriber,
    list_input_devices,
    load_wav_bytes,
    speech_segment_from_wav_bytes,
    torch_cuda_available,
    write_wav_file,
)


@dataclass(frozen=True)
class AudioAnalysis:
    """Computed metrics used by the Streamlit playground."""

    source_label: str
    duration_ms: float
    sample_rate_hz: int
    original_channels: int
    mono_samples: tuple[float, ...]
    normalized_wav_bytes: bytes
    frames: tuple[AudioFrame, ...]
    segments: tuple[SpeechSegment, ...]
    frame_rows: tuple[dict[str, float | int | bool], ...]
    segment_rows: tuple[dict[str, float | int], ...]
    silent_frames: int
    clipped_frames: int
    average_rms_level: float
    max_peak_level: float


@dataclass(frozen=True)
class LiveTranscriptionSession:
    """Captured live-audio session with incremental transcriptions."""

    device_name: str
    duration_ms: float
    audio_wav_bytes: bytes
    frame_rows: tuple[dict[str, float | int | bool], ...]
    segment_rows: tuple[dict[str, float | int | str | None], ...]
    transcriptions: tuple[SegmentTranscription, ...]
    silent_frames: int
    clipped_frames: int
    average_rms_level: float


def mix_down_to_mono(samples: tuple[float, ...], channels: int) -> tuple[float, ...]:
    """Average all channels into one mono stream."""

    if channels == 1:
        return samples

    mono_samples: list[float] = []
    for index in range(0, len(samples), channels):
        chunk = samples[index : index + channels]
        mono_samples.append(sum(chunk) / channels)
    return tuple(mono_samples)


def wav_bytes_from_samples(samples: tuple[float, ...], sample_rate_hz: int) -> bytes:
    """Encode normalized mono samples as a temporary WAV payload."""

    with NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        temp_path = Path(temporary_file.name)

    try:
        write_wav_file(
            path=temp_path,
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            channels=1,
        )
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def build_frames(samples: tuple[float, ...], config: MicrophoneConfig) -> tuple[AudioFrame, ...]:
    """Split a sample stream into fixed-size frames using microphone semantics."""

    frames: list[AudioFrame] = []
    frame_size = config.samples_per_frame
    elapsed_ms = 0.0

    for frame_index, start in enumerate(range(0, len(samples), frame_size)):
        chunk = samples[start : start + frame_size]
        if not chunk:
            continue
        if len(chunk) < frame_size:
            chunk = (*chunk, *((0.0,) * (frame_size - len(chunk))))
        frame = AudioFrame(
            samples=chunk,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            frame_index=frame_index,
            started_at_ms=elapsed_ms,
        )
        frames.append(frame)
        elapsed_ms += frame.duration_ms

    return tuple(frames)


@st.cache_data(show_spinner=False)
def analyze_audio(
    audio_bytes: bytes,
    source_label: str,
    frame_duration_ms: int,
    silence_threshold: float,
    clipping_threshold: float,
    speech_threshold: float,
    noise_floor: float,
    min_speech_frames: int,
    min_silence_frames: int,
    pre_roll_frames: int,
) -> AudioAnalysis:
    """Decode WAV bytes and compute VAD + capture-like diagnostics."""

    wav_audio = load_wav_bytes(audio_bytes)
    mono_samples = mix_down_to_mono(wav_audio.samples, wav_audio.channels)
    normalized_wav_bytes = wav_bytes_from_samples(mono_samples, wav_audio.sample_rate_hz)

    capture_config = MicrophoneConfig(
        device_name="streamlit-source",
        sample_rate_hz=wav_audio.sample_rate_hz,
        channels=1,
        frame_duration_ms=frame_duration_ms,
        silence_threshold=silence_threshold,
        clipping_threshold=clipping_threshold,
    )
    frames = build_frames(mono_samples, capture_config)

    vad = EnergyVAD(
        VoiceActivityConfig(
            speech_threshold=speech_threshold,
            noise_floor=noise_floor,
            min_speech_frames=min_speech_frames,
            min_silence_frames=min_silence_frames,
            pre_roll_frames=pre_roll_frames,
        )
    )
    segments = vad.segment(frames)

    frame_rows = []
    for frame in frames:
        decision = vad.classify(frame)
        frame_rows.append(
            {
                "frame_index": frame.frame_index,
                "started_at_ms": round(frame.started_at_ms, 1),
                "duration_ms": round(frame.duration_ms, 1),
                "rms_level": round(frame.rms_level, 4),
                "peak_level": round(frame.peak_level, 4),
                "threshold": round(decision.threshold, 4),
                "is_speech": decision.is_speech,
                "is_silent": frame.rms_level <= capture_config.silence_threshold,
                "is_clipped": frame.peak_level >= capture_config.clipping_threshold,
            }
        )

    segment_rows = []
    for index, segment in enumerate(segments, start=1):
        start_ms = segment.frames[0].started_at_ms
        end_ms = start_ms + segment.duration_ms
        segment_rows.append(
            {
                "segment_index": index,
                "start_frame_index": segment.start_frame_index,
                "end_frame_index": segment.end_frame_index,
                "start_ms": round(start_ms, 1),
                "end_ms": round(end_ms, 1),
                "duration_ms": round(segment.duration_ms, 1),
            }
        )

    total_frames = len(frames)
    average_rms_level = (
        sum(frame.rms_level for frame in frames) / total_frames if total_frames else 0.0
    )
    duration_ms = (
        (len(mono_samples) / wav_audio.sample_rate_hz) * 1000 if wav_audio.sample_rate_hz else 0.0
    )

    return AudioAnalysis(
        source_label=source_label,
        duration_ms=duration_ms,
        sample_rate_hz=wav_audio.sample_rate_hz,
        original_channels=wav_audio.channels,
        mono_samples=mono_samples,
        normalized_wav_bytes=normalized_wav_bytes,
        frames=frames,
        segments=segments,
        frame_rows=tuple(frame_rows),
        segment_rows=tuple(segment_rows),
        silent_frames=sum(1 for row in frame_rows if row["is_silent"]),
        clipped_frames=sum(1 for row in frame_rows if row["is_clipped"]),
        average_rms_level=average_rms_level,
        max_peak_level=max((frame.peak_level for frame in frames), default=0.0),
    )


@st.cache_resource(show_spinner=False)
def get_transcriber(
    model_name: str,
    language: str,
    prompt: str,
    device: str,
    fp16: bool,
    beam_size: int,
    word_timestamps: bool,
) -> WhisperTranscriber:
    """Cache the Whisper model between Streamlit reruns."""

    config = WhisperConfig(
        model_name=model_name,
        language=language,
        prompt=prompt,
        min_segment_duration_ms=0,
        device=device,
        fp16=fp16,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        condition_on_previous_text=False,
    )
    return WhisperTranscriber(backend=OpenAIWhisperBackend(config=config), config=config)


@st.cache_data(show_spinner=False)
def get_input_device_rows() -> tuple[dict[str, int | float | str], ...]:
    """Read available server-side input devices exposed by sounddevice."""

    return tuple(
        {
            "index": device.index,
            "name": device.name,
            "channels": device.channels,
            "default_sample_rate_hz": round(device.default_sample_rate_hz, 1),
        }
        for device in list_input_devices()
    )


def run_live_transcription_session(
    *,
    device_name: str,
    duration_seconds: float,
    sample_rate_hz: int,
    frame_duration_ms: int,
    silence_threshold: float,
    clipping_threshold: float,
    speech_threshold: float,
    noise_floor: float,
    min_speech_frames: int,
    min_silence_frames: int,
    pre_roll_frames: int,
    transcriber: WhisperTranscriber,
    on_frame: Callable[[int, int, AudioFrame], None] | None = None,
    on_segment: Callable[[SpeechSegment, SegmentTranscription | None], None] | None = None,
) -> LiveTranscriptionSession:
    """Capture one live session and transcribe speech segments incrementally."""

    frame_budget = max(1, int(round((duration_seconds * 1000) / frame_duration_ms)))
    capture_config = MicrophoneConfig(
        device_name=device_name,
        sample_rate_hz=sample_rate_hz,
        channels=1,
        frame_duration_ms=frame_duration_ms,
        silence_threshold=silence_threshold,
        clipping_threshold=clipping_threshold,
    )
    vad_config = VoiceActivityConfig(
        speech_threshold=speech_threshold,
        noise_floor=noise_floor,
        min_speech_frames=min_speech_frames,
        min_silence_frames=min_silence_frames,
        pre_roll_frames=pre_roll_frames,
    )
    microphone = RealTimeMicrophone(device=SoundDeviceInput(), config=capture_config)
    segmenter = StreamingSpeechSegmenter(config=vad_config)
    vad = EnergyVAD(vad_config)

    captured_frames: list[AudioFrame] = []
    emitted_segments: list[SpeechSegment] = []
    transcriptions: list[SegmentTranscription] = []

    try:
        for frame_index in range(frame_budget):
            frame = microphone.capture_frame()
            captured_frames.append(frame)
            if on_frame is not None:
                on_frame(frame_index + 1, frame_budget, frame)

            segment = segmenter.push_frame(frame)
            if segment is None:
                continue

            emitted_segments.append(segment)
            transcription = transcriber.transcribe_segment(segment)
            if transcription is not None:
                transcriptions.append(transcription)
            if on_segment is not None:
                on_segment(segment, transcription)

        trailing_segment = segmenter.flush()
        if trailing_segment is not None:
            emitted_segments.append(trailing_segment)
            trailing_transcription = transcriber.transcribe_segment(trailing_segment)
            if trailing_transcription is not None:
                transcriptions.append(trailing_transcription)
            if on_segment is not None:
                on_segment(trailing_segment, trailing_transcription)
    finally:
        microphone.stop()

    health_log = microphone.health_log()
    health_report = microphone.health_report()
    merged_samples = tuple(sample for frame in captured_frames for sample in frame.samples)
    session_wav_bytes = wav_bytes_from_samples(merged_samples, sample_rate_hz)

    frame_rows = []
    for frame, snapshot in zip(captured_frames, health_log, strict=False):
        decision = vad.classify(frame)
        frame_rows.append(
            {
                "frame_index": frame.frame_index,
                "started_at_ms": round(frame.started_at_ms, 1),
                "duration_ms": round(frame.duration_ms, 1),
                "rms_level": round(frame.rms_level, 4),
                "peak_level": round(frame.peak_level, 4),
                "threshold": round(decision.threshold, 4),
                "is_speech": decision.is_speech,
                "is_silent": snapshot.silent,
                "is_clipped": snapshot.clipped,
            }
        )

    segment_rows = []
    for index, segment in enumerate(emitted_segments, start=1):
        matching_transcription = next(
            (
                item
                for item in transcriptions
                if item.start_frame_index == segment.start_frame_index
                and item.end_frame_index == segment.end_frame_index
            ),
            None,
        )
        segment_start_ms = segment.frames[0].started_at_ms
        segment_rows.append(
            {
                "segment_index": index,
                "start_frame_index": segment.start_frame_index,
                "end_frame_index": segment.end_frame_index,
                "start_ms": round(segment_start_ms, 1),
                "end_ms": round(segment_start_ms + segment.duration_ms, 1),
                "duration_ms": round(segment.duration_ms, 1),
                "text": matching_transcription.text if matching_transcription is not None else None,
                "confidence": (
                    matching_transcription.confidence if matching_transcription is not None else None
                ),
            }
        )

    return LiveTranscriptionSession(
        device_name=device_name,
        duration_ms=sum(frame.duration_ms for frame in captured_frames),
        audio_wav_bytes=session_wav_bytes,
        frame_rows=tuple(frame_rows),
        segment_rows=tuple(segment_rows),
        transcriptions=tuple(transcriptions),
        silent_frames=health_report.silent_frames,
        clipped_frames=health_report.clipped_frames,
        average_rms_level=health_report.average_rms_level,
    )


def render_runtime_panel() -> None:
    """Display hardware/runtime diagnostics for the audio stack."""

    st.subheader("Runtime")
    st.write(
        "Cette section expose le runtime du package `memento.audio`. "
        "Le microphone du navigateur et les peripheriques `sounddevice` ne passent pas par le meme chemin."
    )

    diagnostic_rows = (
        {"check": "torch.cuda.is_available()", "value": str(torch_cuda_available())},
        {"check": "streamlit audio_input", "value": "Actif si le navigateur autorise le micro"},
        {"check": "list_input_devices()", "value": "Interroge le serveur Streamlit"},
    )
    st.dataframe(diagnostic_rows, use_container_width=True, hide_index=True)

    try:
        device_rows = get_input_device_rows()
    except Exception as error:  # pragma: no cover - runtime diagnostics only
        st.warning(f"Impossible de lister les micros `sounddevice`: {error}")
        return

    if not device_rows:
        st.info("Aucun micro `sounddevice` detecte sur la machine qui execute Streamlit.")
        return

    st.write("Micros exposes cote serveur:")
    st.dataframe(device_rows, use_container_width=True, hide_index=True)


def render_transcription_result(title: str, transcription) -> None:
    """Render one Whisper transcription block."""

    st.markdown(f"### {title}")
    if transcription is None or not transcription.text:
        st.warning("Aucun texte detecte pour cette selection.")
        return

    st.write(transcription.text)
    st.caption(
        f"Latence {transcription.latency_ms:.0f} ms | "
        f"Duree {transcription.duration_ms:.0f} ms | "
        f"Modele {transcription.model_name} | "
        f"Confiance {transcription.confidence if transcription.confidence is not None else 'n/a'}"
    )

    if transcription.words:
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


def render_live_session(session: LiveTranscriptionSession) -> None:
    """Render the result of one live transcription session."""

    st.subheader("Derniere session live")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Micro", session.device_name)
    metric_columns[1].metric("Duree", f"{session.duration_ms / 1000:.2f} s")
    metric_columns[2].metric("Segments", len(session.segment_rows))
    metric_columns[3].metric("Average RMS", f"{session.average_rms_level:.4f}")

    capture_columns = st.columns(3)
    capture_columns[0].metric("Frames", len(session.frame_rows))
    capture_columns[1].metric("Silent frames", session.silent_frames)
    capture_columns[2].metric("Clipped frames", session.clipped_frames)

    st.audio(session.audio_wav_bytes)
    st.download_button(
        label="Telecharger la capture live",
        data=session.audio_wav_bytes,
        file_name="memento_live_capture.wav",
        mime="audio/wav",
    )

    if session.segment_rows:
        st.write("Segments transcrits:")
        st.dataframe(session.segment_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Aucun segment de parole n'a ete detecte pendant la session.")

    if session.transcriptions:
        for index, transcription in enumerate(session.transcriptions, start=1):
            render_transcription_result(f"Segment live {index}", transcription)

    with st.expander("Details frame par frame"):
        st.dataframe(session.frame_rows, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit audio lab."""

    st.set_page_config(page_title="Memento Audio Lab", layout="wide")
    st.title("Memento Audio Lab")
    st.caption(
        "Front de test pour le package `memento.audio`: import WAV, micro navigateur, "
        "diagnostics de capture, VAD, transcription Whisper et live transcription."
    )

    with st.sidebar:
        st.header("Capture preview")
        frame_duration_ms = st.slider("Frame duration (ms)", min_value=10, max_value=200, value=30)
        silence_threshold = st.slider(
            "Silence threshold",
            min_value=0.0,
            max_value=0.2,
            value=0.01,
            step=0.005,
        )
        clipping_threshold = st.slider(
            "Clipping threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.98,
            step=0.01,
        )

        st.header("VAD")
        speech_threshold = st.slider(
            "Speech threshold",
            min_value=0.01,
            max_value=0.3,
            value=0.04,
            step=0.005,
        )
        noise_floor = st.slider(
            "Noise floor",
            min_value=0.0,
            max_value=0.1,
            value=0.008,
            step=0.001,
        )
        min_speech_frames = st.slider("Min speech frames", min_value=1, max_value=10, value=2)
        min_silence_frames = st.slider("Min silence frames", min_value=1, max_value=10, value=2)
        pre_roll_frames = st.slider("Pre-roll frames", min_value=0, max_value=10, value=1)

        st.header("Whisper")
        model_name = st.selectbox(
            "Modele",
            options=("tiny", "base", "small", "medium", "large-v3", "turbo"),
            index=4,
        )
        language = st.text_input("Langue", value="fr")
        prompt = st.text_area("Prompt initial", value="", height=80)
        device = st.selectbox("Device", options=("cpu", "cuda"), index=0)
        fp16 = st.checkbox("Activer fp16", value=False, disabled=device != "cuda")
        beam_size = st.slider("Beam size", min_value=1, max_value=8, value=5)
        word_timestamps = st.checkbox("Word timestamps", value=True)

    source_tab, vad_tab, transcription_tab, live_tab, runtime_tab = st.tabs(
        ["Source", "VAD", "Transcription", "Live", "Runtime"]
    )

    with source_tab:
        st.subheader("Source audio")
        st.write("Charge un fichier WAV PCM 16 bits ou enregistre un court message depuis le navigateur.")

        uploaded_audio = st.file_uploader("Fichier WAV", type=["wav"])
        recorded_audio = st.audio_input("Micro navigateur", sample_rate=16_000)

        active_audio = None
        source_label = ""
        if uploaded_audio is not None:
            active_audio = uploaded_audio.getvalue()
            source_label = f"upload:{uploaded_audio.name}"
        elif recorded_audio is not None:
            active_audio = recorded_audio.getvalue()
            source_label = "browser-microphone"

        if active_audio is None:
            st.info("Charge un fichier WAV ou enregistre un message pour lancer l'analyse.")
            with runtime_tab:
                render_runtime_panel()
            return

        st.audio(active_audio)

        try:
            analysis = analyze_audio(
                audio_bytes=active_audio,
                source_label=source_label,
                frame_duration_ms=frame_duration_ms,
                silence_threshold=silence_threshold,
                clipping_threshold=clipping_threshold,
                speech_threshold=speech_threshold,
                noise_floor=noise_floor,
                min_speech_frames=min_speech_frames,
                min_silence_frames=min_silence_frames,
                pre_roll_frames=pre_roll_frames,
            )
        except Exception as error:
            st.error(f"Analyse impossible: {error}")
            with runtime_tab:
                render_runtime_panel()
            return

        metric_columns = st.columns(4)
        metric_columns[0].metric("Duree", f"{analysis.duration_ms / 1000:.2f} s")
        metric_columns[1].metric("Sample rate", f"{analysis.sample_rate_hz} Hz")
        metric_columns[2].metric("Canaux origine", analysis.original_channels)
        metric_columns[3].metric("Frames", len(analysis.frames))

        capture_columns = st.columns(4)
        capture_columns[0].metric("Segments VAD", len(analysis.segments))
        capture_columns[1].metric("Silent frames", analysis.silent_frames)
        capture_columns[2].metric("Clipped frames", analysis.clipped_frames)
        capture_columns[3].metric("Average RMS", f"{analysis.average_rms_level:.4f}")

        st.download_button(
            label="Telecharger le WAV mono normalise",
            data=analysis.normalized_wav_bytes,
            file_name="memento_audio_lab_normalized.wav",
            mime="audio/wav",
        )

        if analysis.original_channels > 1:
            st.info(
                "Le fichier a ete converti en mono pour uniformiser l'analyse VAD et la transcription."
            )

    with vad_tab:
        if "analysis" not in locals():
            st.info("Ajoute d'abord une source audio.")
        else:
            st.subheader("Frames et segmentation")
            st.line_chart(
                analysis.frame_rows,
                x="frame_index",
                y=["rms_level", "threshold", "peak_level"],
                use_container_width=True,
            )

            if analysis.segment_rows:
                st.write("Segments detectes:")
                st.dataframe(analysis.segment_rows, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun segment de parole detecte avec la configuration VAD courante.")

            st.write("Detail frame par frame:")
            st.dataframe(analysis.frame_rows, use_container_width=True, hide_index=True)

    with transcription_tab:
        if "analysis" not in locals():
            st.info("Ajoute d'abord une source audio.")
        else:
            mode = st.radio(
                "Mode de transcription",
                options=("Clip complet", "Segments VAD"),
                horizontal=True,
            )

            if st.button("Transcrire", type="primary"):
                try:
                    transcriber = get_transcriber(
                        model_name=model_name,
                        language=language,
                        prompt=prompt,
                        device=device,
                        fp16=fp16,
                        beam_size=beam_size,
                        word_timestamps=word_timestamps,
                    )
                except Exception as error:
                    st.error(f"Chargement Whisper impossible: {error}")
                else:
                    try:
                        with st.spinner("Transcription en cours..."):
                            if mode == "Clip complet":
                                full_segment = speech_segment_from_wav_bytes(
                                    analysis.normalized_wav_bytes
                                )
                                transcription = transcriber.transcribe_segment(full_segment)
                                render_transcription_result("Clip complet", transcription)
                            else:
                                if not analysis.segments:
                                    st.warning("Aucun segment VAD disponible pour la transcription.")
                                for index, segment in enumerate(analysis.segments, start=1):
                                    transcription = transcriber.transcribe_segment(segment)
                                    render_transcription_result(
                                        f"Segment {index} ({segment.duration_ms:.0f} ms)",
                                        transcription,
                                    )
                    except Exception as error:
                        st.error(f"Transcription impossible: {error}")

    with live_tab:
        st.subheader("Live transcription")
        st.write(
            "Cette session utilise le micro `sounddevice` de la machine qui execute Streamlit, "
            "pas le micro du navigateur."
        )

        try:
            device_rows = get_input_device_rows()
        except Exception as error:
            st.error(f"Impossible de preparer la capture live: {error}")
        else:
            device_labels = {"default (system)": "default"}
            for row in device_rows:
                label = f"{row['name']} [{row['index']}]"
                device_labels[label] = str(row["name"])

            selected_device_label = st.selectbox(
                "Micro live",
                options=tuple(device_labels.keys()),
                index=0,
            )
            live_duration_seconds = st.slider(
                "Duree de capture (s)",
                min_value=2,
                max_value=20,
                value=6,
            )
            live_sample_rate_hz = st.selectbox(
                "Sample rate live",
                options=(16_000, 44_100, 48_000),
                index=0,
            )

            st.caption(
                "Le traitement est synchrone: pendant la capture, Streamlit bloque l'UI puis affiche les segments detectes."
            )

            if st.button("Lancer la live transcription", type="primary"):
                try:
                    transcriber = get_transcriber(
                        model_name=model_name,
                        language=language,
                        prompt=prompt,
                        device=device,
                        fp16=fp16,
                        beam_size=beam_size,
                        word_timestamps=word_timestamps,
                    )
                except Exception as error:
                    st.error(f"Chargement Whisper impossible: {error}")
                else:
                    progress_bar = st.progress(0.0)
                    status_placeholder = st.empty()
                    transcript_placeholder = st.empty()
                    streamed_transcripts: list[str] = []

                    def on_frame(processed_frames: int, total_frames: int, frame: AudioFrame) -> None:
                        progress_bar.progress(
                            processed_frames / total_frames,
                            text=(
                                f"Capture frame {processed_frames}/{total_frames} | "
                                f"rms={frame.rms_level:.4f} peak={frame.peak_level:.4f}"
                            ),
                        )
                        status_placeholder.caption(
                            f"Ecoute en cours sur `{device_labels[selected_device_label]}`..."
                        )

                    def on_segment(
                        segment: SpeechSegment, transcription: SegmentTranscription | None
                    ) -> None:
                        if transcription is None or not transcription.text:
                            streamed_transcripts.append(
                                f"- frames {segment.start_frame_index}-{segment.end_frame_index}: aucun texte"
                            )
                        else:
                            streamed_transcripts.append(
                                f"- frames {segment.start_frame_index}-{segment.end_frame_index}: {transcription.text}"
                            )
                        transcript_placeholder.markdown(
                            "Segments detectes pendant la capture:\n"
                            + "\n".join(streamed_transcripts)
                        )

                    try:
                        live_session = run_live_transcription_session(
                            device_name=device_labels[selected_device_label],
                            duration_seconds=float(live_duration_seconds),
                            sample_rate_hz=int(live_sample_rate_hz),
                            frame_duration_ms=frame_duration_ms,
                            silence_threshold=silence_threshold,
                            clipping_threshold=clipping_threshold,
                            speech_threshold=speech_threshold,
                            noise_floor=noise_floor,
                            min_speech_frames=min_speech_frames,
                            min_silence_frames=min_silence_frames,
                            pre_roll_frames=pre_roll_frames,
                            transcriber=transcriber,
                            on_frame=on_frame,
                            on_segment=on_segment,
                        )
                    except Exception as error:
                        st.error(f"Live transcription impossible: {error}")
                    else:
                        st.session_state["live_session_result"] = live_session
                        progress_bar.progress(1.0, text="Capture terminee")
                        status_placeholder.success("Session live terminee.")

            existing_live_session = st.session_state.get("live_session_result")
            if isinstance(existing_live_session, LiveTranscriptionSession):
                render_live_session(existing_live_session)

    with runtime_tab:
        render_runtime_panel()


if __name__ == "__main__":
    main()
