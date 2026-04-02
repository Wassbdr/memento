"""Interactive Streamlit lab for testing the Memento TTS stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from memento.audio import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    DecodedAudio,
    QwenTTSBackend,
    SoundDeviceOutput,
    SpeakerConfig,
    SpeakerPlayer,
    SpeechSynthesizer,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SynthesizedSpeech,
    TextToSpeechConfig,
    VoiceExperienceTargets,
    VoiceResponsePipeline,
    VoiceResponseResult,
    decode_audio_bytes,
    write_wav_file,
)


def default_device_map() -> str:
    """Prefer an explicit device target over `auto` for Qwen local loading."""

    try:
        torch = import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TTSPreview:
    """Serializable snapshot kept in Streamlit session state."""

    synthesis: SynthesizedSpeech
    decoded: DecodedAudio
    browser_audio_bytes: bytes
    browser_audio_format: str
    language: str | None
    instruction: str | None


def wav_bytes_from_samples(
    samples: tuple[float, ...],
    sample_rate_hz: int,
    channels: int = 1,
) -> bytes:
    """Encode normalized samples as a WAV payload for browser preview."""

    with NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        temp_path = Path(temporary_file.name)

    try:
        write_wav_file(
            path=temp_path,
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def build_browser_preview(synthesis: SynthesizedSpeech) -> tuple[DecodedAudio, bytes, str]:
    """Decode a TTS response and normalize it for Streamlit playback."""

    decoded = decode_audio_bytes(
        audio_bytes=synthesis.audio_bytes,
        response_format=synthesis.response_format,
        sample_rate_hz=synthesis.sample_rate_hz,
        channels=synthesis.channels,
    )
    browser_audio_bytes = wav_bytes_from_samples(
        samples=decoded.samples,
        sample_rate_hz=decoded.sample_rate_hz,
        channels=decoded.channels,
    )
    return decoded, browser_audio_bytes, "audio/wav"


def build_tts_config(
    *,
    model_name: str,
    voice_id: str,
    response_format: str,
    sample_rate_hz: int | None,
    channels: int,
    language: str,
    instruction: str,
    device_map: str,
    dtype: str,
    attn_implementation: str,
) -> TextToSpeechConfig:
    """Create a normalized config from UI inputs."""

    return TextToSpeechConfig(
        model_name=model_name,
        voice_id=voice_id.strip() or None,
        response_format=response_format,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        language=language.strip() or None,
        instruction=instruction.strip() or None,
        device_map=device_map.strip() or None,
        dtype=dtype.strip() or None,
        attn_implementation=attn_implementation.strip() or None,
    )


def synthesize_preview(
    *,
    text: str,
    config: TextToSpeechConfig,
    voice_id_override: str,
) -> TTSPreview:
    """Run one synthesis and prepare a browser-friendly preview."""

    synthesizer = SpeechSynthesizer(
        backend=QwenTTSBackend(config=config),
        config=config,
    )
    synthesis = synthesizer.synthesize(
        text,
        voice_id=voice_id_override.strip() or None,
    )
    decoded, browser_audio_bytes, browser_audio_format = build_browser_preview(synthesis)
    return TTSPreview(
        synthesis=synthesis,
        decoded=decoded,
        browser_audio_bytes=browser_audio_bytes,
        browser_audio_format=browser_audio_format,
        language=config.language,
        instruction=config.instruction,
    )


def speak_on_server(
    *,
    text: str,
    config: TextToSpeechConfig,
    voice_id_override: str,
    playback_device_name: str,
    blocking: bool,
    interrupt_current: bool,
    targets: VoiceExperienceTargets,
) -> tuple[VoiceResponseResult, TTSPreview]:
    """Synthesize and dispatch playback on the Streamlit server."""

    player = SpeakerPlayer(
        device=SoundDeviceOutput(),
        config=SpeakerConfig(
            device_name=playback_device_name.strip() or "default",
            blocking=blocking,
            interrupt_current=interrupt_current,
        ),
    )
    pipeline = VoiceResponsePipeline(
        synthesizer=SpeechSynthesizer(
            backend=QwenTTSBackend(config=config),
            config=config,
        ),
        player=player,
        targets=targets,
    )
    result = pipeline.speak(
        text,
        voice_id=voice_id_override.strip() or None,
    )
    decoded, browser_audio_bytes, browser_audio_format = build_browser_preview(result.synthesis)
    preview = TTSPreview(
        synthesis=result.synthesis,
        decoded=decoded,
        browser_audio_bytes=browser_audio_bytes,
        browser_audio_format=browser_audio_format,
        language=config.language,
        instruction=config.instruction,
    )
    return result, preview


@st.cache_data(show_spinner=False)
def get_output_device_rows() -> tuple[dict[str, int | float | str], ...]:
    """List server-side output devices exposed by sounddevice."""

    sounddevice = import_module("sounddevice")
    rows = []
    for index, device in enumerate(sounddevice.query_devices()):
        max_output_channels = int(device.get("max_output_channels", 0))
        if max_output_channels <= 0:
            continue
        rows.append(
            {
                "index": index,
                "name": str(device.get("name", f"device-{index}")),
                "max_output_channels": max_output_channels,
                "default_samplerate_hz": round(float(device.get("default_samplerate", 0.0)), 1),
            }
        )
    return tuple(rows)


def render_preview(preview: TTSPreview) -> None:
    """Render the last synthesized utterance and its metrics."""

    st.subheader("Derniere synthese")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Format", preview.synthesis.response_format)
    metric_columns[1].metric("Latence synthese", f"{preview.synthesis.latency_ms:.0f} ms")
    metric_columns[2].metric("Sample rate", f"{preview.decoded.sample_rate_hz} Hz")
    metric_columns[3].metric("Duree audio", f"{preview.decoded.duration_ms:.0f} ms")

    detail_columns = st.columns(4)
    detail_columns[0].metric("Canaux", preview.decoded.channels)
    detail_columns[1].metric("Caracteres", preview.synthesis.character_count)
    detail_columns[2].metric("Modele", preview.synthesis.model_name)
    detail_columns[3].metric("Speaker", preview.synthesis.voice_id or "n/a")

    if preview.language is not None:
        st.caption(f"Langue: `{preview.language}`")
    if preview.instruction is not None:
        st.caption(f"Instruction: `{preview.instruction}`")

    st.audio(preview.browser_audio_bytes, format=preview.browser_audio_format)
    st.download_button(
        label="Telecharger le rendu navigateur (WAV)",
        data=preview.browser_audio_bytes,
        file_name="memento_tts_preview.wav",
        mime="audio/wav",
    )

    with st.expander("Metadonnees"):
        st.json(
            {
                "synthesis": asdict(preview.synthesis),
                "decoded": {
                    "sample_rate_hz": preview.decoded.sample_rate_hz,
                    "channels": preview.decoded.channels,
                    "duration_ms": preview.decoded.duration_ms,
                    "sample_count": len(preview.decoded.samples),
                },
                "language": preview.language,
                "instruction": preview.instruction,
            }
        )


def render_voice_result(result: VoiceResponseResult) -> None:
    """Render end-to-end metrics when server playback is used."""

    st.subheader("Derniere lecture serveur")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Dispatch",
        f"{result.playback.dispatch_latency_ms:.1f} ms"
        if result.playback.dispatch_latency_ms is not None
        else "n/a",
    )
    metric_columns[1].metric(
        "Completion",
        f"{result.playback.completion_latency_ms:.1f} ms"
        if result.playback.completion_latency_ms is not None
        else "n/a",
    )
    metric_columns[2].metric(
        "End-to-end",
        f"{result.metrics.end_to_end_latency_ms:.1f} ms"
        if result.metrics.end_to_end_latency_ms is not None
        else "n/a",
    )
    metric_columns[3].metric(
        "RTF",
        f"{result.metrics.realtime_factor:.2f}"
        if result.metrics.realtime_factor is not None
        else "n/a",
    )
    metric_columns[4].metric("Targets", "OK" if result.meets_targets else "KO")

    st.caption(
        "La lecture `server speaker` joue sur la machine qui execute Streamlit, pas dans le navigateur."
    )
    with st.expander("Details playback"):
        st.json(
            {
                "playback": asdict(result.playback),
                "metrics": asdict(result.metrics),
                "targets": asdict(result.targets),
                "meets_targets": result.meets_targets,
            }
        )


def render_runtime_panel(model_name: str) -> None:
    """Display diagnostics useful for TTS experimentation."""

    st.subheader("Runtime")
    st.write(
        "Le preview audio joue dans le navigateur. "
        "Le bouton `server speaker` joue sur les haut-parleurs de la machine qui execute Streamlit."
    )

    diagnostic_rows = (
        {"check": "Backend", "value": "Qwen local via `qwen-tts`"},
        {"check": "Modele local", "value": model_name},
        {"check": "Package", "value": "`qwen_tts.Qwen3TTSModel.from_pretrained(...)`"},
        {"check": "Supported response formats", "value": ", ".join(SUPPORTED_TTS_RESPONSE_FORMATS)},
        {"check": "Preview browser", "value": "Toujours converti en WAV pour Streamlit"},
    )
    st.dataframe(diagnostic_rows, use_container_width=True, hide_index=True)

    try:
        output_devices = get_output_device_rows()
    except Exception as error:  # pragma: no cover - runtime diagnostics only
        st.warning(f"Impossible de lister les sorties audio `sounddevice`: {error}")
        return

    if not output_devices:
        st.info("Aucune sortie audio `sounddevice` detectee cote serveur.")
        return

    st.write("Sorties audio exposees cote serveur:")
    st.dataframe(output_devices, use_container_width=True, hide_index=True)


def main() -> None:
    """Run the Streamlit TTS lab."""

    st.set_page_config(page_title="Memento TTS Lab", layout="wide")
    st.title("Memento TTS Lab")
    st.caption(
        "Front de test local pour Qwen3-TTS CustomVoice: texte, speaker, langue, instruction "
        "et lecture optionnelle sur le haut-parleur du serveur."
    )

    with st.sidebar:
        st.header("Modele local")
        model_name = st.text_input("Modele", value=DEFAULT_QWEN_TTS_MODEL_NAME)
        configured_voice_id = st.text_input("Speaker par defaut", value=DEFAULT_QWEN_TTS_SPEAKER)
        voice_id_override = st.text_input("Speaker pour cet essai", value="")
        language = st.text_input("Langue", value=DEFAULT_QWEN_TTS_LANGUAGE)
        instruction = st.text_area("Instruction de style", value="", height=100)
        response_format = st.selectbox(
            "Response format",
            options=SUPPORTED_TTS_RESPONSE_FORMATS,
            index=1,
        )
        sample_rate_hz = st.selectbox(
            "Sample rate attendu",
            options=(None, 16_000, 24_000, 44_100, 48_000),
            index=2,
            format_func=lambda value: "auto" if value is None else f"{value} Hz",
        )
        device_map = st.text_input("device_map", value=default_device_map())
        dtype = st.text_input("dtype torch", value="")
        attn_implementation = st.text_input("attn_implementation", value="")

        st.header("Playback serveur")
        playback_device_name = st.text_input("Device name", value="default")
        blocking = st.checkbox("Playback blocking", value=False)
        interrupt_current = st.checkbox("Interrompre la lecture courante", value=True)

        st.header("Targets")
        max_synthesis_latency_ms = st.slider(
            "Max synthesis latency",
            min_value=100,
            max_value=30_000,
            value=10_000,
            step=100,
        )
        max_playback_dispatch_latency_ms = st.slider(
            "Max dispatch latency",
            min_value=10,
            max_value=2_000,
            value=200,
            step=10,
        )
        max_end_to_end_latency_ms = st.slider(
            "Max end-to-end latency",
            min_value=100,
            max_value=30_000,
            value=12_000,
            step=100,
        )
        max_realtime_factor = st.slider(
            "Max realtime factor",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )

    synthesis_tab, playback_tab, runtime_tab = st.tabs(
        ["Synthese", "Playback serveur", "Runtime"]
    )

    with synthesis_tab:
        st.subheader("Prompt")
        st.info(
            "Le front charge `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` localement via le package "
            "`qwen-tts`. D'apres la doc officielle, le mode CustomVoice utilise "
            "`generate_custom_voice(text, language, speaker, instruct)`. "
            "Si tu vois une erreur sur des `meta tensors`, utilise `cpu` ou `cuda:0` plutot que `auto`."
        )
        text = st.text_area(
            "Texte a prononcer",
            value="Bonjour Charles, je suis la pour t'aider a retrouver tes reperes aujourd'hui.",
            height=140,
        )

        config_error: str | None = None
        try:
            config = build_tts_config(
                model_name=model_name,
                voice_id=configured_voice_id,
                response_format=response_format,
                sample_rate_hz=sample_rate_hz,
                channels=1,
                language=language,
                instruction=instruction,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
            targets = VoiceExperienceTargets(
                max_synthesis_latency_ms=float(max_synthesis_latency_ms),
                max_playback_dispatch_latency_ms=float(max_playback_dispatch_latency_ms),
                max_end_to_end_latency_ms=float(max_end_to_end_latency_ms),
                max_realtime_factor=float(max_realtime_factor),
            )
        except ValueError as error:
            config = None
            targets = None
            config_error = str(error)

        if config_error is not None:
            st.error(f"Configuration TTS invalide: {config_error}")

        action_columns = st.columns(3)
        synthesize_clicked = action_columns[0].button(
            "Synthese seulement",
            type="primary",
            disabled=config is None,
        )
        speak_clicked = action_columns[1].button(
            "Synthese + server speaker",
            disabled=config is None,
        )
        clear_clicked = action_columns[2].button("Effacer les resultats")

        if clear_clicked:
            st.session_state.pop("tts_preview", None)
            st.session_state.pop("tts_voice_result", None)

        if synthesize_clicked:
            try:
                with st.spinner("Synthese TTS en cours..."):
                    preview = synthesize_preview(
                        text=text,
                        config=config,
                        voice_id_override=voice_id_override,
                    )
            except Exception as error:
                st.error(f"Synthese impossible: {error}")
            else:
                st.session_state["tts_preview"] = preview
                st.session_state.pop("tts_voice_result", None)
                st.success("Synthese terminee. Le rendu navigateur est pret.")

        if speak_clicked:
            try:
                with st.spinner("Synthese + lecture serveur en cours..."):
                    voice_result, preview = speak_on_server(
                        text=text,
                        config=config,
                        voice_id_override=voice_id_override,
                        playback_device_name=playback_device_name,
                        blocking=blocking,
                        interrupt_current=interrupt_current,
                        targets=targets,
                    )
            except Exception as error:
                st.error(f"Lecture serveur impossible: {error}")
            else:
                st.session_state["tts_preview"] = preview
                st.session_state["tts_voice_result"] = voice_result
                st.success("Lecture serveur declenchee.")

        existing_preview = st.session_state.get("tts_preview")
        if isinstance(existing_preview, TTSPreview):
            render_preview(existing_preview)
        else:
            st.info("Lance une synthese pour afficher le preview audio et les metriques.")

    with playback_tab:
        existing_voice_result = st.session_state.get("tts_voice_result")
        if isinstance(existing_voice_result, VoiceResponseResult):
            render_voice_result(existing_voice_result)
        else:
            st.info(
                "Utilise `Synthese + server speaker` pour mesurer la lecture locale et les latences end-to-end."
            )

    with runtime_tab:
        render_runtime_panel(model_name=model_name)


if __name__ == "__main__":
    main()
