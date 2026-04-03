"""Immersive Streamlit front-end for the Memento runtime."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from memento.audio import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    OpenAIWhisperBackend,
    QwenTTSBackend,
    SpeechSynthesizer,
    TextToSpeechConfig,
    WhisperConfig,
    WhisperTranscriber,
    decode_audio_bytes,
    speech_segment_from_wav_bytes,
    torch_cuda_available,
    write_wav_file,
)
from memento.conversation import (
    ConversationConfig,
    ConversationMessage,
    ConversationOrchestrator,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)
from memento.memory import MemorySyncEngine, PatientMemorySnapshot
from memento.runtime.bootstrap import snapshot_from_dict


DEFAULT_SNAPSHOT_JSON = json.dumps(
    {
        "patient": {
            "patient_id": "rose",
            "display_name": "Rose Martin",
            "preferred_name": "Mamie Rose",
            "care_notes": [
                "Rassurer avant de recontextualiser.",
                "Parler lentement avec des phrases courtes.",
            ],
            "anchors": [
                "Appartement rue des Lilas",
                "Claire vient souvent le dimanche",
            ],
        },
        "people": [
            {
                "person_id": "claire",
                "name": "Claire Martin",
                "relationship_to_patient": "sa fille",
                "notes": "Passe le dimanche pour le dejeuner.",
                "emotional_significance": 0.98,
            },
            {
                "person_id": "lucas",
                "name": "Lucas",
                "relationship_to_patient": "son petit-fils",
                "notes": "Appelle en video le mercredi soir.",
                "emotional_significance": 0.86,
            },
        ],
        "places": [
            {
                "place_id": "home",
                "name": "Appartement rue des Lilas",
                "category": "domicile",
                "notes": "Salon lumineux avec les photos de famille sur la commode.",
            }
        ],
        "routines": [
            {
                "routine_id": "lunch_sunday",
                "title": "Dejeuner du dimanche",
                "schedule": "dimanche midi",
                "description": "Claire vient partager le repas avec Rose.",
                "cue": "Mettre la nappe claire sur la table.",
                "support_strategy": "Rappeler que Claire arrive apres la matinee.",
                "place_id": "home",
            }
        ],
        "episodes": [
            {
                "episode_id": "ep_family_lunch",
                "title": "Repas de famille",
                "narrative": "Rose aime les dejeuners calmes avec Claire dans le salon.",
                "happened_on": "2026-03-30",
                "people_ids": ["claire"],
                "place_id": "home",
                "emotions": [
                    {
                        "label": "apaisement",
                        "valence": 0.9,
                        "intensity": 0.7,
                        "notes": "La presence de Claire rassure Rose.",
                    }
                ],
                "tags": ["famille", "repere", "dimanche"],
            }
        ],
    },
    indent=2,
    ensure_ascii=True,
)


@dataclass(frozen=True)
class RuntimeFrontendContext:
    """Cached runtime context used across Streamlit reruns."""

    snapshot: PatientMemorySnapshot
    orchestrator: ConversationOrchestrator


@dataclass(frozen=True)
class RuntimeFrontendTurn:
    """Serializable summary of one UI turn."""

    source: str
    user_text: str
    assistant_text: str
    transcript_latency_ms: float | None
    generation_latency_ms: float | None
    synthesis_latency_ms: float | None
    audio_duration_ms: float | None
    browser_audio_bytes: bytes | None
    browser_audio_format: str | None
    retrieved_memories: tuple[dict[str, object], ...]
    guard_applied: bool
    guard_reason: str
    tts_error: str | None = None


def default_device_map() -> str:
    """Prefer an explicit device target over `auto` for local Qwen loading."""

    try:
        torch = import_module("torch")
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def apply_global_styles() -> None:
    """Inject the visual language for the orb interface."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
          --bg: #040507;
          --panel: rgba(10, 14, 22, 0.76);
          --panel-strong: rgba(14, 18, 28, 0.92);
          --line: rgba(132, 165, 255, 0.22);
          --text: #f4f7ff;
          --muted: #95a3bf;
          --blue: #4d7dff;
          --cyan: #62e8ff;
          --rose: #ff5b9a;
          --amber: #ffb86b;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
          background:
            radial-gradient(circle at 20% 20%, rgba(77, 125, 255, 0.18), transparent 30%),
            radial-gradient(circle at 80% 12%, rgba(255, 91, 154, 0.14), transparent 26%),
            radial-gradient(circle at 50% 80%, rgba(98, 232, 255, 0.10), transparent 28%),
            linear-gradient(180deg, #020305 0%, #05070a 50%, #020305 100%);
          color: var(--text);
          font-family: "IBM Plex Sans", sans-serif;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"] {
          display: none;
        }

        .block-container {
          max-width: 1180px;
          padding-top: 2rem;
          padding-bottom: 4rem;
        }

        .runtime-shell {
          position: relative;
          padding: 1.4rem;
          border: 1px solid var(--line);
          border-radius: 32px;
          background: linear-gradient(180deg, rgba(10, 14, 22, 0.84), rgba(6, 8, 13, 0.92));
          overflow: hidden;
          box-shadow:
            0 30px 80px rgba(0, 0, 0, 0.42),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .runtime-shell::before {
          content: "";
          position: absolute;
          inset: -20% 35% 45% -10%;
          background: radial-gradient(circle, rgba(77, 125, 255, 0.22), transparent 60%);
          filter: blur(26px);
          pointer-events: none;
        }

        .runtime-shell::after {
          content: "";
          position: absolute;
          inset: 55% -10% -25% 35%;
          background: radial-gradient(circle, rgba(255, 91, 154, 0.16), transparent 58%);
          filter: blur(34px);
          pointer-events: none;
        }

        .runtime-kicker {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.45rem 0.9rem;
          border: 1px solid rgba(98, 232, 255, 0.22);
          border-radius: 999px;
          background: rgba(7, 14, 21, 0.72);
          color: #dff7ff;
          font-size: 0.78rem;
          letter-spacing: 0.14em;
          text-transform: uppercase;
        }

        .runtime-title {
          margin: 1rem 0 0.35rem;
          font-family: "Space Grotesk", sans-serif;
          font-size: clamp(2.4rem, 5vw, 4.5rem);
          line-height: 0.95;
          letter-spacing: -0.05em;
        }

        .runtime-copy {
          max-width: 48rem;
          margin: 0;
          color: var(--muted);
          font-size: 1rem;
        }

        .orb-stage {
          display: flex;
          justify-content: center;
          padding: 2.2rem 0 1.8rem;
        }

        .orb {
          position: relative;
          width: min(58vw, 380px);
          aspect-ratio: 1;
          border-radius: 50%;
        }

        .orb::before {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: 50%;
          background:
            conic-gradient(
              from 0deg,
              rgba(77, 125, 255, 0.25),
              rgba(98, 232, 255, 0.9),
              rgba(255, 184, 107, 0.36),
              rgba(255, 91, 154, 0.92),
              rgba(77, 125, 255, 0.25)
            );
          filter: blur(18px);
          animation: orb-spin 10s linear infinite;
        }

        .orb::after {
          content: "";
          position: absolute;
          inset: 4%;
          border-radius: 50%;
          border: 1px solid rgba(255, 255, 255, 0.16);
          box-shadow:
            inset 0 0 30px rgba(255, 255, 255, 0.06),
            0 0 20px rgba(98, 232, 255, 0.18);
        }

        .orb-spectrum {
          position: absolute;
          inset: 0;
          border-radius: 50%;
          background:
            radial-gradient(circle, transparent 55%, rgba(0, 0, 0, 0) 58%),
            repeating-conic-gradient(
              from 0deg,
              rgba(255, 255, 255, 0.92) 0deg 1deg,
              transparent 1deg 6deg
            );
          mask:
            radial-gradient(circle, transparent 0 59%, #fff 60% 69%, transparent 70%);
          opacity: 0.65;
          animation: orb-pulse 3.2s ease-in-out infinite;
        }

        .orb-core {
          position: absolute;
          inset: 13%;
          border-radius: 50%;
          background:
            radial-gradient(circle at 34% 28%, rgba(255, 255, 255, 0.12), transparent 16%),
            radial-gradient(circle at 68% 40%, rgba(77, 125, 255, 0.12), transparent 24%),
            radial-gradient(circle at 50% 50%, rgba(15, 18, 28, 0.85), rgba(2, 3, 5, 0.98) 72%);
          box-shadow:
            inset 0 0 65px rgba(0, 0, 0, 0.75),
            inset 0 0 20px rgba(98, 232, 255, 0.08),
            0 0 60px rgba(77, 125, 255, 0.12);
        }

        .orb-label {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          text-align: center;
          z-index: 2;
        }

        .orb-label strong {
          font-family: "Space Grotesk", sans-serif;
          font-size: 1.25rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .orb-label span {
          color: var(--muted);
          font-size: 0.92rem;
          max-width: 13rem;
        }

        .status-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.85rem;
          margin-top: 1rem;
        }

        .status-card {
          padding: 0.9rem 1rem;
          border-radius: 20px;
          border: 1px solid rgba(132, 165, 255, 0.16);
          background: rgba(5, 8, 14, 0.55);
          backdrop-filter: blur(16px);
        }

        .status-label {
          color: var(--muted);
          font-size: 0.76rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
        }

        .status-value {
          margin-top: 0.3rem;
          font-family: "Space Grotesk", sans-serif;
          font-size: 1rem;
        }

        .message-card {
          border: 1px solid rgba(132, 165, 255, 0.14);
          border-radius: 22px;
          padding: 1rem 1rem 0.9rem;
          background: linear-gradient(180deg, rgba(10, 13, 20, 0.88), rgba(8, 11, 16, 0.72));
          margin-bottom: 0.9rem;
        }

        .message-label {
          display: inline-flex;
          padding: 0.32rem 0.72rem;
          border-radius: 999px;
          margin-bottom: 0.85rem;
          font-size: 0.75rem;
          font-weight: 600;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .message-label.user {
          color: #dff7ff;
          background: rgba(98, 232, 255, 0.12);
          border: 1px solid rgba(98, 232, 255, 0.2);
        }

        .message-label.assistant {
          color: #ffe9f2;
          background: rgba(255, 91, 154, 0.12);
          border: 1px solid rgba(255, 91, 154, 0.2);
        }

        .memory-chip {
          display: inline-flex;
          margin: 0.2rem 0.35rem 0.2rem 0;
          padding: 0.4rem 0.7rem;
          border-radius: 999px;
          background: rgba(77, 125, 255, 0.1);
          border: 1px solid rgba(77, 125, 255, 0.16);
          color: #dae6ff;
          font-size: 0.8rem;
        }

        div[data-testid="stAudioInput"],
        div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stTextAreaRootElement"] > div {
          border-radius: 20px;
        }

        div[data-testid="stAudioInput"] {
          padding: 0.8rem;
          background: rgba(7, 11, 18, 0.72);
          border: 1px solid rgba(132, 165, 255, 0.16);
        }

        div[data-testid="stTextInputRootElement"] input,
        div[data-testid="stTextAreaRootElement"] textarea {
          background: rgba(7, 11, 18, 0.85);
          color: var(--text);
          border: 1px solid rgba(132, 165, 255, 0.18);
          border-radius: 18px;
        }

        .stButton > button {
          border-radius: 999px;
          border: 1px solid rgba(98, 232, 255, 0.18);
          background: linear-gradient(135deg, rgba(77, 125, 255, 0.24), rgba(255, 91, 154, 0.18));
          color: var(--text);
          min-height: 2.9rem;
          font-family: "Space Grotesk", sans-serif;
        }

        @keyframes orb-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @keyframes orb-pulse {
          0%, 100% { transform: scale(0.98); opacity: 0.45; }
          50% { transform: scale(1.03); opacity: 0.85; }
        }

        @media (max-width: 900px) {
          .status-row {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Prepare mutable Streamlit state."""

    st.session_state.setdefault("runtime_turns", [])
    st.session_state.setdefault("runtime_history", [])
    st.session_state.setdefault("runtime_last_audio_hash", "")
    st.session_state.setdefault("runtime_status", "Pret a ecouter")
    st.session_state.setdefault("runtime_error", "")


def reset_conversation() -> None:
    """Clear transient turn state."""

    st.session_state["runtime_turns"] = []
    st.session_state["runtime_history"] = []
    st.session_state["runtime_last_audio_hash"] = ""
    st.session_state["runtime_status"] = "Pret a ecouter"
    st.session_state["runtime_error"] = ""


def history_messages() -> tuple[ConversationMessage, ...]:
    """Read normalized conversation history from session state."""

    return tuple(
        ConversationMessage(role=item["role"], content=item["content"])
        for item in st.session_state["runtime_history"]
    )


def append_history(user_text: str, assistant_text: str, max_history_messages: int) -> None:
    """Append one turn and keep the configured rolling window."""

    if max_history_messages == 0:
        st.session_state["runtime_history"] = []
        return

    state_history = st.session_state["runtime_history"]
    state_history.append({"role": "user", "content": user_text})
    state_history.append({"role": "assistant", "content": assistant_text})
    if len(state_history) > max_history_messages:
        st.session_state["runtime_history"] = state_history[-max_history_messages:]


def wav_bytes_from_samples(
    samples: tuple[float, ...],
    sample_rate_hz: int,
    channels: int = 1,
) -> bytes:
    """Encode normalized samples as a temporary WAV payload."""

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


def build_browser_preview(synthesis) -> tuple[bytes, str, float]:
    """Normalize synthesized speech for browser playback."""

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
    return browser_audio_bytes, "audio/wav", decoded.duration_ms


def render_hero(*, patient_name: str, status: str, llm_model: str, whisper_device: str) -> None:
    """Render the central orb scene."""

    st.markdown(
        f"""
        <div class="runtime-shell">
          <div class="runtime-kicker">Memento Runtime Interface</div>
          <h1 class="runtime-title">Une sphere calme pour parler a l'assistant.</h1>
          <p class="runtime-copy">
            Le micro du navigateur capture la question, Whisper transcrit, la memoire patient
            recentre la reponse, puis le TTS renvoie une voix lisible directement dans l'interface.
          </p>
          <div class="orb-stage">
            <div class="orb">
              <div class="orb-spectrum"></div>
              <div class="orb-core"></div>
              <div class="orb-label">
                <strong>{patient_name}</strong>
                <span>{status}</span>
              </div>
            </div>
          </div>
          <div class="status-row">
            <div class="status-card">
              <div class="status-label">Modele LLM</div>
              <div class="status-value">{llm_model}</div>
            </div>
            <div class="status-card">
              <div class="status-label">Ecoute</div>
              <div class="status-value">Micro navigateur</div>
            </div>
            <div class="status-card">
              <div class="status-label">Whisper</div>
              <div class="status-value">{whisper_device}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message_card(role: str, text: str) -> None:
    """Render one message card using the custom visual language."""

    label = "Vous" if role == "user" else "Assistant"
    st.markdown(
        f"""
        <div class="message-card">
          <div class="message-label {role}">{label}</div>
          <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_memory_chips(retrieved_memories: tuple[dict[str, object], ...]) -> None:
    """Render compact retrieval evidence."""

    if not retrieved_memories:
        st.caption("Aucun souvenir remonte pour ce tour.")
        return

    chips = []
    for memory in retrieved_memories:
        chips.append(
            f'<span class="memory-chip">{memory["source_label"]}: {memory["source_display_name"]}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def build_runtime_context(
    *,
    snapshot_json: str,
    llm_base_url: str,
    llm_api_key: str | None,
    llm_timeout_seconds: float,
    llm_model: str,
    temperature: float,
    top_k: int,
    max_prompt_memories: int,
) -> RuntimeFrontendContext:
    """Cache the memory+conversation stack between reruns."""

    payload = json.loads(snapshot_json)
    if not isinstance(payload, dict):
        raise ValueError("Le snapshot doit etre un objet JSON.")
    snapshot = snapshot_from_dict(payload)

    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(snapshot)
    orchestrator = ConversationOrchestrator(
        memory_engine=memory_engine,
        backend=OpenAICompatibleConversationBackend(
            config=OpenAICompatibleBackendConfig(
                base_url=llm_base_url,
                api_key=llm_api_key or None,
                timeout_seconds=llm_timeout_seconds,
            )
        ),
        config=ConversationConfig(
            model_name=llm_model,
            temperature=temperature,
            top_k=top_k,
            max_prompt_memories=max_prompt_memories,
        ),
    )
    return RuntimeFrontendContext(snapshot=snapshot, orchestrator=orchestrator)


@st.cache_resource(show_spinner=False)
def get_transcriber(
    *,
    model_name: str,
    language: str,
    device: str,
    fp16: bool,
) -> WhisperTranscriber:
    """Cache the Whisper transcriber between reruns."""

    config = WhisperConfig(
        model_name=model_name,
        language=language.strip() or None,
        device=device,
        fp16=fp16,
        min_segment_duration_ms=0,
        condition_on_previous_text=False,
    )
    return WhisperTranscriber(backend=OpenAIWhisperBackend(config=config), config=config)


@st.cache_resource(show_spinner=False)
def get_synthesizer(
    *,
    model_name: str,
    voice_id: str,
    language: str,
    instruction: str,
    device_map: str,
) -> SpeechSynthesizer:
    """Cache the TTS synthesizer between reruns."""

    config = TextToSpeechConfig(
        model_name=model_name,
        voice_id=voice_id.strip() or None,
        response_format="wav",
        language=language.strip() or None,
        instruction=instruction.strip() or None,
        device_map=device_map.strip() or None,
    )
    return SpeechSynthesizer(
        backend=QwenTTSBackend(config=config),
        config=config,
    )


def transcribe_browser_audio(audio_bytes: bytes, transcriber: WhisperTranscriber):
    """Transcribe one browser-recorded clip."""

    segment = speech_segment_from_wav_bytes(audio_bytes)
    transcription = transcriber.transcribe_segment(segment)
    if transcription is None:
        raise ValueError("Aucun texte exploitable n'a ete detecte.")
    user_text = transcription.text.strip()
    if not user_text:
        raise ValueError("Le clip est vide ou trop court.")
    return transcription, user_text


def build_turn(
    *,
    source: str,
    user_text: str,
    context: RuntimeFrontendContext,
    synthesizer: SpeechSynthesizer | None,
    max_history_messages: int,
    transcript_latency_ms: float | None = None,
) -> RuntimeFrontendTurn:
    """Run one assistant turn from normalized text input."""

    response = context.orchestrator.respond(
        context.snapshot.patient.patient_id,
        user_text,
        conversation_history=history_messages(),
    )

    browser_audio_bytes = None
    browser_audio_format = None
    audio_duration_ms = None
    synthesis_latency_ms = None
    tts_error = None
    if synthesizer is None:
        tts_error = "Le TTS n'est pas disponible."
    else:
        try:
            synthesis = synthesizer.synthesize(response.answer)
            browser_audio_bytes, browser_audio_format, audio_duration_ms = build_browser_preview(
                synthesis
            )
            synthesis_latency_ms = synthesis.latency_ms
        except Exception as error:
            tts_error = str(error)

    append_history(user_text, response.answer, max_history_messages=max_history_messages)

    retrieved_memories = tuple(
        {
            "source_label": evidence.source_label,
            "source_display_name": evidence.source_display_name,
            "summary": evidence.summary,
            "ranking_score": evidence.ranking_score,
            "signals": evidence.signals,
        }
        for evidence in response.trace.retrieved_memories
    )

    return RuntimeFrontendTurn(
        source=source,
        user_text=user_text,
        assistant_text=response.answer,
        transcript_latency_ms=transcript_latency_ms,
        generation_latency_ms=response.generation.latency_ms,
        synthesis_latency_ms=synthesis_latency_ms,
        audio_duration_ms=audio_duration_ms,
        browser_audio_bytes=browser_audio_bytes,
        browser_audio_format=browser_audio_format,
        retrieved_memories=retrieved_memories,
        guard_applied=response.trace.guard_applied,
        guard_reason=response.trace.guard_reason,
        tts_error=tts_error,
    )


def push_turn(turn: RuntimeFrontendTurn) -> None:
    """Persist one rendered turn."""

    turns = st.session_state["runtime_turns"]
    turns.append(turn)
    st.session_state["runtime_turns"] = turns[-12:]
    st.session_state["runtime_error"] = ""
    st.session_state["runtime_status"] = "Reponse prete"


def render_turn(turn: RuntimeFrontendTurn, *, index: int) -> None:
    """Render one full conversation turn."""

    render_message_card("user", turn.user_text)
    render_message_card("assistant", turn.assistant_text)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Source", turn.source)
    metric_columns[1].metric(
        "Transcription",
        f"{turn.transcript_latency_ms:.0f} ms" if turn.transcript_latency_ms is not None else "n/a",
    )
    metric_columns[2].metric(
        "Generation",
        f"{turn.generation_latency_ms:.0f} ms" if turn.generation_latency_ms is not None else "n/a",
    )
    metric_columns[3].metric(
        "Synthese",
        f"{turn.synthesis_latency_ms:.0f} ms" if turn.synthesis_latency_ms is not None else "n/a",
    )

    if turn.browser_audio_bytes is not None and turn.browser_audio_format is not None:
        st.audio(turn.browser_audio_bytes, format=turn.browser_audio_format)
        if turn.audio_duration_ms is not None:
            st.caption(f"Audio assistant: {turn.audio_duration_ms:.0f} ms")

    if turn.tts_error is not None:
        st.warning(f"Synthese vocale indisponible: {turn.tts_error}")

    with st.expander(f"Memoire mobilisee pour le tour {index}", expanded=False):
        render_memory_chips(turn.retrieved_memories)
        for memory in turn.retrieved_memories:
            st.write(
                f"{memory['source_label']} | {memory['source_display_name']} | "
                f"score={memory['ranking_score']:.3f}"
            )
            st.caption(str(memory["summary"]))
            if memory["signals"]:
                st.caption("Indices: " + ", ".join(memory["signals"]))

        if turn.guard_applied:
            st.warning(f"Garde de grounding activee: {turn.guard_reason or 'raison non precisee'}")


def main() -> None:
    """Run the Streamlit runtime frontend."""

    st.set_page_config(
        page_title="Memento Runtime",
        page_icon="o",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_global_styles()
    init_session_state()

    with st.sidebar:
        st.header("Configuration runtime")
        snapshot_json = st.text_area(
            "Snapshot patient (JSON)",
            value=DEFAULT_SNAPSHOT_JSON,
            height=380,
        )
        llm_base_url = st.text_input(
            "LLM base URL",
            value=os.getenv("MEMENTO_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        )
        llm_api_key = st.text_input(
            "LLM API key",
            value=os.getenv("MEMENTO_LLM_API_KEY", ""),
            type="password",
        )
        llm_timeout_seconds = st.slider(
            "Timeout LLM (s)",
            min_value=5.0,
            max_value=120.0,
            value=60.0,
            step=5.0,
        )
        llm_model = st.text_input("Modele LLM", value="Ministral 3 8B")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        top_k = st.slider("Top-k memoire", min_value=1, max_value=8, value=3)
        max_prompt_memories = st.slider("Souvenirs injectes", min_value=1, max_value=6, value=3)
        max_history_messages = st.slider("Historique conserve", min_value=0, max_value=12, value=6)

        st.header("Whisper")
        whisper_model = st.selectbox(
            "Modele Whisper",
            options=("tiny", "base", "small", "medium", "large-v3", "turbo"),
            index=4,
        )
        whisper_language = st.text_input("Langue STT", value="fr")
        whisper_device = st.selectbox(
            "Device Whisper",
            options=("cpu", "cuda"),
            index=1 if torch_cuda_available() else 0,
        )
        whisper_fp16 = st.checkbox("Activer fp16", value=False, disabled=whisper_device != "cuda")

        st.header("Qwen TTS")
        tts_model = st.text_input("Modele TTS", value=DEFAULT_QWEN_TTS_MODEL_NAME)
        tts_speaker = st.text_input("Speaker", value=DEFAULT_QWEN_TTS_SPEAKER)
        tts_language = st.text_input("Langue TTS", value=DEFAULT_QWEN_TTS_LANGUAGE)
        tts_instruction = st.text_area("Instruction de style", value="", height=100)
        tts_device_map = st.text_input("device_map", value=default_device_map())

        action_columns = st.columns(2)
        if action_columns[0].button("Effacer l'historique", use_container_width=True):
            reset_conversation()
        if action_columns[1].button("Recharger l'UI", use_container_width=True):
            st.rerun()

    context_error = None
    transcriber_error = None
    synthesizer_error = None
    runtime_context = None
    transcriber = None
    synthesizer = None

    try:
        runtime_context = build_runtime_context(
            snapshot_json=snapshot_json,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key or None,
            llm_timeout_seconds=float(llm_timeout_seconds),
            llm_model=llm_model,
            temperature=float(temperature),
            top_k=int(top_k),
            max_prompt_memories=int(max_prompt_memories),
        )
    except Exception as error:
        context_error = str(error)

    if context_error is None:
        try:
            transcriber = get_transcriber(
                model_name=whisper_model,
                language=whisper_language,
                device=whisper_device,
                fp16=bool(whisper_fp16),
            )
        except Exception as error:
            transcriber_error = str(error)

        try:
            synthesizer = get_synthesizer(
                model_name=tts_model,
                voice_id=tts_speaker,
                language=tts_language,
                instruction=tts_instruction,
                device_map=tts_device_map,
            )
        except Exception as error:
            synthesizer_error = str(error)

    patient_name = "Patient non charge"
    if runtime_context is not None:
        patient_name = (
            runtime_context.snapshot.patient.preferred_name
            or runtime_context.snapshot.patient.display_name
        )

    render_hero(
        patient_name=patient_name,
        status=st.session_state["runtime_status"],
        llm_model=llm_model,
        whisper_device=whisper_device,
    )

    if context_error is not None:
        st.error(f"Snapshot/runtime invalide: {context_error}")
        return

    warning_messages = [message for message in (transcriber_error, synthesizer_error) if message]
    for message in warning_messages:
        st.warning(message)

    controls_left, controls_center, controls_right = st.columns([1.2, 2.6, 1.2])
    with controls_center:
        st.caption(
            "Enregistre une question avec le micro navigateur ou tape un message. "
            "La reponse se genere automatiquement avec la memoire patient."
        )
        recorded_audio = st.audio_input(
            "Parler a Memento",
            sample_rate=16_000,
            key="runtime_audio_input",
        )
        text_question = st.text_input(
            "Ou ecrire une question",
            value="",
            placeholder="Ex: Qui vient dimanche ?",
            key="runtime_text_question",
        )
        send_text = st.button("Envoyer le message", type="primary", use_container_width=True)

    if transcriber is not None and recorded_audio is not None:
        audio_bytes = recorded_audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state["runtime_last_audio_hash"]:
            st.session_state["runtime_last_audio_hash"] = audio_hash
            st.session_state["runtime_status"] = "Transcription et reponse en cours"
            try:
                with st.spinner("Traitement de la question vocale..."):
                    transcription, user_text = transcribe_browser_audio(audio_bytes, transcriber)
                    turn = build_turn(
                        source="voix",
                        user_text=user_text,
                        context=runtime_context,
                        synthesizer=synthesizer,
                        max_history_messages=max_history_messages,
                        transcript_latency_ms=transcription.latency_ms,
                    )
            except Exception as error:
                st.session_state["runtime_error"] = str(error)
                st.session_state["runtime_status"] = "Erreur"
            else:
                push_turn(turn)
                st.rerun()

    if send_text:
        normalized_question = text_question.strip()
        if not normalized_question:
            st.session_state["runtime_error"] = "Le message texte est vide."
            st.session_state["runtime_status"] = "En attente d'une question"
        else:
            st.session_state["runtime_status"] = "Generation en cours"
            try:
                with st.spinner("Generation de la reponse..."):
                    turn = build_turn(
                        source="texte",
                        user_text=normalized_question,
                        context=runtime_context,
                        synthesizer=synthesizer,
                        max_history_messages=max_history_messages,
                    )
            except Exception as error:
                st.session_state["runtime_error"] = str(error)
                st.session_state["runtime_status"] = "Erreur"
            else:
                push_turn(turn)
                st.session_state["runtime_text_question"] = ""
                st.rerun()

    if st.session_state["runtime_error"]:
        st.error(st.session_state["runtime_error"])

    turns: list[RuntimeFrontendTurn] = list(st.session_state["runtime_turns"])
    overview_left, overview_right = st.columns([1.45, 1.05])

    with overview_left:
        st.subheader("Conversation")
        if not turns:
            st.info("Aucun tour encore. Pose une question a la voix ou au clavier.")
        for index, turn in enumerate(reversed(turns), start=1):
            render_turn(turn, index=index)

    with overview_right:
        st.subheader("Patient")
        patient = runtime_context.snapshot.patient
        st.write(f"Nom affiche: {patient.display_name}")
        st.write(f"Nom prefere: {patient.preferred_name or 'n/a'}")
        st.write("Reperes rassurants:")
        for anchor in patient.anchors:
            st.caption(anchor)

        st.write("Notes de soin:")
        for note in patient.care_notes:
            st.caption(note)

        st.subheader("Etat de session")
        session_columns = st.columns(2)
        session_columns[0].metric("Tours", len(turns))
        session_columns[1].metric("Historique", len(st.session_state["runtime_history"]))

        if turns:
            last_turn = turns[-1]
            st.write("Derniers souvenirs utilises:")
            render_memory_chips(last_turn.retrieved_memories)


if __name__ == "__main__":
    main()
