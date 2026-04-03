"""Helpers to bootstrap the live runtime from a patient snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from memento.audio import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    MicrophoneConfig,
    OpenAIWhisperBackend,
    QwenTTSBackend,
    RealTimeMicrophone,
    SoundDeviceInput,
    SoundDeviceOutput,
    SpeakerConfig,
    SpeakerPlayer,
    SpeechSynthesizer,
    StreamingSpeechSegmenter,
    TextToSpeechConfig,
    VoiceActivityConfig,
    VoiceResponsePipeline,
    WhisperConfig,
    WhisperTranscriber,
)
from memento.conversation import (
    ConversationConfig,
    ConversationModelBackend,
    ConversationOrchestrator,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)
from memento.memory import (
    AffectiveState,
    MemoryEpisode,
    MemorySyncEngine,
    PatientMemorySnapshot,
    PatientProfile,
    PersonProfile,
    PlaceProfile,
    RoutineProfile,
)

from .config import RuntimeConfig
from .observability import RuntimeObserver
from .service import MementoRuntime


def load_snapshot_from_json_file(path: str | Path) -> PatientMemorySnapshot:
    """Load one patient snapshot from a JSON file."""

    snapshot_path = Path(path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot JSON root must be an object")
    return snapshot_from_dict(payload)


def snapshot_from_dict(payload: dict[str, object]) -> PatientMemorySnapshot:
    """Build a typed patient snapshot from a plain JSON-like dictionary."""

    patient_payload = _mapping(payload.get("patient"), field_name="patient")
    people_payload = _list(payload.get("people", []), field_name="people")
    places_payload = _list(payload.get("places", []), field_name="places")
    routines_payload = _list(payload.get("routines", []), field_name="routines")
    episodes_payload = _list(payload.get("episodes", []), field_name="episodes")

    return PatientMemorySnapshot(
        patient=PatientProfile(
            patient_id=_required_string(patient_payload.get("patient_id"), field_name="patient.patient_id"),
            display_name=_required_string(patient_payload.get("display_name"), field_name="patient.display_name"),
            preferred_name=_optional_string(patient_payload.get("preferred_name")),
            care_notes=_string_tuple(patient_payload.get("care_notes")),
            anchors=_string_tuple(patient_payload.get("anchors")),
        ),
        people=tuple(
            PersonProfile(
                person_id=_required_string(item.get("person_id"), field_name="people[].person_id"),
                name=_required_string(item.get("name"), field_name="people[].name"),
                relationship_to_patient=_required_string(
                    item.get("relationship_to_patient"),
                    field_name="people[].relationship_to_patient",
                ),
                notes=_optional_string(item.get("notes")),
                emotional_significance=float(item.get("emotional_significance", 0.5)),
            )
            for item in (_mapping(person, field_name="people[]") for person in people_payload)
        ),
        places=tuple(
            PlaceProfile(
                place_id=_required_string(item.get("place_id"), field_name="places[].place_id"),
                name=_required_string(item.get("name"), field_name="places[].name"),
                category=_required_string(item.get("category"), field_name="places[].category"),
                notes=_optional_string(item.get("notes")),
            )
            for item in (_mapping(place, field_name="places[]") for place in places_payload)
        ),
        routines=tuple(
            RoutineProfile(
                routine_id=_required_string(item.get("routine_id"), field_name="routines[].routine_id"),
                title=_required_string(item.get("title"), field_name="routines[].title"),
                schedule=_required_string(item.get("schedule"), field_name="routines[].schedule"),
                description=_required_string(item.get("description"), field_name="routines[].description"),
                cue=_optional_string(item.get("cue")),
                support_strategy=_optional_string(item.get("support_strategy")),
                place_id=_optional_string(item.get("place_id")) or None,
            )
            for item in (_mapping(routine, field_name="routines[]") for routine in routines_payload)
        ),
        episodes=tuple(
            MemoryEpisode(
                episode_id=_required_string(item.get("episode_id"), field_name="episodes[].episode_id"),
                title=_required_string(item.get("title"), field_name="episodes[].title"),
                narrative=_required_string(item.get("narrative"), field_name="episodes[].narrative"),
                happened_on=_optional_string(item.get("happened_on")),
                people_ids=_string_tuple(item.get("people_ids")),
                place_id=_optional_string(item.get("place_id")) or None,
                emotions=tuple(
                    AffectiveState(
                        label=_required_string(emotion.get("label"), field_name="episodes[].emotions[].label"),
                        valence=float(emotion.get("valence", 0.0)),
                        intensity=float(emotion.get("intensity", 0.0)),
                        notes=_optional_string(emotion.get("notes")),
                    )
                    for emotion in (
                        _mapping(emotion_payload, field_name="episodes[].emotions[]")
                        for emotion_payload in _list(item.get("emotions", []), field_name="episodes[].emotions")
                    )
                ),
                tags=_string_tuple(item.get("tags")),
            )
            for item in (_mapping(episode, field_name="episodes[]") for episode in episodes_payload)
        ),
    )


def build_live_runtime(
    *,
    snapshot: PatientMemorySnapshot,
    runtime_config: RuntimeConfig | None = None,
    conversation_config: ConversationConfig | None = None,
    llm_backend: ConversationModelBackend | None = None,
    llm_backend_config: OpenAICompatibleBackendConfig | None = None,
    whisper_config: WhisperConfig | None = None,
    microphone_config: MicrophoneConfig | None = None,
    vad_config: VoiceActivityConfig | None = None,
    tts_config: TextToSpeechConfig | None = None,
    speaker_config: SpeakerConfig | None = None,
    observer: RuntimeObserver | None = None,
) -> MementoRuntime:
    """Build the default live runtime stack for one patient snapshot."""

    resolved_runtime_config = runtime_config or RuntimeConfig(patient_id=snapshot.patient.patient_id)
    if resolved_runtime_config.patient_id != snapshot.patient.patient_id:
        raise ValueError("runtime_config.patient_id must match snapshot.patient.patient_id")

    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(snapshot)

    resolved_whisper_config = whisper_config or WhisperConfig()
    resolved_tts_config = tts_config or TextToSpeechConfig(
        model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
        voice_id=DEFAULT_QWEN_TTS_SPEAKER,
        language=DEFAULT_QWEN_TTS_LANGUAGE,
    )
    resolved_conversation_backend = llm_backend or OpenAICompatibleConversationBackend(
        config=llm_backend_config or OpenAICompatibleBackendConfig()
    )

    return MementoRuntime(
        microphone=RealTimeMicrophone(
            device=SoundDeviceInput(),
            config=microphone_config or MicrophoneConfig(device_name="default"),
        ),
        segmenter=StreamingSpeechSegmenter(config=vad_config),
        transcriber=WhisperTranscriber(
            backend=OpenAIWhisperBackend(config=resolved_whisper_config),
            config=resolved_whisper_config,
        ),
        orchestrator=ConversationOrchestrator(
            memory_engine=memory_engine,
            backend=resolved_conversation_backend,
            config=conversation_config,
        ),
        voice_pipeline=VoiceResponsePipeline(
            synthesizer=SpeechSynthesizer(
                backend=QwenTTSBackend(config=resolved_tts_config),
                config=resolved_tts_config,
            ),
            player=SpeakerPlayer(
                device=SoundDeviceOutput(),
                config=speaker_config or SpeakerConfig(),
            ),
        ),
        config=resolved_runtime_config,
        observer=observer,
    )


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _list(value: object, *, field_name: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _required_string(value: object, *, field_name: str) -> str:
    normalized = _optional_string(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if not isinstance(value, list):
        raise ValueError("expected an array of strings")
    return tuple(
        normalized
        for normalized in (str(item).strip() for item in value)
        if normalized
    )
