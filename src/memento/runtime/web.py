"""Small HTTP API exposing the Memento runtime to a web frontend."""

from __future__ import annotations

import argparse
import base64
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

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
    speech_segment_from_wav_bytes,
)
from memento.conversation import (
    ConversationConfig,
    ConversationMessage,
    ConversationOrchestrator,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)
from memento.memory import MemorySyncEngine, PatientMemorySnapshot

from .bootstrap import load_snapshot_from_json_file, snapshot_from_dict


class RuntimeApiHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing a minimal runtime API."""

    server_version = "MementoRuntimeAPI/0.1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/api/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._write_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "snapshot_file": self.server.snapshot_file,
                "patient_id": self.server.snapshot.patient.patient_id,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/runtime/turn":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            payload = self._read_json_body()
            response = _handle_turn_request(payload, default_snapshot=self.server.snapshot)
        except ValueError as error:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # pragma: no cover - runtime only
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            return

        self._write_json(HTTPStatus.OK, response)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            raise ValueError("empty request body")
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        response_body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(int(status))
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class RuntimeApiServer(ThreadingHTTPServer):
    """Threaded HTTP server storing the default snapshot."""

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        snapshot: PatientMemorySnapshot,
        snapshot_file: str,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.snapshot = snapshot
        self.snapshot_file = snapshot_file


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the runtime HTTP API."""

    parser = argparse.ArgumentParser(description="Run the Memento runtime HTTP API.")
    parser.add_argument("--snapshot-file", required=True, help="Path to a patient snapshot JSON file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the HTTP API server."""

    args = build_argument_parser().parse_args(argv)
    snapshot = load_snapshot_from_json_file(args.snapshot_file)
    server = RuntimeApiServer(
        (args.host, args.port),
        RuntimeApiHandler,
        snapshot=snapshot,
        snapshot_file=args.snapshot_file,
    )
    try:
        print(f"Memento runtime API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop only
        pass
    finally:
        server.server_close()
    return 0


def _handle_turn_request(
    payload: dict[str, Any],
    *,
    default_snapshot: PatientMemorySnapshot,
) -> dict[str, Any]:
    snapshot = _snapshot_from_payload(payload.get("snapshot"), default_snapshot=default_snapshot)
    context = _get_runtime_context(
        snapshot_json=_snapshot_cache_key(snapshot),
        llm_base_url=_string_value(payload.get("llmBaseUrl"), default="http://127.0.0.1:11434/v1"),
        llm_api_key=_optional_string_value(payload.get("llmApiKey")),
        llm_timeout_seconds=_float_value(payload.get("llmTimeoutSeconds"), default=60.0),
        llm_model=_string_value(payload.get("llmModel"), default="Ministral 3 8B"),
        temperature=_float_value(payload.get("temperature"), default=0.2),
        top_k=_int_value(payload.get("topK"), default=3),
        max_prompt_memories=_int_value(payload.get("maxPromptMemories"), default=3),
    )

    history = _conversation_history(payload.get("conversationHistory"))
    transcriber = None
    transcript = None
    user_text = _optional_string_value(payload.get("userText"))
    mode = _string_value(payload.get("mode"), default="text").lower()

    if mode == "voice":
        audio_base64 = _string_value(payload.get("audioBase64"), default="")
        if not audio_base64:
            raise ValueError("audioBase64 is required in voice mode")
        audio_bytes = _decode_audio_base64(audio_base64)
        transcriber = _get_transcriber(
            model_name=_string_value(payload.get("whisperModel"), default="large-v3"),
            language=_string_value(payload.get("whisperLanguage"), default="fr"),
            device=_string_value(payload.get("whisperDevice"), default="cpu"),
            fp16=_bool_value(payload.get("whisperFp16"), default=False),
        )
        segment = speech_segment_from_wav_bytes(audio_bytes)
        transcript = transcriber.transcribe_segment(segment)
        if transcript is None or not transcript.text.strip():
            raise ValueError("no usable speech transcript was detected")
        user_text = transcript.text.strip()
    elif mode != "text":
        raise ValueError("mode must be `text` or `voice`")

    if not user_text:
        raise ValueError("userText must not be empty")

    response = context["orchestrator"].respond(
        context["snapshot"].patient.patient_id,
        user_text,
        conversation_history=history,
    )

    synthesis_latency_ms = None
    tts_error = None
    assistant_audio_base64 = None
    assistant_audio_mime_type = None
    if _bool_value(payload.get("ttsEnabled"), default=True):
        try:
            synthesizer = _get_synthesizer(
                model_name=_string_value(payload.get("ttsModel"), default=DEFAULT_QWEN_TTS_MODEL_NAME),
                voice_id=_string_value(payload.get("ttsSpeaker"), default=DEFAULT_QWEN_TTS_SPEAKER),
                language=_string_value(payload.get("ttsLanguage"), default=DEFAULT_QWEN_TTS_LANGUAGE),
                instruction=_optional_string_value(payload.get("ttsInstruction")),
                device_map=_optional_string_value(payload.get("ttsDeviceMap")) or "auto",
            )
            synthesis = synthesizer.synthesize(response.answer)
            assistant_audio_base64 = base64.b64encode(synthesis.audio_bytes).decode("ascii")
            assistant_audio_mime_type = "audio/wav"
            synthesis_latency_ms = synthesis.latency_ms
        except Exception as error:  # pragma: no cover - runtime only
            tts_error = str(error)

    return {
        "patient": {
            "patientId": context["snapshot"].patient.patient_id,
            "displayName": context["snapshot"].patient.display_name,
            "preferredName": context["snapshot"].patient.preferred_name,
            "anchors": list(context["snapshot"].patient.anchors),
            "careNotes": list(context["snapshot"].patient.care_notes),
        },
        "turn": {
            "source": mode,
            "userText": user_text,
            "assistantText": response.answer,
            "transcriptLatencyMs": transcript.latency_ms if transcript is not None else None,
            "generationLatencyMs": response.generation.latency_ms,
            "synthesisLatencyMs": synthesis_latency_ms,
            "audioBase64": assistant_audio_base64,
            "audioMimeType": assistant_audio_mime_type,
            "retrievedMemories": [
                {
                    "sourceLabel": evidence.source_label,
                    "sourceDisplayName": evidence.source_display_name,
                    "summary": evidence.summary,
                    "rankingScore": evidence.ranking_score,
                    "signals": list(evidence.signals),
                }
                for evidence in response.trace.retrieved_memories
            ],
            "guardApplied": response.trace.guard_applied,
            "guardReason": response.trace.guard_reason,
            "ttsError": tts_error,
        },
    }


@lru_cache(maxsize=4)
def _get_runtime_context(
    *,
    snapshot_json: str,
    llm_base_url: str,
    llm_api_key: str | None,
    llm_timeout_seconds: float,
    llm_model: str,
    temperature: float,
    top_k: int,
    max_prompt_memories: int,
) -> dict[str, Any]:
    snapshot = snapshot_from_dict(json.loads(snapshot_json))
    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(snapshot)
    orchestrator = ConversationOrchestrator(
        memory_engine=memory_engine,
        backend=OpenAICompatibleConversationBackend(
            config=OpenAICompatibleBackendConfig(
                base_url=llm_base_url,
                api_key=llm_api_key,
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
    return {"snapshot": snapshot, "orchestrator": orchestrator}


@lru_cache(maxsize=4)
def _get_transcriber(
    *,
    model_name: str,
    language: str,
    device: str,
    fp16: bool,
) -> WhisperTranscriber:
    config = WhisperConfig(
        model_name=model_name,
        language=language,
        device=device,
        fp16=fp16,
        min_segment_duration_ms=0.0,
        condition_on_previous_text=False,
    )
    return WhisperTranscriber(backend=OpenAIWhisperBackend(config=config), config=config)


@lru_cache(maxsize=4)
def _get_synthesizer(
    *,
    model_name: str,
    voice_id: str,
    language: str,
    instruction: str | None,
    device_map: str,
) -> SpeechSynthesizer:
    config = TextToSpeechConfig(
        model_name=model_name,
        voice_id=voice_id,
        response_format="wav",
        language=language,
        instruction=instruction,
        device_map=device_map,
    )
    return SpeechSynthesizer(backend=QwenTTSBackend(config=config), config=config)


def _snapshot_from_payload(
    snapshot_payload: Any,
    *,
    default_snapshot: PatientMemorySnapshot,
) -> PatientMemorySnapshot:
    if snapshot_payload is None:
        return default_snapshot
    if not isinstance(snapshot_payload, dict):
        raise ValueError("snapshot must be a JSON object when provided")
    return snapshot_from_dict(snapshot_payload)


def _snapshot_cache_key(snapshot: PatientMemorySnapshot) -> str:
    payload = {
        "patient": {
            "patient_id": snapshot.patient.patient_id,
            "display_name": snapshot.patient.display_name,
            "preferred_name": snapshot.patient.preferred_name,
            "care_notes": list(snapshot.patient.care_notes),
            "anchors": list(snapshot.patient.anchors),
        },
        "people": [
            {
                "person_id": item.person_id,
                "name": item.name,
                "relationship_to_patient": item.relationship_to_patient,
                "notes": item.notes,
                "emotional_significance": item.emotional_significance,
            }
            for item in snapshot.people
        ],
        "places": [
            {
                "place_id": item.place_id,
                "name": item.name,
                "category": item.category,
                "notes": item.notes,
            }
            for item in snapshot.places
        ],
        "routines": [
            {
                "routine_id": item.routine_id,
                "title": item.title,
                "schedule": item.schedule,
                "description": item.description,
                "cue": item.cue,
                "support_strategy": item.support_strategy,
                "place_id": item.place_id,
            }
            for item in snapshot.routines
        ],
        "episodes": [
            {
                "episode_id": item.episode_id,
                "title": item.title,
                "narrative": item.narrative,
                "happened_on": item.happened_on,
                "people_ids": list(item.people_ids),
                "place_id": item.place_id,
                "emotions": [
                    {
                        "label": emotion.label,
                        "valence": emotion.valence,
                        "intensity": emotion.intensity,
                        "notes": emotion.notes,
                    }
                    for emotion in item.emotions
                ],
                "tags": list(item.tags),
            }
            for item in snapshot.episodes
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _conversation_history(value: Any) -> tuple[ConversationMessage, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("conversationHistory must be an array")
    history: list[ConversationMessage] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("conversationHistory entries must be objects")
        history.append(
            ConversationMessage(
                role=_string_value(item.get("role"), default=""),
                content=_string_value(item.get("content"), default=""),
            )
        )
    return tuple(history)


def _decode_audio_base64(value: str) -> bytes:
    payload = value.strip()
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise ValueError("audioBase64 is not valid base64") from exc


def _string_value(value: Any, *, default: str) -> str:
    normalized = _optional_string_value(value)
    return normalized if normalized is not None else default


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _float_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float value: {value}") from exc


def _int_value(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer value: {value}") from exc


def _bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"invalid boolean value: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
