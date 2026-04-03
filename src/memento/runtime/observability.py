"""Structured observability helpers for the live runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Protocol

from .models import RuntimeEvent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuntimeLatencyTrace:
    """End-to-end latency trace for one completed assistant turn."""

    session_id: str
    patient_id: str
    turn_id: str
    traced_at: str
    turn_latency_ms: float
    transcription_latency_ms: float
    transcription_duration_ms: float
    generation_latency_ms: float | None
    synthesis_latency_ms: float
    playback_dispatch_latency_ms: float | None
    playback_completion_latency_ms: float | None
    end_to_end_latency_ms: float | None
    end_to_end_completion_latency_ms: float | None
    audio_duration_ms: float
    meets_targets: bool
    guard_applied: bool
    guard_reason: str = ""


@dataclass(frozen=True)
class RuntimeErrorRecord:
    """Structured runtime failure attached to one stage and correlation ids."""

    session_id: str
    patient_id: str
    stage: str
    occurred_at: str
    error_type: str
    error_message: str
    turn_id: str | None = None
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class RuntimeAlert:
    """Operational alert derived from runtime metrics or failures."""

    session_id: str
    patient_id: str
    alert_type: str
    severity: str
    emitted_at: str
    detail: str
    turn_id: str | None = None
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class RuntimeMetricsSnapshot:
    """Persisted runtime counters and rolling latency aggregates."""

    session_id: str
    patient_id: str
    started_at: str
    updated_at: str
    frames_processed: int
    turns_completed: int
    turns_aborted: int
    segments_skipped: int
    transcripts_skipped: int
    playback_interruptions: int
    errors_total: int
    alerts_total: int
    average_turn_latency_ms: float | None
    average_transcription_latency_ms: float | None
    average_generation_latency_ms: float | None
    average_end_to_end_latency_ms: float | None
    last_error_stage: str = ""


class RuntimeObserver(Protocol):
    """Protocol implemented by concrete observability sinks."""

    def on_session_started(self, *, session_id: str, patient_id: str, payload: dict[str, object]) -> None:
        """Record session metadata before the runtime loop starts."""

    def on_event(self, event: RuntimeEvent) -> None:
        """Record one structured runtime event."""

    def on_latency_trace(self, trace: RuntimeLatencyTrace) -> None:
        """Record one turn-level latency trace."""

    def on_error(self, error: RuntimeErrorRecord) -> None:
        """Record one runtime error."""

    def on_alert(self, alert: RuntimeAlert) -> None:
        """Record one derived alert."""

    def on_metrics(self, snapshot: RuntimeMetricsSnapshot) -> None:
        """Persist one metrics snapshot."""

    def on_session_stopped(self, *, session_id: str, patient_id: str, payload: dict[str, object]) -> None:
        """Record session stop metadata."""


class FileRuntimeObserver:
    """Persist observability records as JSONL files plus a metrics snapshot."""

    def __init__(self, base_directory: str | Path) -> None:
        self._base_directory = Path(base_directory)
        self._base_directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._session_directory: Path | None = None

    def on_session_started(self, *, session_id: str, patient_id: str, payload: dict[str, object]) -> None:
        session_directory = self._base_directory / session_id
        session_directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._session_directory = session_directory
        self._append_runtime_record(
            {
                "record_type": "session_started",
                "session_id": session_id,
                "patient_id": patient_id,
                "recorded_at": utc_now_iso(),
                "payload": payload,
            }
        )

    def on_event(self, event: RuntimeEvent) -> None:
        self._append_runtime_record(
            {
                "record_type": "event",
                **_json_ready(asdict(event)),
            }
        )

    def on_latency_trace(self, trace: RuntimeLatencyTrace) -> None:
        self._append_runtime_record(
            {
                "record_type": "latency_trace",
                **_json_ready(asdict(trace)),
            }
        )

    def on_error(self, error: RuntimeErrorRecord) -> None:
        payload = _json_ready(asdict(error))
        self._append_runtime_record({"record_type": "error", **payload})
        self._append_jsonl("errors.jsonl", payload)

    def on_alert(self, alert: RuntimeAlert) -> None:
        payload = _json_ready(asdict(alert))
        self._append_runtime_record({"record_type": "alert", **payload})
        self._append_jsonl("alerts.jsonl", payload)

    def on_metrics(self, snapshot: RuntimeMetricsSnapshot) -> None:
        self._write_json("metrics.json", _json_ready(asdict(snapshot)))

    def on_session_stopped(self, *, session_id: str, patient_id: str, payload: dict[str, object]) -> None:
        self._append_runtime_record(
            {
                "record_type": "session_stopped",
                "session_id": session_id,
                "patient_id": patient_id,
                "recorded_at": utc_now_iso(),
                "payload": payload,
            }
        )

    def _append_runtime_record(self, payload: dict[str, object]) -> None:
        self._append_jsonl("runtime.jsonl", payload)

    def _append_jsonl(self, filename: str, payload: dict[str, object]) -> None:
        session_directory = self._require_session_directory()
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        with self._lock:
            with (session_directory / filename).open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _write_json(self, filename: str, payload: dict[str, object]) -> None:
        session_directory = self._require_session_directory()
        content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
        with self._lock:
            (session_directory / filename).write_text(content + "\n", encoding="utf-8")

    def _require_session_directory(self) -> Path:
        if self._session_directory is None:
            raise RuntimeError("runtime observer session has not been started")
        return self._session_directory


def _json_ready(value: object):
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
