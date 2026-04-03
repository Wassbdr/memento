"""End-to-end runtime chaining microphone, STT, memory, LLM and TTS."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from memento.audio import (
    EnergyVAD,
    RealTimeMicrophone,
    SpeechSegment,
    StreamingSpeechSegmenter,
    VoiceResponsePipeline,
    WhisperTranscriber,
)
from memento.conversation import ConversationMessage, ConversationOrchestrator

from .config import RuntimeConfig
from .models import ConversationHistory, RuntimeEvent, RuntimeEventHandler, RuntimeTurn
from .observability import (
    RuntimeAlert,
    RuntimeErrorRecord,
    RuntimeLatencyTrace,
    RuntimeMetricsSnapshot,
    RuntimeObserver,
    utc_now_iso,
)


@dataclass(frozen=True)
class _PlaybackState:
    deadline_monotonic: float | None = None

    @property
    def active(self) -> bool:
        if self.deadline_monotonic is None:
            return False
        return perf_counter() < self.deadline_monotonic


class MementoRuntime:
    """Live runtime that continuously turns speech into spoken answers."""

    def __init__(
        self,
        *,
        microphone: RealTimeMicrophone,
        segmenter: StreamingSpeechSegmenter,
        transcriber: WhisperTranscriber,
        orchestrator: ConversationOrchestrator,
        voice_pipeline: VoiceResponsePipeline,
        config: RuntimeConfig,
        event_handler: RuntimeEventHandler | None = None,
        observer: RuntimeObserver | None = None,
        initial_history: ConversationHistory = (),
    ) -> None:
        self._microphone = microphone
        self._segmenter = segmenter
        self._transcriber = transcriber
        self._orchestrator = orchestrator
        self._voice_pipeline = voice_pipeline
        self._config = config
        self._event_handler = event_handler
        self._observer = observer
        self._history = list(_normalize_history(initial_history))
        self._running = False
        self._shutdown_requested = False
        self._session_id = ""
        self._started_at = ""
        self._turn_index = 0
        self._frames_processed = 0
        self._turns_completed = 0
        self._turns_aborted = 0
        self._segments_skipped = 0
        self._transcripts_skipped = 0
        self._playback_interruptions = 0
        self._errors_total = 0
        self._alerts_total = 0
        self._last_error_stage = ""
        self._turn_latency_total_ms = 0.0
        self._transcription_latency_total_ms = 0.0
        self._generation_latency_total_ms = 0.0
        self._end_to_end_latency_total_ms = 0.0
        self._playback_state = _PlaybackState()
        self._speech_detector = EnergyVAD(config=self._segmenter.config)

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def conversation_history(self) -> ConversationHistory:
        return tuple(self._history)

    @property
    def session_id(self) -> str:
        return self._session_id

    def run_forever(self, *, max_frames: int | None = None) -> tuple[RuntimeTurn, ...]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if self._running:
            raise RuntimeError("runtime loop is already running")

        self._start_session()
        turns: list[RuntimeTurn] = []
        self._running = True
        loop_failed = False
        self._emit("runtime_started", "Live runtime loop started.")
        self._microphone.start()

        try:
            while self._running:
                if max_frames is not None and self._frames_processed >= max_frames:
                    break

                try:
                    frame = self._microphone.capture_frame()
                except Exception as exc:
                    self._record_error("audio_capture", exc)
                    raise
                self._frames_processed += 1
                self._maybe_interrupt_playback_for_user_speech(frame)

                segment = self._segmenter.push_frame(frame)
                if segment is None:
                    continue

                turn = self._handle_segment(segment)
                if turn is not None:
                    turns.append(turn)
        except Exception:
            loop_failed = True
            raise
        finally:
            if not loop_failed and not self._shutdown_requested:
                trailing_segment = self._segmenter.flush()
                if trailing_segment is not None:
                    trailing_turn = self._handle_segment(trailing_segment)
                    if trailing_turn is not None:
                        turns.append(trailing_turn)
            else:
                self._segmenter.flush()
            self._microphone.stop()
            self._running = False
            self._publish_metrics()
            self._emit("runtime_stopped", "Live runtime loop stopped.")
            self._notify_session_stopped()

        return tuple(turns)

    def stop(self) -> bool | None:
        self._running = False
        self._shutdown_requested = True
        stopped = self._voice_pipeline.stop()
        self._playback_state = _PlaybackState()
        return stopped

    def close(self) -> None:
        self.stop()
        self._microphone.stop()
        self._orchestrator.close()

    def _maybe_interrupt_playback_for_user_speech(self, frame) -> None:
        if not self._config.interrupt_playback_on_user_speech:
            return
        if not self._playback_state.active:
            return
        if not self._speech_detector.classify(frame).is_speech:
            return

        interrupted = self._voice_pipeline.stop()
        self._playback_state = _PlaybackState()
        self._playback_interruptions += 1
        self._publish_metrics()
        self._emit(
            "playback_interrupted",
            "User speech interrupted active playback.",
            payload={"interrupted": bool(interrupted)},
        )

    def _handle_segment(self, segment: SpeechSegment) -> RuntimeTurn | None:
        turn_id = self._next_turn_id()
        turn_started_at = perf_counter()
        if self._shutdown_requested:
            return None

        try:
            transcription = self._transcriber.transcribe_segment(segment)
        except Exception as exc:
            self._record_error(
                "transcription",
                exc,
                turn_id=turn_id,
                payload={
                    "start_frame_index": segment.start_frame_index,
                    "end_frame_index": segment.end_frame_index,
                    "duration_ms": segment.duration_ms,
                },
            )
            raise
        if transcription is None:
            self._segments_skipped += 1
            self._publish_metrics()
            self._emit(
                "segment_skipped",
                "Speech segment ignored because it was too short for transcription.",
                turn_id=turn_id,
            )
            return None

        user_text = transcription.text.strip()
        if len(user_text) < self._config.min_transcript_chars:
            self._transcripts_skipped += 1
            self._publish_metrics()
            self._emit(
                "transcript_skipped",
                "Transcript ignored because it did not meet the minimum length.",
                turn_id=turn_id,
                payload={"text": user_text},
            )
            return None

        if self._shutdown_requested:
            self._abort_turn(
                turn_id,
                "Runtime stop requested before response generation completed.",
            )
            return None

        self._emit(
            "transcript_ready",
            user_text,
            turn_id=turn_id,
            payload={
                "duration_ms": transcription.duration_ms,
                "latency_ms": transcription.latency_ms,
            },
        )

        try:
            response = self._orchestrator.respond(
                self._config.patient_id,
                user_text,
                conversation_history=self.conversation_history,
            )
        except Exception as exc:
            self._record_error(
                "conversation_generation",
                exc,
                turn_id=turn_id,
                payload={"user_text": user_text},
            )
            raise
        if self._shutdown_requested:
            self._abort_turn(
                turn_id,
                "Runtime stop requested before voice playback started.",
            )
            return None

        try:
            voice_response = self._voice_pipeline.speak(response.answer)
        except Exception as exc:
            self._record_error(
                "voice_response",
                exc,
                turn_id=turn_id,
                payload={"assistant_text": response.answer},
            )
            raise
        if self._shutdown_requested:
            self._voice_pipeline.stop()
            self._playback_state = _PlaybackState()
            self._abort_turn(
                turn_id,
                "Runtime stop requested while preparing the spoken response.",
            )
            return None

        self._append_history(user_text, response.answer)
        self._playback_state = _playback_state_from_voice_response(voice_response)
        trace = RuntimeLatencyTrace(
            session_id=self._session_id,
            patient_id=self._config.patient_id,
            turn_id=turn_id,
            traced_at=utc_now_iso(),
            turn_latency_ms=(perf_counter() - turn_started_at) * 1000,
            transcription_latency_ms=transcription.latency_ms,
            transcription_duration_ms=transcription.duration_ms,
            generation_latency_ms=response.generation.latency_ms,
            synthesis_latency_ms=voice_response.metrics.synthesis_latency_ms,
            playback_dispatch_latency_ms=voice_response.metrics.playback_dispatch_latency_ms,
            playback_completion_latency_ms=voice_response.metrics.playback_completion_latency_ms,
            end_to_end_latency_ms=voice_response.metrics.end_to_end_latency_ms,
            end_to_end_completion_latency_ms=voice_response.metrics.end_to_end_completion_latency_ms,
            audio_duration_ms=voice_response.metrics.audio_duration_ms,
            meets_targets=voice_response.meets_targets,
            guard_applied=response.trace.guard_applied,
            guard_reason=response.trace.guard_reason,
        )
        self._record_latency_trace(trace)

        self._emit(
            "assistant_replied",
            response.answer,
            turn_id=turn_id,
            payload={
                "end_to_end_latency_ms": voice_response.metrics.end_to_end_latency_ms,
                "generation_latency_ms": response.generation.latency_ms,
                "synthesis_latency_ms": voice_response.metrics.synthesis_latency_ms,
                "meets_targets": voice_response.meets_targets,
                "guard_applied": response.trace.guard_applied,
                "guard_reason": response.trace.guard_reason,
            },
        )

        return RuntimeTurn(
            patient_id=self._config.patient_id,
            user_text=user_text,
            assistant_text=response.answer,
            transcription=transcription,
            response=response,
            voice_response=voice_response,
        )

    def _append_history(self, user_text: str, assistant_text: str) -> None:
        if self._config.max_history_messages == 0:
            self._history.clear()
            return

        self._history.append(ConversationMessage(role="user", content=user_text))
        self._history.append(ConversationMessage(role="assistant", content=assistant_text))
        if len(self._history) > self._config.max_history_messages:
            self._history = self._history[-self._config.max_history_messages :]

    def _emit(
        self,
        event_type: str,
        detail: str,
        *,
        turn_id: str | None = None,
        level: str = "info",
        payload: dict[str, object] | None = None,
    ) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            patient_id=self._config.patient_id,
            session_id=self._session_id,
            turn_id=turn_id,
            recorded_at=utc_now_iso(),
            level=level,
            detail=detail,
            payload=payload,
        )
        if self._observer is not None:
            self._safe_observer_call(self._observer.on_event, event)
        if self._event_handler is None:
            return
        self._event_handler(event)

    def _start_session(self) -> None:
        self._session_id = uuid4().hex
        self._started_at = utc_now_iso()
        self._turn_index = 0
        self._frames_processed = 0
        self._turns_completed = 0
        self._turns_aborted = 0
        self._segments_skipped = 0
        self._transcripts_skipped = 0
        self._playback_interruptions = 0
        self._errors_total = 0
        self._alerts_total = 0
        self._last_error_stage = ""
        self._turn_latency_total_ms = 0.0
        self._transcription_latency_total_ms = 0.0
        self._generation_latency_total_ms = 0.0
        self._end_to_end_latency_total_ms = 0.0
        if self._observer is not None:
            self._safe_observer_call(
                self._observer.on_session_started,
                session_id=self._session_id,
                patient_id=self._config.patient_id,
                payload={
                    "runtime_config": {
                        "max_history_messages": self._config.max_history_messages,
                        "min_transcript_chars": self._config.min_transcript_chars,
                        "interrupt_playback_on_user_speech": self._config.interrupt_playback_on_user_speech,
                    }
                },
            )
            self._publish_metrics()

    def _notify_session_stopped(self) -> None:
        if not self._session_id or self._observer is None:
            return
        self._safe_observer_call(
            self._observer.on_session_stopped,
            session_id=self._session_id,
            patient_id=self._config.patient_id,
            payload={
                "frames_processed": self._frames_processed,
                "turns_completed": self._turns_completed,
                "turns_aborted": self._turns_aborted,
                "errors_total": self._errors_total,
                "alerts_total": self._alerts_total,
            },
        )

    def _next_turn_id(self) -> str:
        self._turn_index += 1
        return f"{self._session_id}:{self._turn_index:04d}"

    def _abort_turn(self, turn_id: str, detail: str) -> None:
        self._turns_aborted += 1
        self._publish_metrics()
        self._emit("turn_aborted", detail, turn_id=turn_id, level="warning")

    def _record_latency_trace(self, trace: RuntimeLatencyTrace) -> None:
        self._turns_completed += 1
        self._turn_latency_total_ms += trace.turn_latency_ms
        self._transcription_latency_total_ms += trace.transcription_latency_ms
        if trace.generation_latency_ms is not None:
            self._generation_latency_total_ms += trace.generation_latency_ms
        if trace.end_to_end_latency_ms is not None:
            self._end_to_end_latency_total_ms += trace.end_to_end_latency_ms

        if self._observer is not None:
            self._safe_observer_call(self._observer.on_latency_trace, trace)

        if trace.guard_applied:
            self._emit_alert(
                "grounding_guard_triggered",
                "warning",
                "Model response was replaced by the grounding guard.",
                turn_id=trace.turn_id,
                payload={"guard_reason": trace.guard_reason},
            )
        if not trace.meets_targets:
            self._emit_alert(
                "latency_target_missed",
                "warning",
                "Voice response latency targets were not met.",
                turn_id=trace.turn_id,
                payload={
                    "end_to_end_latency_ms": trace.end_to_end_latency_ms,
                    "turn_latency_ms": trace.turn_latency_ms,
                },
            )

        self._publish_metrics()

    def _record_error(
        self,
        stage: str,
        error: BaseException,
        *,
        turn_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._errors_total += 1
        self._last_error_stage = stage
        record = RuntimeErrorRecord(
            session_id=self._session_id,
            patient_id=self._config.patient_id,
            stage=stage,
            occurred_at=utc_now_iso(),
            error_type=type(error).__name__,
            error_message=str(error),
            turn_id=turn_id,
            payload=payload,
        )
        if self._observer is not None:
            self._safe_observer_call(self._observer.on_error, record)
        self._publish_metrics()
        self._emit(
            "runtime_error",
            f"{stage} failed: {error}",
            turn_id=turn_id,
            level="error",
            payload={
                "stage": stage,
                "error_type": type(error).__name__,
            },
        )
        self._emit_alert(
            "runtime_error",
            "critical",
            f"{stage} failed: {error}",
            turn_id=turn_id,
            payload={"stage": stage, "error_type": type(error).__name__},
        )

    def _emit_alert(
        self,
        alert_type: str,
        severity: str,
        detail: str,
        *,
        turn_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._alerts_total += 1
        alert = RuntimeAlert(
            session_id=self._session_id,
            patient_id=self._config.patient_id,
            alert_type=alert_type,
            severity=severity,
            emitted_at=utc_now_iso(),
            detail=detail,
            turn_id=turn_id,
            payload=payload,
        )
        if self._observer is not None:
            self._safe_observer_call(self._observer.on_alert, alert)
        self._publish_metrics()

    def _publish_metrics(self) -> None:
        if self._observer is None or not self._session_id:
            return
        snapshot = RuntimeMetricsSnapshot(
            session_id=self._session_id,
            patient_id=self._config.patient_id,
            started_at=self._started_at,
            updated_at=utc_now_iso(),
            frames_processed=self._frames_processed,
            turns_completed=self._turns_completed,
            turns_aborted=self._turns_aborted,
            segments_skipped=self._segments_skipped,
            transcripts_skipped=self._transcripts_skipped,
            playback_interruptions=self._playback_interruptions,
            errors_total=self._errors_total,
            alerts_total=self._alerts_total,
            average_turn_latency_ms=_average(self._turn_latency_total_ms, self._turns_completed),
            average_transcription_latency_ms=_average(
                self._transcription_latency_total_ms,
                self._turns_completed,
            ),
            average_generation_latency_ms=_average(
                self._generation_latency_total_ms,
                self._turns_completed,
            ),
            average_end_to_end_latency_ms=_average(
                self._end_to_end_latency_total_ms,
                self._turns_completed,
            ),
            last_error_stage=self._last_error_stage,
        )
        self._safe_observer_call(self._observer.on_metrics, snapshot)

    def _safe_observer_call(self, callback, *args, **kwargs) -> None:
        try:
            callback(*args, **kwargs)
        except Exception:
            return


def _normalize_history(initial_history: ConversationHistory) -> ConversationHistory:
    normalized_messages: list[ConversationMessage] = []
    for message in initial_history:
        if message.role == "system":
            raise ValueError("initial_history must not contain system messages")
        normalized_messages.append(message)
    return tuple(normalized_messages)


def _playback_state_from_voice_response(voice_response) -> _PlaybackState:
    playback = voice_response.playback
    if playback.blocking:
        return _PlaybackState()
    if playback.duration_ms <= 0:
        return _PlaybackState()
    return _PlaybackState(deadline_monotonic=perf_counter() + (playback.duration_ms / 1000))


def _average(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return round(total / count, 4)
