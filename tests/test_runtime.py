import json
from pathlib import Path

import pytest

from memento import (
    ConversationConfig,
    ConversationGeneration,
    ConversationMessage,
    ConversationOrchestrator,
    MemorySyncEngine,
    MicrophoneConfig,
    PlaybackResult,
    RealTimeMicrophone,
    RuntimeConfig,
    RuntimeTurn,
    SegmentTranscription,
    SpeakerConfig,
    SpeechSynthesizer,
    StreamingSpeechSegmenter,
    TextToSpeechBackendResult,
    TextToSpeechConfig,
    VoiceActivityConfig,
    VoiceExperienceTargets,
    VoiceResponsePipeline,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriber,
)
from memento.runtime import FileRuntimeObserver, MementoRuntime, load_snapshot_from_json_file
from memento.runtime import cli as runtime_cli

from memory_fixtures import build_snapshot


class FakeInputDevice:
    def __init__(self, frames: list[tuple[float, ...]]) -> None:
        self.frames = list(frames)
        self.open_calls = 0
        self.close_calls = 0

    def open(self, config: MicrophoneConfig) -> None:
        self.open_calls += 1

    def read(self, sample_count: int) -> tuple[float, ...]:
        frame = self.frames.pop(0)
        assert len(frame) == sample_count
        return frame

    def close(self) -> None:
        self.close_calls += 1


class CallbackInputDevice(FakeInputDevice):
    def __init__(self, frames: list[tuple[float, ...]], on_read=None) -> None:
        super().__init__(frames)
        self._on_read = on_read
        self._read_count = 0

    def read(self, sample_count: int) -> tuple[float, ...]:
        frame = super().read(sample_count)
        self._read_count += 1
        if self._on_read is not None:
            self._on_read(self._read_count)
        return frame


class FakeWhisperBackend:
    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = list(transcripts)

    def transcribe(self, samples, sample_rate_hz, language, prompt) -> WhisperBackendResult:
        return WhisperBackendResult(text=self.transcripts.pop(0), confidence=0.92)


class FakeConversationBackend:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[ConversationMessage, ...]] = []
        self.closed = False

    def generate(self, messages, *, model_name, temperature) -> ConversationGeneration:
        self.calls.append(messages)
        return ConversationGeneration(
            text=self.answers.pop(0),
            model_name=model_name,
            latency_ms=15.0,
            finish_reason="stop",
        )

    def close(self) -> None:
        self.closed = True


class StopDuringGenerateBackend(FakeConversationBackend):
    def __init__(self, answer: str, stop_runtime) -> None:
        super().__init__([answer])
        self._stop_runtime = stop_runtime

    def generate(self, messages, *, model_name, temperature) -> ConversationGeneration:
        self._stop_runtime()
        return super().generate(messages, model_name=model_name, temperature=temperature)


class FailingConversationBackend(FakeConversationBackend):
    def __init__(self) -> None:
        super().__init__([])

    def generate(self, messages, *, model_name, temperature) -> ConversationGeneration:
        raise RuntimeError("backend unavailable")


class FakeTTSBackend:
    def synthesize(
        self,
        text: str,
        model_name: str,
        voice_id: str | None,
        response_format: str,
        reference_audio_base64: str | None,
    ) -> TextToSpeechBackendResult:
        return TextToSpeechBackendResult(
            audio_bytes=b"wav-bytes",
            response_format="wav",
            sample_rate_hz=16_000,
            channels=1,
        )


class FakeVoicePlayer:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def play(self, speech) -> PlaybackResult:
        self.actions.append("play")
        return PlaybackResult(
            duration_ms=1_000.0,
            dispatch_latency_ms=10.0,
            completion_latency_ms=None,
            sample_rate_hz=16_000,
            channels=1,
            interrupted_previous=False,
            blocking=False,
        )

    def stop(self) -> bool:
        self.actions.append("stop")
        return True


def test_runtime_processes_one_turn_end_to_end_and_updates_history() -> None:
    runtime, conversation_backend, player = _build_runtime(
        frames=[
            _frame(0.0),
            _frame(0.08),
            _frame(0.08),
            _frame(0.09),
            _frame(0.0),
            _frame(0.0),
        ],
        transcripts=["Qui vient dimanche ?"],
        answers=["Claire vient dimanche."],
        interrupt_playback_on_user_speech=False,
    )
    events = []
    runtime_with_events = MementoRuntime(
        microphone=runtime._microphone,
        segmenter=runtime._segmenter,
        transcriber=runtime._transcriber,
        orchestrator=runtime._orchestrator,
        voice_pipeline=runtime._voice_pipeline,
        config=runtime.config,
        event_handler=events.append,
    )

    turns = runtime_with_events.run_forever(max_frames=6)

    assert len(turns) == 1
    assert isinstance(turns[0], RuntimeTurn)
    assert turns[0].user_text == "Qui vient dimanche ?"
    assert turns[0].assistant_text == "Claire vient dimanche."
    assert [message.role for message in runtime_with_events.conversation_history] == ["user", "assistant"]
    assert player.actions == ["play"]
    assert conversation_backend.calls
    assert [event.event_type for event in events] == [
        "runtime_started",
        "transcript_ready",
        "assistant_replied",
        "runtime_stopped",
    ]


def test_runtime_interrupts_playback_when_user_starts_speaking_again() -> None:
    runtime, _, player = _build_runtime(
        frames=[
            _frame(0.0),
            _frame(0.08),
            _frame(0.08),
            _frame(0.09),
            _frame(0.0),
            _frame(0.0),
            _frame(0.09),
            _frame(0.09),
            _frame(0.08),
            _frame(0.0),
            _frame(0.0),
        ],
        transcripts=["Qui vient dimanche ?", "Et demain ?"],
        answers=["Claire vient dimanche.", "Demain il n'y a rien de prevu."],
        interrupt_playback_on_user_speech=True,
    )

    turns = runtime.run_forever(max_frames=11)

    assert len(turns) == 2
    assert player.actions == ["play", "stop", "play"]


def test_runtime_stop_does_not_flush_and_process_pending_segment() -> None:
    runtime_holder: dict[str, MementoRuntime] = {}

    def stop_after_second_read(read_count: int) -> None:
        if read_count == 2:
            runtime_holder["runtime"].stop()

    microphone = RealTimeMicrophone(
        device=CallbackInputDevice(
            frames=[
                _frame(0.08),
                _frame(0.08),
            ],
            on_read=stop_after_second_read,
        ),
        config=MicrophoneConfig(
            device_name="fake-mic",
            sample_rate_hz=100,
            frame_duration_ms=40,
        ),
    )
    segmenter = StreamingSpeechSegmenter(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=0,
        )
    )
    transcriber = WhisperTranscriber(
        backend=FakeWhisperBackend(["Qui vient dimanche ?"]),
        config=WhisperConfig(min_segment_duration_ms=0.0),
    )
    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(build_snapshot())
    conversation_backend = FakeConversationBackend(["Claire vient dimanche."])
    player = FakeVoicePlayer()
    runtime = MementoRuntime(
        microphone=microphone,
        segmenter=segmenter,
        transcriber=transcriber,
        orchestrator=ConversationOrchestrator(
            memory_engine=memory_engine,
            backend=conversation_backend,
            config=ConversationConfig(max_prompt_memories=1),
        ),
        voice_pipeline=VoiceResponsePipeline(
            synthesizer=SpeechSynthesizer(
                backend=FakeTTSBackend(),
                config=TextToSpeechConfig(response_format="wav"),
            ),
            player=player,
            targets=VoiceExperienceTargets(
                max_synthesis_latency_ms=5_000.0,
                max_playback_dispatch_latency_ms=5_000.0,
                max_end_to_end_latency_ms=5_000.0,
                max_realtime_factor=50.0,
            ),
        ),
        config=RuntimeConfig(
            patient_id="rose",
            max_history_messages=4,
            min_transcript_chars=2,
            interrupt_playback_on_user_speech=False,
        ),
    )
    runtime_holder["runtime"] = runtime

    turns = runtime.run_forever(max_frames=10)

    assert turns == ()
    assert conversation_backend.calls == []
    assert player.actions == ["stop"]


def test_runtime_stop_during_generation_prevents_tts_playback() -> None:
    runtime, _, player = _build_runtime(
        frames=[
            _frame(0.0),
            _frame(0.08),
            _frame(0.08),
            _frame(0.09),
            _frame(0.0),
            _frame(0.0),
        ],
        transcripts=["Qui vient dimanche ?"],
        answers=["Claire vient dimanche."],
        interrupt_playback_on_user_speech=False,
    )
    stop_backend = StopDuringGenerateBackend("Claire vient dimanche.", runtime.stop)
    runtime._orchestrator = ConversationOrchestrator(
        memory_engine=runtime._orchestrator._memory_engine,
        backend=stop_backend,
        config=ConversationConfig(max_prompt_memories=1),
    )

    turns = runtime.run_forever(max_frames=6)

    assert turns == ()
    assert player.actions == ["stop"]


def test_runtime_persists_structured_observability_records(tmp_path: Path) -> None:
    runtime, _, _ = _build_runtime(
        frames=[
            _frame(0.0),
            _frame(0.08),
            _frame(0.08),
            _frame(0.09),
            _frame(0.0),
            _frame(0.0),
        ],
        transcripts=["Qui vient dimanche ?"],
        answers=["Claire vient dimanche."],
        interrupt_playback_on_user_speech=False,
    )
    runtime._observer = FileRuntimeObserver(tmp_path / "observability")

    turns = runtime.run_forever(max_frames=6)

    assert len(turns) == 1
    session_dir = tmp_path / "observability" / runtime.session_id
    metrics = json.loads((session_dir / "metrics.json").read_text(encoding="utf-8"))
    runtime_records = (session_dir / "runtime.jsonl").read_text(encoding="utf-8").splitlines()

    assert session_dir.exists()
    assert metrics["patient_id"] == "rose"
    assert metrics["turns_completed"] == 1
    assert metrics["errors_total"] == 0
    assert any('"record_type": "latency_trace"' in line for line in runtime_records)
    assert any('"record_type": "session_started"' in line for line in runtime_records)
    assert any('"record_type": "session_stopped"' in line for line in runtime_records)


def test_runtime_observability_tracks_errors_and_alerts(tmp_path: Path) -> None:
    runtime, _, _ = _build_runtime(
        frames=[
            _frame(0.0),
            _frame(0.08),
            _frame(0.08),
            _frame(0.09),
            _frame(0.0),
            _frame(0.0),
        ],
        transcripts=["Qui vient dimanche ?"],
        answers=["Claire vient dimanche."],
        interrupt_playback_on_user_speech=False,
    )
    runtime._orchestrator = ConversationOrchestrator(
        memory_engine=runtime._orchestrator._memory_engine,
        backend=FailingConversationBackend(),
        config=ConversationConfig(max_prompt_memories=1),
    )
    runtime._observer = FileRuntimeObserver(tmp_path / "observability")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        runtime.run_forever(max_frames=6)

    session_dir = tmp_path / "observability" / runtime.session_id
    metrics = json.loads((session_dir / "metrics.json").read_text(encoding="utf-8"))
    error_records = (session_dir / "errors.jsonl").read_text(encoding="utf-8")
    alert_records = (session_dir / "alerts.jsonl").read_text(encoding="utf-8")

    assert metrics["errors_total"] == 1
    assert metrics["alerts_total"] >= 1
    assert metrics["last_error_stage"] == "conversation_generation"
    assert "backend unavailable" in error_records
    assert '"alert_type": "runtime_error"' in alert_records


def test_load_snapshot_from_json_file_builds_typed_snapshot(tmp_path: Path) -> None:
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text(
        json.dumps(
            {
                "patient": {
                    "patient_id": "rose",
                    "display_name": "Rose Martin",
                    "preferred_name": "Mamie Rose",
                    "care_notes": ["Rassurer avant de recontextualiser."],
                    "anchors": ["Appartement rue des Lilas"],
                },
                "people": [
                    {
                        "person_id": "claire",
                        "name": "Claire Martin",
                        "relationship_to_patient": "sa fille",
                        "notes": "Vient le dimanche.",
                        "emotional_significance": 0.95,
                    }
                ],
                "places": [
                    {
                        "place_id": "cuisine",
                        "name": "Cuisine",
                        "category": "home_room",
                        "notes": "Table ronde.",
                    }
                ],
                "routines": [
                    {
                        "routine_id": "breakfast",
                        "title": "Petit-dejeuner",
                        "schedule": "Tous les jours a 08:00",
                        "description": "The et tartines.",
                        "cue": "La bouilloire siffle",
                        "support_strategy": "Montrer la tasse rouge.",
                        "place_id": "cuisine",
                    }
                ],
                "episodes": [
                    {
                        "episode_id": "dimanche",
                        "title": "Dejeuner du dimanche",
                        "narrative": "Claire apporte une tarte.",
                        "happened_on": "2024-01-07",
                        "people_ids": ["claire"],
                        "place_id": "cuisine",
                        "emotions": [
                            {"label": "joie", "valence": 0.9, "intensity": 0.8}
                        ],
                        "tags": ["famille"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_snapshot_from_json_file(snapshot_file)

    assert snapshot.patient.patient_id == "rose"
    assert snapshot.people[0].name == "Claire Martin"
    assert snapshot.episodes[0].emotions[0].label == "joie"


def test_runtime_cli_main_builds_and_runs_runtime(monkeypatch, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "patient": {
                    "patient_id": "rose",
                    "display_name": "Rose Martin",
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.max_frames = None
            self.closed = False

        def run_forever(self, *, max_frames=None):
            self.max_frames = max_frames
            return ()

        def close(self) -> None:
            self.closed = True

    fake_runtime = FakeRuntime()

    def fake_build_live_runtime(**kwargs):
        assert kwargs["runtime_config"].patient_id == "rose"
        assert kwargs["conversation_config"].model_name == "ministral-local"
        return fake_runtime

    monkeypatch.setattr(runtime_cli, "build_live_runtime", fake_build_live_runtime)

    exit_code = runtime_cli.main(
        [
            "--snapshot-file",
            str(snapshot_path),
            "--llm-model",
            "ministral-local",
            "--max-frames",
            "4",
        ]
    )

    assert exit_code == 0
    assert fake_runtime.max_frames == 4
    assert fake_runtime.closed is True


def _build_runtime(
    *,
    frames: list[tuple[float, ...]],
    transcripts: list[str],
    answers: list[str],
    interrupt_playback_on_user_speech: bool,
) -> tuple[MementoRuntime, FakeConversationBackend, FakeVoicePlayer]:
    microphone = RealTimeMicrophone(
        device=FakeInputDevice(frames),
        config=MicrophoneConfig(
            device_name="fake-mic",
            sample_rate_hz=100,
            frame_duration_ms=40,
        ),
    )
    segmenter = StreamingSpeechSegmenter(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=1,
        )
    )
    transcriber = WhisperTranscriber(
        backend=FakeWhisperBackend(transcripts),
        config=WhisperConfig(min_segment_duration_ms=0.0),
    )

    memory_engine = MemorySyncEngine()
    memory_engine.sync_snapshot(build_snapshot())
    conversation_backend = FakeConversationBackend(answers)
    orchestrator = ConversationOrchestrator(
        memory_engine=memory_engine,
        backend=conversation_backend,
        config=ConversationConfig(max_prompt_memories=1),
    )

    player = FakeVoicePlayer()
    voice_pipeline = VoiceResponsePipeline(
        synthesizer=SpeechSynthesizer(
            backend=FakeTTSBackend(),
            config=TextToSpeechConfig(response_format="wav"),
        ),
        player=player,
        targets=VoiceExperienceTargets(
            max_synthesis_latency_ms=5_000.0,
            max_playback_dispatch_latency_ms=5_000.0,
            max_end_to_end_latency_ms=5_000.0,
            max_realtime_factor=50.0,
        ),
    )

    runtime = MementoRuntime(
        microphone=microphone,
        segmenter=segmenter,
        transcriber=transcriber,
        orchestrator=orchestrator,
        voice_pipeline=voice_pipeline,
        config=RuntimeConfig(
            patient_id="rose",
            max_history_messages=4,
            min_transcript_chars=2,
            interrupt_playback_on_user_speech=interrupt_playback_on_user_speech,
        ),
    )
    return runtime, conversation_backend, player


def _frame(amplitude: float) -> tuple[float, ...]:
    return (amplitude, amplitude, amplitude, amplitude)
