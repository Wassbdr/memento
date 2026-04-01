from dataclasses import dataclass

from memento.audio import (
    AudioFrame,
    EnergyVAD,
    SpeechSegment,
    VoiceActivityConfig,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriber,
    WhisperTranscriptionPipeline,
)


def build_frame(frame_index: int, amplitude: float) -> AudioFrame:
    return AudioFrame(
        samples=(amplitude, amplitude, amplitude, amplitude),
        sample_rate_hz=100,
        frame_index=frame_index,
    )


@dataclass
class FakeWhisperBackend:
    text: str = "bonjour maman"
    confidence: float = 0.91

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        language: str,
        prompt: str,
    ) -> WhisperBackendResult:
        self.calls.append(
            {
                "samples": samples,
                "sample_rate_hz": sample_rate_hz,
                "language": language,
                "prompt": prompt,
            }
        )
        return WhisperBackendResult(text=self.text, confidence=self.confidence)


def test_whisper_transcriber_calls_backend_and_returns_metadata() -> None:
    backend = FakeWhisperBackend()
    transcriber = WhisperTranscriber(
        backend=backend,
        config=WhisperConfig(
            model_name="whisper-small",
            language="fr",
            prompt="Contexte patient",
            min_segment_duration_ms=0,
        ),
    )
    segment = SpeechSegment(frames=(build_frame(3, 0.12), build_frame(4, 0.11)))

    result = transcriber.transcribe_segment(segment)

    assert result is not None
    assert result.text == "bonjour maman"
    assert result.model_name == "whisper-small"
    assert result.sample_rate_hz == 100
    assert result.start_frame_index == 3
    assert result.end_frame_index == 4
    assert result.confidence == 0.91
    assert result.latency_ms >= 0
    assert backend.calls[0]["language"] == "fr"
    assert backend.calls[0]["prompt"] == "Contexte patient"
    assert backend.calls[0]["samples"] == segment.samples


def test_whisper_transcriber_skips_segments_shorter_than_min_duration() -> None:
    backend = FakeWhisperBackend()
    transcriber = WhisperTranscriber(
        backend=backend,
        config=WhisperConfig(min_segment_duration_ms=100),
    )
    short_segment = SpeechSegment(frames=(build_frame(0, 0.1),))

    result = transcriber.transcribe_segment(short_segment)

    assert result is None
    assert backend.calls == []


def test_transcription_pipeline_uses_vad_segments() -> None:
    backend = FakeWhisperBackend(text="ou est mon manteau")
    vad = EnergyVAD(
        VoiceActivityConfig(
            speech_threshold=0.05,
            noise_floor=0.01,
            min_speech_frames=2,
            min_silence_frames=2,
            pre_roll_frames=0,
        )
    )
    transcriber = WhisperTranscriber(
        backend=backend,
        config=WhisperConfig(min_segment_duration_ms=0),
    )
    pipeline = WhisperTranscriptionPipeline(vad=vad, transcriber=transcriber)
    frames = (
        build_frame(0, 0.01),
        build_frame(1, 0.07),
        build_frame(2, 0.08),
        build_frame(3, 0.0),
        build_frame(4, 0.0),
    )

    results = pipeline.transcribe_frames(frames)

    assert len(results) == 1
    assert results[0].text == "ou est mon manteau"
    assert results[0].start_frame_index == 1
    assert results[0].end_frame_index == 2
