import sys
import types
from dataclasses import dataclass
from pathlib import Path

from memento.audio import (
    AudioFrame,
    EnergyVAD,
    FasterWhisperBackend,
    SpeechSegment,
    TranscribedWord,
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
    words: tuple[TranscribedWord, ...] = (
        TranscribedWord(word="bonjour", start_ms=0.0, end_ms=250.0, probability=0.93),
        TranscribedWord(word="maman", start_ms=250.0, end_ms=500.0, probability=0.89),
    )

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
        return WhisperBackendResult(text=self.text, confidence=self.confidence, words=self.words)


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
    assert tuple(word.word for word in result.words) == ("bonjour", "maman")
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


def test_faster_whisper_backend_wraps_real_runtime_contract(monkeypatch) -> None:
    class FakeWord:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class FakeSegment:
        def __init__(self, text: str, words: list[FakeWord]) -> None:
            self.text = text
            self.words = words

    class FakeInfo:
        language_probability = 0.77

    class FakeWhisperModel:
        last_init: dict[str, object] | None = None
        transcribe_calls: list[dict[str, object]] = []

        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            type(self).last_init = {
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
            }

        def transcribe(self, audio_path: str, **kwargs):
            type(self).transcribe_calls.append({"audio_path": audio_path, **kwargs})
            assert Path(audio_path).exists()
            return iter(
                [
                    FakeSegment(
                        "bonjour maman",
                        [
                            FakeWord("bonjour", 0.0, 0.25, 0.95),
                            FakeWord("maman", 0.25, 0.5, 0.85),
                        ],
                    )
                ]
            ), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    backend = FasterWhisperBackend(
        WhisperConfig(
            model_name="small",
            language="fr",
            prompt="contexte",
            device="cpu",
            compute_type="int8",
            beam_size=3,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            min_segment_duration_ms=0,
        )
    )

    result = backend.transcribe(
        samples=(0.0, 0.1, -0.1, 0.0),
        sample_rate_hz=16_000,
        language="fr",
        prompt="bonjour",
    )

    assert FakeWhisperModel.last_init == {
        "model_name": "small",
        "device": "cpu",
        "compute_type": "int8",
    }
    assert FakeWhisperModel.transcribe_calls[0]["language"] == "fr"
    assert FakeWhisperModel.transcribe_calls[0]["initial_prompt"] == "bonjour"
    assert FakeWhisperModel.transcribe_calls[0]["beam_size"] == 3
    assert result.text == "bonjour maman"
    assert result.confidence == 0.9
    assert tuple(word.word for word in result.words) == ("bonjour", "maman")
