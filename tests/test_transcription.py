import sys
from dataclasses import dataclass

from memento.audio import (
    AudioFrame,
    EnergyVAD,
    OpenAIWhisperBackend,
    SpeechSegment,
    TranscribedWord,
    VoiceActivityConfig,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriber,
    WhisperTranscriptionPipeline,
    is_cuda_runtime_error,
    is_missing_ffmpeg_error,
    torch_cuda_available,
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
            model_name="small",
            language="fr",
            prompt="Contexte patient",
            min_segment_duration_ms=0,
        ),
    )
    segment = SpeechSegment(frames=(build_frame(3, 0.12), build_frame(4, 0.11)))

    result = transcriber.transcribe_segment(segment)

    assert result is not None
    assert result.text == "bonjour maman"
    assert result.model_name == "small"
    assert result.sample_rate_hz == 100
    assert result.start_frame_index == 3
    assert result.end_frame_index == 4
    assert result.confidence == 0.91
    assert tuple(word.word for word in result.words) == ("bonjour", "maman")
    assert result.latency_ms >= 0
    assert backend.calls[0]["language"] == "fr"
    assert backend.calls[0]["prompt"] == "Contexte patient"
    assert backend.calls[0]["samples"] == segment.samples


def test_whisper_config_defaults_to_large_v3() -> None:
    config = WhisperConfig()

    assert config.model_name == "large-v3"


def test_whisper_config_normalizes_legacy_prefixed_model_names() -> None:
    config = WhisperConfig(model_name="whisper-large-v3")

    assert config.model_name == "large-v3"


def test_detects_missing_cuda_runtime_errors() -> None:
    error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    assert is_cuda_runtime_error(error) is True


def test_detects_torch_cuda_deserialization_errors() -> None:
    error = RuntimeError(
        "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
    )

    assert is_cuda_runtime_error(error) is True


def test_ignores_non_cuda_runtime_errors() -> None:
    error = RuntimeError("openai-whisper is not installed")

    assert is_cuda_runtime_error(error) is False


def test_detects_missing_ffmpeg_errors() -> None:
    error = FileNotFoundError("[WinError 2] The system cannot find the file specified: 'ffmpeg'")

    assert is_missing_ffmpeg_error(error) is True


def test_torch_cuda_available_reads_torch_state(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    assert torch_cuda_available() is False


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


def test_openai_whisper_backend_wraps_real_runtime_contract(monkeypatch) -> None:
    class FakeWhisperModel:
        transcribe_calls: list[dict[str, object]] = []

        def transcribe(self, audio_path: str, **kwargs):
            type(self).transcribe_calls.append({"audio_path": audio_path, **kwargs})
            assert hasattr(audio_path, "shape")
            return {
                "text": "bonjour maman",
                "segments": [
                    {
                        "text": "bonjour maman",
                        "avg_logprob": -0.15,
                        "words": [
                            {"word": "bonjour", "start": 0.0, "end": 0.25, "probability": 0.95},
                            {"word": "maman", "start": 0.25, "end": 0.5, "probability": 0.85},
                        ],
                    }
                ],
            }

    class FakeWhisperModule:
        last_load_model: dict[str, object] | None = None

        @staticmethod
        def load_model(model_name: str, device: str) -> FakeWhisperModel:
            FakeWhisperModule.last_load_model = {
                "model_name": model_name,
                "device": device,
            }
            return FakeWhisperModel()

    monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule)

    backend = OpenAIWhisperBackend(
        WhisperConfig(
            model_name="large-v3",
            language="fr",
            prompt="contexte",
            device="cpu",
            fp16=False,
            beam_size=3,
            word_timestamps=True,
            condition_on_previous_text=False,
            min_segment_duration_ms=0,
        )
    )

    result = backend.transcribe(
        samples=(0.0, 0.1, -0.1, 0.0),
        sample_rate_hz=16_000,
        language="fr",
        prompt="bonjour",
    )

    assert FakeWhisperModule.last_load_model == {
        "model_name": "large-v3",
        "device": "cpu",
    }
    assert FakeWhisperModel.transcribe_calls[0]["language"] == "fr"
    assert FakeWhisperModel.transcribe_calls[0]["initial_prompt"] == "bonjour"
    assert FakeWhisperModel.transcribe_calls[0]["beam_size"] == 3
    assert FakeWhisperModel.transcribe_calls[0]["fp16"] is False
    assert result.text == "bonjour maman"
    assert result.confidence == 0.9
    assert tuple(word.word for word in result.words) == ("bonjour", "maman")
