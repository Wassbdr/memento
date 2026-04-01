"""Whisper-style transcription adapters and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import exp
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Protocol

from .io import write_wav_file
from .vad import EnergyVAD, SpeechSegment


DEFAULT_WHISPER_MODEL_NAME = "large-v3"
CUDA_RUNTIME_ERROR_MARKERS = (
    "attempting to deserialize object on a cuda device",
    "cublas64_12.dll",
    "cudnn",
    "cuda driver",
    "cuda runtime",
    "cuda is not available",
    "found no nvidia driver",
    "libcublas",
    "libcudnn",
    "torch.cuda.is_available() is false",
    "torch not compiled with cuda",
)
FFMPEG_ERROR_MARKERS = ("ffmpeg",)


@dataclass(frozen=True)
class WhisperConfig:
    """Settings for a Whisper-compatible transcription backend."""

    model_name: str = DEFAULT_WHISPER_MODEL_NAME
    language: str = "fr"
    prompt: str = ""
    min_segment_duration_ms: float = 150.0
    device: str = "cpu"
    fp16: bool = False
    beam_size: int = 5
    word_timestamps: bool = True
    condition_on_previous_text: bool = False

    def __post_init__(self) -> None:
        normalized_model_name = _normalize_model_name(self.model_name)
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")
        object.__setattr__(self, "model_name", normalized_model_name)
        normalized_device = self.device.strip().lower()
        if not normalized_device:
            raise ValueError("device must not be empty")
        object.__setattr__(self, "device", normalized_device)
        if self.min_segment_duration_ms < 0:
            raise ValueError("min_segment_duration_ms must be non-negative")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")


@dataclass(frozen=True)
class TranscribedWord:
    """Word-level transcription metadata."""

    word: str
    start_ms: float | None = None
    end_ms: float | None = None
    probability: float | None = None


@dataclass(frozen=True)
class WhisperBackendResult:
    """Normalized transcription response from the backend."""

    text: str
    confidence: float | None = None
    words: tuple[TranscribedWord, ...] = ()


class WhisperBackend(Protocol):
    """Backend contract implemented by a real Whisper runtime."""

    def transcribe(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        language: str,
        prompt: str,
    ) -> WhisperBackendResult:
        """Transcribe one speech segment."""


@dataclass(frozen=True)
class SegmentTranscription:
    """Transcription metadata for one speech segment."""

    text: str
    duration_ms: float
    latency_ms: float
    sample_rate_hz: int
    model_name: str
    confidence: float | None
    start_frame_index: int
    end_frame_index: int
    words: tuple[TranscribedWord, ...] = ()


class WhisperTranscriber:
    """Thin adapter around a Whisper-compatible backend."""

    def __init__(self, backend: WhisperBackend, config: WhisperConfig | None = None) -> None:
        self._backend = backend
        self._config = config or WhisperConfig()

    @property
    def config(self) -> WhisperConfig:
        return self._config

    def transcribe_segment(self, segment: SpeechSegment) -> SegmentTranscription | None:
        if segment.duration_ms < self._config.min_segment_duration_ms:
            return None

        started_at = perf_counter()
        backend_result = self._backend.transcribe(
            samples=segment.samples,
            sample_rate_hz=segment.frames[0].sample_rate_hz,
            language=self._config.language,
            prompt=self._config.prompt,
        )
        latency_ms = (perf_counter() - started_at) * 1000

        return SegmentTranscription(
            text=backend_result.text.strip(),
            duration_ms=segment.duration_ms,
            latency_ms=latency_ms,
            sample_rate_hz=segment.frames[0].sample_rate_hz,
            model_name=self._config.model_name,
            confidence=backend_result.confidence,
            start_frame_index=segment.start_frame_index,
            end_frame_index=segment.end_frame_index,
            words=backend_result.words,
        )


class WhisperTranscriptionPipeline:
    """Combine VAD segmentation and Whisper transcription."""

    def __init__(self, vad: EnergyVAD, transcriber: WhisperTranscriber) -> None:
        self._vad = vad
        self._transcriber = transcriber

    def transcribe_frames(self, frames: tuple) -> tuple[SegmentTranscription, ...]:
        segments = self._vad.segment(frames)
        results: list[SegmentTranscription] = []
        for segment in segments:
            transcription = self._transcriber.transcribe_segment(segment)
            if transcription is not None:
                results.append(transcription)
        return tuple(results)


class OpenAIWhisperBackend:
    """Real Whisper backend powered by the official openai-whisper package."""

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self._config = config or WhisperConfig()
        self._model = None

    @property
    def config(self) -> WhisperConfig:
        return self._config

    def transcribe(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        language: str,
        prompt: str,
    ) -> WhisperBackendResult:
        if sample_rate_hz != 16_000:
            with NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                temp_path = Path(temporary_file.name)

            try:
                write_wav_file(
                    path=temp_path,
                    samples=samples,
                    sample_rate_hz=sample_rate_hz,
                )
                return self.transcribe_file(temp_path, language=language, prompt=prompt)
            finally:
                temp_path.unlink(missing_ok=True)

        model = self._get_model()
        result = model.transcribe(
            _samples_to_numpy(samples),
            language=language or self._config.language,
            initial_prompt=prompt or self._config.prompt or None,
            beam_size=self._config.beam_size,
            word_timestamps=self._config.word_timestamps,
            condition_on_previous_text=self._config.condition_on_previous_text,
            fp16=self._config.fp16,
            verbose=False,
        )
        return self._normalize_result(result)

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        prompt: str = "",
    ) -> WhisperBackendResult:
        model = self._get_model()
        try:
            result = model.transcribe(
                str(audio_path),
                language=language or self._config.language,
                initial_prompt=prompt or self._config.prompt or None,
                beam_size=self._config.beam_size,
                word_timestamps=self._config.word_timestamps,
                condition_on_previous_text=self._config.condition_on_previous_text,
                fp16=self._config.fp16,
                verbose=False,
            )
        except FileNotFoundError as exc:
            if is_missing_ffmpeg_error(exc):
                raise RuntimeError(
                    "ffmpeg is required by openai-whisper but was not found in PATH. "
                    "Install ffmpeg, then retry."
                ) from exc
            raise

        return self._normalize_result(result)

    def _normalize_result(self, result: dict) -> WhisperBackendResult:
        collected_segments = list(result.get("segments", []) or [])
        text = str(result.get("text", "")).strip()
        if not text:
            text = " ".join(
                str(segment.get("text", "")).strip()
                for segment in collected_segments
                if str(segment.get("text", "")).strip()
            )

        words: list[TranscribedWord] = []
        for segment in collected_segments:
            for word in segment.get("words", []) or []:
                words.append(
                    TranscribedWord(
                        word=str(word.get("word", "")).strip(),
                        start_ms=_seconds_to_ms(word.get("start")),
                        end_ms=_seconds_to_ms(word.get("end")),
                        probability=word.get("probability"),
                    )
                )

        confidence = _compute_confidence(words, collected_segments)
        return WhisperBackendResult(text=text, confidence=confidence, words=tuple(words))

    def _get_model(self):
        if self._model is None:
            if self._config.device == "cuda" and not torch_cuda_available():
                raise RuntimeError(
                    "CUDA device requested but torch.cuda.is_available() is False. "
                    "Switch device to `cpu` or install a compatible CUDA runtime."
                )
            module = self._import_whisper()
            self._model = module.load_model(
                self._config.model_name,
                device=self._config.device,
            )
        return self._model

    @staticmethod
    def _import_whisper():
        try:
            return import_module("whisper")
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Run `uv sync` before using the real Whisper backend."
            ) from exc


def _seconds_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000


def _samples_to_numpy(samples: tuple[float, ...]):
    numpy = import_module("numpy")
    return numpy.asarray(samples, dtype=numpy.float32)


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    legacy_prefix = "whisper-"
    if normalized.startswith(legacy_prefix):
        return normalized[len(legacy_prefix) :]
    return normalized


def is_cuda_runtime_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in CUDA_RUNTIME_ERROR_MARKERS)


def torch_cuda_available() -> bool:
    try:
        torch = import_module("torch")
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def is_missing_ffmpeg_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in FFMPEG_ERROR_MARKERS)


def _compute_confidence(words: list[TranscribedWord], segments: list[dict]) -> float | None:
    probabilities = [word.probability for word in words if word.probability is not None]
    if probabilities:
        return round(sum(probabilities) / len(probabilities), 4)

    segment_probabilities = [
        _avg_logprob_to_probability(segment.get("avg_logprob"))
        for segment in segments
        if segment.get("avg_logprob") is not None
    ]
    segment_probabilities = [value for value in segment_probabilities if value is not None]
    if segment_probabilities:
        return round(sum(segment_probabilities) / len(segment_probabilities), 4)
    return None


def _avg_logprob_to_probability(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, exp(value)))
