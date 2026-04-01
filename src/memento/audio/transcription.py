"""Whisper-style transcription adapters and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Protocol

from .io import write_wav_file
from .vad import EnergyVAD, SpeechSegment


@dataclass(frozen=True)
class WhisperConfig:
    """Settings for a Whisper-compatible transcription backend."""

    model_name: str = "whisper-base"
    language: str = "fr"
    prompt: str = ""
    min_segment_duration_ms: float = 150.0
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    vad_filter: bool = False

    def __post_init__(self) -> None:
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


class FasterWhisperBackend:
    """Real Whisper backend powered by faster-whisper."""

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

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        prompt: str = "",
    ) -> WhisperBackendResult:
        model = self._get_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=language or self._config.language,
            initial_prompt=prompt or self._config.prompt or None,
            beam_size=self._config.beam_size,
            word_timestamps=self._config.word_timestamps,
            condition_on_previous_text=self._config.condition_on_previous_text,
            vad_filter=self._config.vad_filter,
        )
        collected_segments = list(segments)
        text = " ".join(segment.text.strip() for segment in collected_segments if segment.text.strip())

        words: list[TranscribedWord] = []
        for segment in collected_segments:
            for word in getattr(segment, "words", []) or []:
                words.append(
                    TranscribedWord(
                        word=str(getattr(word, "word", "")).strip(),
                        start_ms=_seconds_to_ms(getattr(word, "start", None)),
                        end_ms=_seconds_to_ms(getattr(word, "end", None)),
                        probability=getattr(word, "probability", None),
                    )
                )

        confidence = _compute_confidence(words, info)
        return WhisperBackendResult(text=text, confidence=confidence, words=tuple(words))

    def _get_model(self):
        if self._model is None:
            module = self._import_faster_whisper()
            self._model = module.WhisperModel(
                self._config.model_name,
                device=self._config.device,
                compute_type=self._config.compute_type,
            )
        return self._model

    @staticmethod
    def _import_faster_whisper():
        try:
            return import_module("faster_whisper")
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run `uv sync` before using the real Whisper backend."
            ) from exc


def _seconds_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000


def _compute_confidence(words: list[TranscribedWord], info: object) -> float | None:
    probabilities = [word.probability for word in words if word.probability is not None]
    if probabilities:
        return round(sum(probabilities) / len(probabilities), 4)
    return getattr(info, "language_probability", None)
