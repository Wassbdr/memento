"""Whisper-style transcription adapters and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .vad import EnergyVAD, SpeechSegment


@dataclass(frozen=True)
class WhisperConfig:
    """Settings for a Whisper-compatible transcription backend."""

    model_name: str = "whisper-base"
    language: str = "fr"
    prompt: str = ""
    min_segment_duration_ms: float = 150.0

    def __post_init__(self) -> None:
        if self.min_segment_duration_ms < 0:
            raise ValueError("min_segment_duration_ms must be non-negative")


@dataclass(frozen=True)
class WhisperBackendResult:
    """Normalized transcription response from the backend."""

    text: str
    confidence: float | None = None


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
