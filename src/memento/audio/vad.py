"""Simple energy-based voice activity detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import AudioFrame


@dataclass(frozen=True)
class VoiceActivityConfig:
    """Configuration for the energy-based VAD."""

    speech_threshold: float = 0.04
    noise_floor: float = 0.008
    min_speech_frames: int = 2
    min_silence_frames: int = 2
    pre_roll_frames: int = 1

    def __post_init__(self) -> None:
        if self.speech_threshold <= 0:
            raise ValueError("speech_threshold must be positive")
        if self.noise_floor < 0:
            raise ValueError("noise_floor must be non-negative")
        if self.min_speech_frames <= 0:
            raise ValueError("min_speech_frames must be positive")
        if self.min_silence_frames <= 0:
            raise ValueError("min_silence_frames must be positive")
        if self.pre_roll_frames < 0:
            raise ValueError("pre_roll_frames must be non-negative")

    @property
    def effective_threshold(self) -> float:
        return max(self.speech_threshold, self.noise_floor * 3)


@dataclass(frozen=True)
class VoiceActivityDecision:
    """Frame-level classification metadata."""

    frame_index: int
    rms_level: float
    threshold: float
    is_speech: bool


@dataclass(frozen=True)
class SpeechSegment:
    """A contiguous speech segment produced by the VAD."""

    frames: tuple[AudioFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frames must not be empty")

    @property
    def duration_ms(self) -> float:
        return sum(frame.duration_ms for frame in self.frames)

    @property
    def samples(self) -> tuple[float, ...]:
        merged: list[float] = []
        for frame in self.frames:
            merged.extend(frame.samples)
        return tuple(merged)

    @property
    def start_frame_index(self) -> int:
        return self.frames[0].frame_index

    @property
    def end_frame_index(self) -> int:
        return self.frames[-1].frame_index


class EnergyVAD:
    """Detect speech segments from a stream of audio frames."""

    def __init__(self, config: VoiceActivityConfig | None = None) -> None:
        self._config = config or VoiceActivityConfig()

    @property
    def config(self) -> VoiceActivityConfig:
        return self._config

    def classify(self, frame: AudioFrame) -> VoiceActivityDecision:
        threshold = self._config.effective_threshold
        return VoiceActivityDecision(
            frame_index=frame.frame_index,
            rms_level=frame.rms_level,
            threshold=threshold,
            is_speech=frame.rms_level >= threshold,
        )

    def segment(self, frames: tuple[AudioFrame, ...] | list[AudioFrame]) -> tuple[SpeechSegment, ...]:
        segments: list[SpeechSegment] = []
        active_frames: list[AudioFrame] = []
        pending_speech_frames: list[AudioFrame] = []
        pending_silence_frames: list[AudioFrame] = []
        pre_roll = deque(maxlen=self._config.pre_roll_frames)

        for frame in frames:
            decision = self.classify(frame)
            if active_frames:
                if decision.is_speech:
                    if pending_silence_frames:
                        active_frames.extend(pending_silence_frames)
                        pending_silence_frames = []
                    active_frames.append(frame)
                else:
                    pending_silence_frames.append(frame)
                    if len(pending_silence_frames) >= self._config.min_silence_frames:
                        if active_frames:
                            segments.append(SpeechSegment(frames=tuple(active_frames)))
                        active_frames = []
                        pending_silence_frames = []
                continue

            if decision.is_speech:
                pending_speech_frames.append(frame)
                if len(pending_speech_frames) >= self._config.min_speech_frames:
                    active_frames = [*pre_roll, *pending_speech_frames]
                    pre_roll.clear()
                    pending_speech_frames = []
            else:
                pre_roll.append(frame)
                pending_speech_frames = []

        if active_frames:
            segments.append(SpeechSegment(frames=tuple(active_frames)))

        return tuple(segments)
