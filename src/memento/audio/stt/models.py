"""Shared audio data structures."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class AudioFrame:
    """A short chunk of mono or multi-channel normalized audio samples."""

    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int = 1
    frame_index: int = 0
    started_at_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if not self.samples:
            raise ValueError("samples must not be empty")
        if len(self.samples) % self.channels != 0:
            raise ValueError("samples length must be divisible by channels")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_ms(self) -> float:
        return (self.sample_count / self.channels) / self.sample_rate_hz * 1000

    @property
    def peak_level(self) -> float:
        return max(abs(sample) for sample in self.samples)

    @property
    def rms_level(self) -> float:
        mean_square = sum(sample * sample for sample in self.samples) / self.sample_count
        return sqrt(mean_square)
