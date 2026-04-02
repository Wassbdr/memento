"""Real-time microphone capture primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import AudioFrame


@dataclass(frozen=True)
class MicrophoneConfig:
    """Configuration used to acquire audio frames from an input device."""

    device_name: str
    sample_rate_hz: int = 16_000
    channels: int = 1
    frame_duration_ms: int = 30
    silence_threshold: float = 0.01
    clipping_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if not 0 <= self.silence_threshold <= 1:
            raise ValueError("silence_threshold must be between 0 and 1")
        if not 0 < self.clipping_threshold <= 1:
            raise ValueError("clipping_threshold must be between 0 and 1")

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate_hz * (self.frame_duration_ms / 1000) * self.channels)


class AudioInputDevice(Protocol):
    """Protocol for a microphone-like device."""

    def open(self, config: MicrophoneConfig) -> None:
        """Prepare the device for reading audio samples."""

    def read(self, sample_count: int) -> tuple[float, ...]:
        """Return normalized samples in the range [-1.0, 1.0]."""

    def close(self) -> None:
        """Release device resources."""


@dataclass(frozen=True)
class CaptureHealthSnapshot:
    """Health metrics attached to one captured frame."""

    frame_index: int
    rms_level: float
    peak_level: float
    silent: bool
    clipped: bool


@dataclass(frozen=True)
class CaptureHealthReport:
    """Aggregate capture health metrics for the current session."""

    device_name: str
    total_frames: int
    silent_frames: int
    clipped_frames: int
    average_rms_level: float


class RealTimeMicrophone:
    """Stream fixed-size audio frames from a microphone-like device."""

    def __init__(self, device: AudioInputDevice, config: MicrophoneConfig) -> None:
        self._device = device
        self._config = config
        self._is_open = False
        self._frame_index = 0
        self._elapsed_ms = 0.0
        self._health_log: list[CaptureHealthSnapshot] = []

    @property
    def config(self) -> MicrophoneConfig:
        return self._config

    def start(self) -> None:
        if self._is_open:
            return
        self._device.open(self._config)
        self._is_open = True

    def stop(self) -> None:
        if not self._is_open:
            return
        self._device.close()
        self._is_open = False

    def stream(self, max_frames: int | None = None) -> tuple[AudioFrame, ...]:
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be positive or None")
        self.start()

        frames: list[AudioFrame] = []
        while max_frames is None or len(frames) < max_frames:
            frame = self.capture_frame()
            frames.append(frame)
            if max_frames is None:
                break
        return tuple(frames)

    def capture_frame(self) -> AudioFrame:
        self.start()
        samples = self._device.read(self._config.samples_per_frame)
        if len(samples) != self._config.samples_per_frame:
            raise ValueError("device returned an unexpected number of samples")

        frame = AudioFrame(
            samples=samples,
            sample_rate_hz=self._config.sample_rate_hz,
            channels=self._config.channels,
            frame_index=self._frame_index,
            started_at_ms=self._elapsed_ms,
        )
        self._frame_index += 1
        self._elapsed_ms += frame.duration_ms
        self._health_log.append(
            CaptureHealthSnapshot(
                frame_index=frame.frame_index,
                rms_level=frame.rms_level,
                peak_level=frame.peak_level,
                silent=frame.rms_level <= self._config.silence_threshold,
                clipped=frame.peak_level >= self._config.clipping_threshold,
            )
        )
        return frame

    def health_log(self) -> tuple[CaptureHealthSnapshot, ...]:
        return tuple(self._health_log)

    def health_report(self) -> CaptureHealthReport:
        total_frames = len(self._health_log)
        if total_frames == 0:
            average_rms_level = 0.0
        else:
            average_rms_level = sum(item.rms_level for item in self._health_log) / total_frames

        return CaptureHealthReport(
            device_name=self._config.device_name,
            total_frames=total_frames,
            silent_frames=sum(item.silent for item in self._health_log),
            clipped_frames=sum(item.clipped for item in self._health_log),
            average_rms_level=average_rms_level,
        )
