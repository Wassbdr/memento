"""Speaker playback helpers for synthesized speech."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import struct
from time import perf_counter
from typing import Protocol

from ..stt.io import load_wav_bytes
from .synthesis import SynthesizedSpeech


@dataclass(frozen=True)
class DecodedAudio:
    """Linear PCM payload ready for playback."""

    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int = 1

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
    def duration_ms(self) -> float:
        return (len(self.samples) / self.channels) / self.sample_rate_hz * 1000


@dataclass(frozen=True)
class SpeakerConfig:
    """Playback options for a speaker-like output device."""

    device_name: str = "default"
    blocking: bool = False
    interrupt_current: bool = True


@dataclass(frozen=True)
class PlaybackResult:
    """Observed playback metadata for one synthesized utterance."""

    duration_ms: float
    dispatch_latency_ms: float | None
    completion_latency_ms: float | None
    sample_rate_hz: int
    channels: int
    interrupted_previous: bool
    blocking: bool


class AudioOutputDevice(Protocol):
    """Protocol for a speaker-like device."""

    def play(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        channels: int,
        *,
        device_name: str,
        blocking: bool,
    ) -> None:
        """Play normalized PCM samples."""

    def stop(self) -> bool | None:
        """Interrupt playback and optionally report whether something was stopped."""


class SpeakerPlayer:
    """Decode synthesized audio and send it to an output device."""

    def __init__(self, device: AudioOutputDevice, config: SpeakerConfig | None = None) -> None:
        self._device = device
        self._config = config or SpeakerConfig()

    @property
    def config(self) -> SpeakerConfig:
        return self._config

    def play(self, speech: SynthesizedSpeech) -> PlaybackResult:
        decoded = decode_audio_bytes(
            audio_bytes=speech.audio_bytes,
            response_format=speech.response_format,
            sample_rate_hz=speech.sample_rate_hz,
            channels=speech.channels,
        )

        interrupted_previous = False
        if self._config.interrupt_current:
            interrupted_previous = bool(self._device.stop())

        started_at = perf_counter()
        self._device.play(
            samples=decoded.samples,
            sample_rate_hz=decoded.sample_rate_hz,
            channels=decoded.channels,
            device_name=self._config.device_name,
            blocking=self._config.blocking,
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        dispatch_latency_ms = None if self._config.blocking else elapsed_ms
        completion_latency_ms = elapsed_ms if self._config.blocking else None

        return PlaybackResult(
            duration_ms=decoded.duration_ms,
            dispatch_latency_ms=dispatch_latency_ms,
            completion_latency_ms=completion_latency_ms,
            sample_rate_hz=decoded.sample_rate_hz,
            channels=decoded.channels,
            interrupted_previous=interrupted_previous,
            blocking=self._config.blocking,
        )

    def stop(self) -> bool | None:
        return self._device.stop()


class SoundDeviceOutput:
    """Speaker output backed by the optional `sounddevice` package."""

    def play(
        self,
        samples: tuple[float, ...],
        sample_rate_hz: int,
        channels: int,
        *,
        device_name: str,
        blocking: bool,
    ) -> None:
        sounddevice = _import_sounddevice()
        device = None if device_name == "default" else device_name
        payload = _reshape_for_output(samples, channels)
        sounddevice.play(payload, samplerate=sample_rate_hz, device=device, blocking=blocking)

    def stop(self) -> None:
        _import_sounddevice().stop()


def decode_audio_bytes(
    audio_bytes: bytes,
    response_format: str,
    *,
    sample_rate_hz: int | None = None,
    channels: int = 1,
) -> DecodedAudio:
    """Decode a backend audio payload into playable PCM samples."""

    normalized_format = response_format.strip().lower()
    if normalized_format == "wav":
        wav_audio = load_wav_bytes(audio_bytes)
        return DecodedAudio(
            samples=wav_audio.samples,
            sample_rate_hz=wav_audio.sample_rate_hz,
            channels=wav_audio.channels,
        )

    if normalized_format == "pcm":
        if sample_rate_hz is None:
            raise ValueError("sample_rate_hz is required for PCM playback")
        if len(audio_bytes) % 4 != 0:
            raise ValueError("PCM float32 payload length must be divisible by 4")
        samples = tuple(value[0] for value in struct.iter_unpack("<f", audio_bytes))
        return DecodedAudio(samples=samples, sample_rate_hz=sample_rate_hz, channels=channels)

    raise ValueError("Only `wav` and `pcm` playback decoding are supported")


def _reshape_for_output(samples: tuple[float, ...], channels: int):
    if channels == 1:
        return list(samples)
    return [list(samples[index : index + channels]) for index in range(0, len(samples), channels)]


def _import_sounddevice():
    try:
        return import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "`sounddevice` is not installed. Run `uv sync` before using speaker playback."
        ) from exc
