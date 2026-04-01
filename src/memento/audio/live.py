"""Realtime microphone capture and incremental speech segmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from importlib import import_module

from .capture import AudioInputDevice, MicrophoneConfig
from .models import AudioFrame
from .vad import EnergyVAD, SpeechSegment, VoiceActivityConfig


@dataclass(frozen=True)
class InputDeviceInfo:
    """Metadata about one available microphone-like input device."""

    index: int
    name: str
    channels: int
    default_sample_rate_hz: float


class StreamingSpeechSegmenter:
    """Stateful version of the energy-based VAD for live audio streams."""

    def __init__(self, config: VoiceActivityConfig | None = None) -> None:
        self._vad = EnergyVAD(config=config)
        self._active_frames: list[AudioFrame] = []
        self._pending_speech_frames: list[AudioFrame] = []
        self._pending_silence_frames: list[AudioFrame] = []
        self._pre_roll = deque(maxlen=self._vad.config.pre_roll_frames)

    @property
    def config(self) -> VoiceActivityConfig:
        return self._vad.config

    def push_frame(self, frame: AudioFrame) -> SpeechSegment | None:
        decision = self._vad.classify(frame)

        if self._active_frames:
            if decision.is_speech:
                if self._pending_silence_frames:
                    self._active_frames.extend(self._pending_silence_frames)
                    self._pending_silence_frames = []
                self._active_frames.append(frame)
                return None

            self._pending_silence_frames.append(frame)
            if len(self._pending_silence_frames) >= self._vad.config.min_silence_frames:
                segment = SpeechSegment(frames=tuple(self._active_frames))
                self._active_frames = []
                self._pending_silence_frames = []
                return segment
            return None

        if decision.is_speech:
            self._pending_speech_frames.append(frame)
            if len(self._pending_speech_frames) >= self._vad.config.min_speech_frames:
                self._active_frames = [*self._pre_roll, *self._pending_speech_frames]
                self._pre_roll.clear()
                self._pending_speech_frames = []
            return None

        self._pre_roll.append(frame)
        self._pending_speech_frames = []
        return None

    def flush(self) -> SpeechSegment | None:
        if not self._active_frames:
            return None
        segment = SpeechSegment(frames=tuple(self._active_frames))
        self._active_frames = []
        self._pending_speech_frames = []
        self._pending_silence_frames = []
        self._pre_roll.clear()
        return segment


class SoundDeviceInput(AudioInputDevice):
    """Microphone input backed by the optional `sounddevice` package."""

    def __init__(self) -> None:
        self._stream = None
        self._channels = 1

    def open(self, config: MicrophoneConfig) -> None:
        sounddevice = _import_sounddevice()
        frames_per_read = config.samples_per_frame // config.channels
        device = None if config.device_name == "default" else config.device_name
        self._stream = sounddevice.InputStream(
            samplerate=config.sample_rate_hz,
            blocksize=frames_per_read,
            channels=config.channels,
            dtype="float32",
            device=device,
        )
        self._stream.start()
        self._channels = config.channels

    def read(self, sample_count: int) -> tuple[float, ...]:
        if self._stream is None:
            raise RuntimeError("input stream is not open")
        frame_count = sample_count // self._channels
        samples, overflowed = self._stream.read(frame_count)
        if overflowed:
            raise RuntimeError("microphone input overflowed while reading live audio")
        return tuple(float(value) for value in samples.reshape(-1))

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None


def list_input_devices() -> tuple[InputDeviceInfo, ...]:
    """Return available input devices exposed by `sounddevice`."""

    sounddevice = _import_sounddevice()
    devices = []
    for index, device in enumerate(sounddevice.query_devices()):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            InputDeviceInfo(
                index=index,
                name=str(device.get("name", f"input-{index}")),
                channels=int(device.get("max_input_channels", 0)),
                default_sample_rate_hz=float(device.get("default_samplerate", 0.0)),
            )
        )
    return tuple(devices)


def _import_sounddevice():
    try:
        return import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "`sounddevice` is not installed. Run `uv sync` before using the live microphone tool."
        ) from exc

