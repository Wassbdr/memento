"""Helpers to move between WAV bytes and normalized audio samples."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import wave

from .models import AudioFrame
from .vad import SpeechSegment


@dataclass(frozen=True)
class WavAudio:
    """Decoded mono audio payload from a WAV file."""

    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


def load_wav_bytes(audio_bytes: bytes) -> WavAudio:
    """Decode a PCM WAV payload into normalized float samples."""

    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width_bytes = wav_file.getsampwidth()
        sample_rate_hz = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width_bytes != 2:
        raise ValueError("Only 16-bit PCM WAV input is supported")

    int_samples = memoryview(raw_frames).cast("h")
    normalized = tuple(max(-1.0, min(1.0, sample / 32768.0)) for sample in int_samples)
    return WavAudio(
        samples=normalized,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_width_bytes=sample_width_bytes,
    )


def write_wav_file(
    path: str | Path,
    samples: tuple[float, ...],
    sample_rate_hz: int,
    channels: int = 1,
) -> Path:
    """Write normalized float samples to a 16-bit PCM WAV file."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if len(samples) % channels != 0:
        raise ValueError("samples length must be divisible by channels")

    output_path = Path(path)
    pcm_samples = [
        max(-32768, min(32767, int(round(sample * 32767.0))))
        for sample in samples
    ]
    pcm_bytes = b"".join(int(sample).to_bytes(2, byteorder="little", signed=True) for sample in pcm_samples)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)

    return output_path


def speech_segment_from_wav_bytes(audio_bytes: bytes) -> SpeechSegment:
    """Turn a WAV payload into a single speech segment for transcription."""

    wav_audio = load_wav_bytes(audio_bytes)
    frame = AudioFrame(
        samples=wav_audio.samples,
        sample_rate_hz=wav_audio.sample_rate_hz,
        channels=wav_audio.channels,
        frame_index=0,
        started_at_ms=0.0,
    )
    return SpeechSegment(frames=(frame,))
