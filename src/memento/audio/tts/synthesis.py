"""Text-to-speech adapters for the Memento voice stack."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol
import wave


DEFAULT_QWEN_TTS_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_QWEN_TTS_LANGUAGE = "French"
DEFAULT_QWEN_TTS_SPEAKER = "Vivian"
SUPPORTED_TTS_RESPONSE_FORMATS = ("pcm", "wav")
_QWEN_MODEL_CACHE_LOCK = Lock()
_QWEN_MODEL_CACHE: dict[tuple[str, str | None, str | None, str | None], object] = {}


@dataclass(frozen=True)
class TextToSpeechConfig:
    """Settings for a text-to-speech backend."""

    model_name: str = DEFAULT_QWEN_TTS_MODEL_NAME
    voice_id: str | None = DEFAULT_QWEN_TTS_SPEAKER
    response_format: str = "wav"
    sample_rate_hz: int | None = None
    channels: int = 1
    language: str | None = DEFAULT_QWEN_TTS_LANGUAGE
    instruction: str | None = None
    device_map: str | None = None
    dtype: str | None = None
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        normalized_model_name = self.model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")
        object.__setattr__(self, "model_name", normalized_model_name)

        normalized_format = self.response_format.strip().lower()
        if normalized_format not in SUPPORTED_TTS_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {', '.join(SUPPORTED_TTS_RESPONSE_FORMATS)}"
            )
        object.__setattr__(self, "response_format", normalized_format)

        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.channels <= 0:
            raise ValueError("channels must be positive")

        object.__setattr__(self, "voice_id", _normalize_optional_string(self.voice_id))
        object.__setattr__(self, "language", _normalize_optional_string(self.language))
        object.__setattr__(self, "instruction", _normalize_optional_string(self.instruction))
        object.__setattr__(self, "device_map", _normalize_optional_string(self.device_map))
        object.__setattr__(self, "dtype", _normalize_optional_string(self.dtype))
        object.__setattr__(
            self,
            "attn_implementation",
            _normalize_optional_string(self.attn_implementation),
        )


@dataclass(frozen=True)
class TextToSpeechBackendResult:
    """Normalized payload returned by a TTS backend."""

    audio_bytes: bytes
    response_format: str
    sample_rate_hz: int | None = None
    channels: int = 1

    def __post_init__(self) -> None:
        if not self.audio_bytes:
            raise ValueError("audio_bytes must not be empty")
        normalized_format = self.response_format.strip().lower()
        if normalized_format not in SUPPORTED_TTS_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {', '.join(SUPPORTED_TTS_RESPONSE_FORMATS)}"
            )
        object.__setattr__(self, "response_format", normalized_format)
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.channels <= 0:
            raise ValueError("channels must be positive")


class TextToSpeechBackend(Protocol):
    """Backend contract for text-to-speech engines."""

    def synthesize(
        self,
        text: str,
        model_name: str,
        voice_id: str | None,
        response_format: str,
        reference_audio_base64: str | None,
    ) -> TextToSpeechBackendResult:
        """Synthesize one utterance."""


@dataclass(frozen=True)
class SynthesizedSpeech:
    """Final synthesis result returned by the project API."""

    text: str
    audio_bytes: bytes
    response_format: str
    sample_rate_hz: int | None
    channels: int
    latency_ms: float
    model_name: str
    voice_id: str | None
    character_count: int


class SpeechSynthesizer:
    """Thin adapter around a concrete TTS backend."""

    def __init__(
        self,
        backend: TextToSpeechBackend,
        config: TextToSpeechConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or TextToSpeechConfig()

    @property
    def config(self) -> TextToSpeechConfig:
        return self._config

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        reference_audio: bytes | str | Path | None = None,
    ) -> SynthesizedSpeech:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")

        started_at = perf_counter()
        backend_result = self._backend.synthesize(
            text=normalized_text,
            model_name=self._config.model_name,
            voice_id=voice_id or self._config.voice_id,
            response_format=self._config.response_format,
            reference_audio_base64=_reference_audio_to_base64(reference_audio),
        )
        latency_ms = (perf_counter() - started_at) * 1000

        sample_rate_hz = backend_result.sample_rate_hz or self._config.sample_rate_hz
        channels = backend_result.channels or self._config.channels
        if backend_result.response_format == "pcm" and sample_rate_hz is None:
            raise ValueError("sample_rate_hz is required to play or persist PCM audio")

        return SynthesizedSpeech(
            text=normalized_text,
            audio_bytes=backend_result.audio_bytes,
            response_format=backend_result.response_format,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            latency_ms=latency_ms,
            model_name=self._config.model_name,
            voice_id=voice_id or self._config.voice_id,
            character_count=len(normalized_text),
        )


class QwenTTSBackend:
    """Local TTS backend powered by the official `qwen-tts` Python package."""

    def __init__(self, config: TextToSpeechConfig | None = None) -> None:
        self._config = config or TextToSpeechConfig()
        self._model = None

    @property
    def config(self) -> TextToSpeechConfig:
        return self._config

    def synthesize(
        self,
        text: str,
        model_name: str,
        voice_id: str | None,
        response_format: str,
        reference_audio_base64: str | None,
    ) -> TextToSpeechBackendResult:
        if response_format != "wav":
            raise ValueError("QwenTTSBackend currently supports only `wav` output")

        speaker = voice_id or self._config.voice_id
        if speaker is None:
            raise ValueError("voice_id is required for Qwen CustomVoice generation")

        language = self._config.language or "Auto"
        generate_kwargs: dict[str, object] = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        if self._config.instruction is not None:
            generate_kwargs["instruct"] = self._config.instruction

        wavs, sample_rate_hz = self._get_model().generate_custom_voice(**generate_kwargs)
        samples, channels = _extract_generated_audio_payload(wavs)

        return TextToSpeechBackendResult(
            audio_bytes=_wav_bytes_from_samples(samples, int(sample_rate_hz), channels),
            response_format="wav",
            sample_rate_hz=int(sample_rate_hz),
            channels=channels,
        )

    def _get_model(self):
        if self._model is None:
            model_class = _import_qwen_tts_model()
            load_kwargs: dict[str, object] = {}
            resolved_device_map = _resolve_device_map(self._config.device_map)
            if resolved_device_map is not None:
                load_kwargs["device_map"] = resolved_device_map

            resolved_dtype = _resolve_torch_dtype(self._config.dtype)
            if resolved_dtype is not None:
                load_kwargs["dtype"] = resolved_dtype

            if self._config.attn_implementation is not None:
                load_kwargs["attn_implementation"] = self._config.attn_implementation

            cache_key = (
                self._config.model_name,
                resolved_device_map,
                self._config.dtype,
                self._config.attn_implementation,
            )
            with _QWEN_MODEL_CACHE_LOCK:
                cached_model = _QWEN_MODEL_CACHE.get(cache_key)
                if cached_model is None:
                    # Keep a single loaded Qwen model per process to avoid repeated heavyweight
                    # allocations across short-lived backend instances (for example Streamlit reruns).
                    _QWEN_MODEL_CACHE.clear()
                    cached_model = model_class.from_pretrained(self._config.model_name, **load_kwargs)
                    _QWEN_MODEL_CACHE[cache_key] = cached_model
                self._model = cached_model
        return self._model


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _reference_audio_to_base64(reference_audio: bytes | str | Path | None) -> str | None:
    if reference_audio is None:
        return None
    if isinstance(reference_audio, bytes):
        return base64.b64encode(reference_audio).decode("ascii")
    path = Path(reference_audio)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _resolve_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    torch = _import_torch()
    resolved_dtype = getattr(torch, dtype_name, None)
    if resolved_dtype is None:
        raise ValueError(f"Unsupported torch dtype `{dtype_name}`")
    return resolved_dtype


def _resolve_device_map(device_map: str | None) -> str | None:
    if device_map is None:
        torch = _import_torch()
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    normalized = device_map.strip().lower()
    if normalized == "auto":
        torch = _import_torch()
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_map


def _extract_generated_audio_payload(wavs: object) -> tuple[tuple[float, ...], int]:
    if not isinstance(wavs, list) or not wavs:
        raise RuntimeError("Qwen TTS did not return any audio sample")
    return _flatten_audio_array(wavs[0])


def _flatten_audio_array(audio: object) -> tuple[tuple[float, ...], int]:
    if hasattr(audio, "tolist"):
        audio = audio.tolist()

    if isinstance(audio, tuple):
        audio = list(audio)

    if not isinstance(audio, list) or not audio:
        raise RuntimeError("Unsupported Qwen audio payload")

    if isinstance(audio[0], list):
        first_row = audio[0]
        if not first_row:
            raise RuntimeError("Unsupported empty Qwen audio payload")

        if len(first_row) <= 8:
            channels = len(first_row)
            flattened = tuple(float(sample) for frame in audio for sample in frame)
            return flattened, channels

        if len(audio) <= 8:
            channels = len(audio)
            frame_count = len(first_row)
            flattened: list[float] = []
            for frame_index in range(frame_count):
                for channel_index in range(channels):
                    flattened.append(float(audio[channel_index][frame_index]))
            return tuple(flattened), channels

        raise RuntimeError("Unsupported multi-channel Qwen audio shape")

    return tuple(float(sample) for sample in audio), 1


def _wav_bytes_from_samples(
    samples: tuple[float, ...],
    sample_rate_hz: int,
    channels: int,
) -> bytes:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if not samples:
        raise ValueError("samples must not be empty")
    if len(samples) % channels != 0:
        raise ValueError("samples length must be divisible by channels")

    pcm_samples = [
        max(-32768, min(32767, int(round(sample * 32767.0))))
        for sample in samples
    ]
    pcm_bytes = b"".join(int(sample).to_bytes(2, byteorder="little", signed=True) for sample in pcm_samples)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _import_qwen_tts_model():
    try:
        module = import_module("qwen_tts")
    except ImportError as exc:
        raise RuntimeError(
            "qwen-tts is not installed. Run `uv sync` before using the Qwen TTS backend."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "qwen-tts could not load its native audio stack. This often means `torchaudio` is "
            "not aligned with the installed PyTorch build. Run `uv sync` so `torch` and "
            "`torchaudio` are reinstalled from the same index."
        ) from exc

    model_class = getattr(module, "Qwen3TTSModel", None)
    if model_class is None:
        raise RuntimeError("qwen_tts.Qwen3TTSModel is unavailable in the installed qwen-tts package")
    return model_class


def _import_torch():
    try:
        return import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is not installed. Run `uv sync` before using the Qwen TTS backend.") from exc
