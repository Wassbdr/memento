"""Text-to-speech adapters for the Memento voice stack."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol


DEFAULT_VOXTRAL_MODEL_NAME = "voxtral-tts-2603"
SUPPORTED_TTS_RESPONSE_FORMATS = ("pcm", "wav")


@dataclass(frozen=True)
class TextToSpeechConfig:
    """Settings for a text-to-speech backend."""

    model_name: str = DEFAULT_VOXTRAL_MODEL_NAME
    voice_id: str | None = None
    response_format: str = "wav"
    sample_rate_hz: int | None = None
    channels: int = 1
    api_key: str | None = None
    api_key_env_var: str = "MISTRAL_API_KEY"

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

        normalized_env_var = self.api_key_env_var.strip()
        if not normalized_env_var:
            raise ValueError("api_key_env_var must not be empty")
        object.__setattr__(self, "api_key_env_var", normalized_env_var)


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


class VoxtralTTSBackend:
    """Real TTS backend powered by Mistral's audio speech API."""

    def __init__(self, config: TextToSpeechConfig | None = None) -> None:
        self._config = config or TextToSpeechConfig()
        self._client = None

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
        payload: dict[str, object] = {
            "model": model_name,
            "input": text,
            "response_format": response_format,
        }
        if voice_id:
            payload["voice_id"] = voice_id
        if reference_audio_base64:
            payload["ref_audio"] = reference_audio_base64

        response = self._get_client().audio.speech.complete(**payload)
        audio_data = _lookup_response_value(response, "audio_data")
        if not isinstance(audio_data, str) or not audio_data.strip():
            raise RuntimeError("Voxtral TTS response did not include audio_data")

        sample_rate_hz = _lookup_optional_int(response, "sample_rate_hz", "sample_rate")
        channels = _lookup_optional_int(response, "channels") or self._config.channels

        return TextToSpeechBackendResult(
            audio_bytes=base64.b64decode(audio_data),
            response_format=response_format,
            sample_rate_hz=sample_rate_hz or self._config.sample_rate_hz,
            channels=channels,
        )

    def _get_client(self):
        if self._client is None:
            api_key = self._config.api_key or os.getenv(self._config.api_key_env_var)
            if not api_key:
                raise RuntimeError(
                    f"Missing Mistral API key. Set `{self._config.api_key_env_var}` or pass api_key in TextToSpeechConfig."
                )
            client_class = _import_mistral_client()
            self._client = client_class(api_key=api_key)
        return self._client


def _reference_audio_to_base64(reference_audio: bytes | str | Path | None) -> str | None:
    if reference_audio is None:
        return None
    if isinstance(reference_audio, bytes):
        return base64.b64encode(reference_audio).decode("ascii")
    path = Path(reference_audio)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _lookup_response_value(response: object, *names: str) -> object | None:
    for name in names:
        if hasattr(response, name):
            return getattr(response, name)
        if isinstance(response, dict) and name in response:
            return response[name]
    return None


def _lookup_optional_int(response: object, *names: str) -> int | None:
    value = _lookup_response_value(response, *names)
    if value is None:
        return None
    return int(value)


def _import_mistral_client():
    import_error: ImportError | None = None

    for module_name in ("mistralai.client", "mistralai"):
        try:
            module = import_module(module_name)
        except ImportError as exc:
            import_error = exc
            continue

        client_class = getattr(module, "Mistral", None)
        if client_class is not None:
            return client_class

    raise RuntimeError(
        "mistralai is not installed. Run `uv sync` before using the Voxtral TTS backend."
    ) from import_error
