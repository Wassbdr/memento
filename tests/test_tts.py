import base64
import sys
from pathlib import Path

from memento.audio import (
    DEFAULT_VOXTRAL_MODEL_NAME,
    SpeechSynthesizer,
    TextToSpeechBackendResult,
    TextToSpeechConfig,
    VoxtralTTSBackend,
)


class FakeTTSBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize(
        self,
        text: str,
        model_name: str,
        voice_id: str | None,
        response_format: str,
        reference_audio_base64: str | None,
    ) -> TextToSpeechBackendResult:
        self.calls.append(
            {
                "text": text,
                "model_name": model_name,
                "voice_id": voice_id,
                "response_format": response_format,
                "reference_audio_base64": reference_audio_base64,
            }
        )
        return TextToSpeechBackendResult(
            audio_bytes=b"wav-bytes",
            response_format=response_format,
            sample_rate_hz=24_000,
        )


def test_text_to_speech_config_defaults_to_latest_voxtral_model() -> None:
    config = TextToSpeechConfig()

    assert config.model_name == DEFAULT_VOXTRAL_MODEL_NAME


def test_text_to_speech_config_rejects_formats_without_local_playback_support() -> None:
    try:
        TextToSpeechConfig(response_format="mp3")
    except ValueError as exc:
        assert "response_format" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported compressed response format")


def test_speech_synthesizer_calls_backend_and_encodes_reference_audio(tmp_path: Path) -> None:
    backend = FakeTTSBackend()
    synthesizer = SpeechSynthesizer(
        backend=backend,
        config=TextToSpeechConfig(
            model_name="voxtral-tts-2603",
            voice_id="voice-preset",
            response_format="wav",
        ),
    )
    reference_path = tmp_path / "prompt.wav"
    reference_path.write_bytes(b"prompt")

    result = synthesizer.synthesize("Bonjour maman", reference_audio=reference_path)

    assert result.text == "Bonjour maman"
    assert result.model_name == "voxtral-tts-2603"
    assert result.voice_id == "voice-preset"
    assert result.response_format == "wav"
    assert result.sample_rate_hz == 24_000
    assert result.character_count == len("Bonjour maman")
    assert result.latency_ms >= 0
    assert backend.calls[0]["reference_audio_base64"] == base64.b64encode(b"prompt").decode("ascii")


def test_speech_synthesizer_requires_sample_rate_for_pcm() -> None:
    class PcmBackend(FakeTTSBackend):
        def synthesize(self, *args, **kwargs) -> TextToSpeechBackendResult:
            return TextToSpeechBackendResult(audio_bytes=b"\x00\x00\x00\x00", response_format="pcm")

    synthesizer = SpeechSynthesizer(
        backend=PcmBackend(),
        config=TextToSpeechConfig(response_format="pcm"),
    )

    try:
        synthesizer.synthesize("Bonjour")
    except ValueError as exc:
        assert "sample_rate_hz" in str(exc)
    else:
        raise AssertionError("Expected ValueError when PCM sample rate is missing")


def test_voxtral_backend_wraps_mistral_audio_speech(monkeypatch) -> None:
    class FakeSpeechClient:
        calls: list[dict[str, object]] = []

        def complete(self, **kwargs):
            type(self).calls.append(kwargs)
            return {
                "audio_data": base64.b64encode(b"voice-bytes").decode("ascii"),
                "sample_rate_hz": 24_000,
                "channels": 1,
            }

    class FakeAudioClient:
        speech = FakeSpeechClient()

    class FakeMistral:
        last_api_key: str | None = None

        def __init__(self, api_key: str) -> None:
            type(self).last_api_key = api_key
            self.audio = FakeAudioClient()

    class FakeMistralModule:
        Mistral = FakeMistral

    monkeypatch.setitem(sys.modules, "mistralai.client", FakeMistralModule)

    backend = VoxtralTTSBackend(
        TextToSpeechConfig(
            model_name="voxtral-tts-2603",
            response_format="wav",
            api_key="secret",
        )
    )

    result = backend.synthesize(
        text="Bonjour",
        model_name="voxtral-tts-2603",
        voice_id="voice-id",
        response_format="wav",
        reference_audio_base64=base64.b64encode(b"ref").decode("ascii"),
    )

    assert FakeMistral.last_api_key == "secret"
    assert FakeSpeechClient.calls[0]["model"] == "voxtral-tts-2603"
    assert FakeSpeechClient.calls[0]["input"] == "Bonjour"
    assert FakeSpeechClient.calls[0]["voice_id"] == "voice-id"
    assert FakeSpeechClient.calls[0]["ref_audio"] == base64.b64encode(b"ref").decode("ascii")
    assert result.audio_bytes == b"voice-bytes"
    assert result.sample_rate_hz == 24_000
