import base64
import sys
from pathlib import Path

from memento.audio import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    QwenTTSBackend,
    SpeechSynthesizer,
    TextToSpeechBackendResult,
    TextToSpeechConfig,
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


def test_text_to_speech_config_defaults_to_qwen_custom_voice_model() -> None:
    config = TextToSpeechConfig()

    assert config.model_name == DEFAULT_QWEN_TTS_MODEL_NAME
    assert config.voice_id == DEFAULT_QWEN_TTS_SPEAKER
    assert config.language == DEFAULT_QWEN_TTS_LANGUAGE


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
            model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
            voice_id="Vivian",
            response_format="wav",
        ),
    )
    reference_path = tmp_path / "prompt.wav"
    reference_path.write_bytes(b"prompt")

    result = synthesizer.synthesize("Bonjour maman", reference_audio=reference_path)

    assert result.text == "Bonjour maman"
    assert result.model_name == DEFAULT_QWEN_TTS_MODEL_NAME
    assert result.voice_id == "Vivian"
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


def test_qwen_backend_uses_generate_custom_voice(monkeypatch) -> None:
    class FakeGeneratedAudio:
        def tolist(self):
            return [0.0, 0.25, -0.25]

    class FakeQwenModel:
        last_from_pretrained: dict[str, object] | None = None
        calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.last_from_pretrained = {"model_name": model_name, **kwargs}
            return cls()

        def generate_custom_voice(self, **kwargs):
            type(self).calls.append(kwargs)
            return [FakeGeneratedAudio()], 24_000

    class FakeQwenModule:
        Qwen3TTSModel = FakeQwenModel

    class FakeTorchModule:
        bfloat16 = "bf16-token"

    monkeypatch.setitem(sys.modules, "qwen_tts", FakeQwenModule)
    monkeypatch.setitem(sys.modules, "torch", FakeTorchModule)

    backend = QwenTTSBackend(
        TextToSpeechConfig(
            model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
            voice_id="Vivian",
            response_format="wav",
            language="French",
            instruction="Parle calmement.",
            device_map="cuda:0",
            dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
    )

    result = backend.synthesize(
        text="Bonjour",
        model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
        voice_id="Vivian",
        response_format="wav",
        reference_audio_base64=base64.b64encode(b"ignored").decode("ascii"),
    )

    assert FakeQwenModel.last_from_pretrained == {
        "model_name": DEFAULT_QWEN_TTS_MODEL_NAME,
        "device_map": "cuda:0",
        "dtype": "bf16-token",
        "attn_implementation": "flash_attention_2",
    }
    assert FakeQwenModel.calls[0] == {
        "text": "Bonjour",
        "language": "French",
        "speaker": "Vivian",
        "instruct": "Parle calmement.",
    }
    assert result.response_format == "wav"
    assert result.sample_rate_hz == 24_000
    assert result.channels == 1
    assert result.audio_bytes[:4] == b"RIFF"


def test_qwen_backend_rejects_non_wav_output(monkeypatch) -> None:
    class FakeQwenModule:
        class Qwen3TTSModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

    monkeypatch.setitem(sys.modules, "qwen_tts", FakeQwenModule)

    backend = QwenTTSBackend(TextToSpeechConfig(response_format="wav"))

    try:
        backend.synthesize(
            text="Bonjour",
            model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
            voice_id="Vivian",
            response_format="pcm",
            reference_audio_base64=None,
        )
    except ValueError as exc:
        assert "only `wav`" in str(exc)
    else:
        raise AssertionError("Expected ValueError when Qwen backend is asked for PCM output")


def test_qwen_backend_reuses_loaded_model_across_backend_instances(monkeypatch) -> None:
    class FakeGeneratedAudio:
        def tolist(self):
            return [0.0, 0.25, -0.25]

    class FakeQwenModel:
        from_pretrained_calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_calls += 1
            return cls()

        def generate_custom_voice(self, **kwargs):
            return [FakeGeneratedAudio()], 24_000

    class FakeQwenModule:
        Qwen3TTSModel = FakeQwenModel

    monkeypatch.setitem(sys.modules, "qwen_tts", FakeQwenModule)

    config = TextToSpeechConfig(
        model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
        voice_id="Vivian",
        response_format="wav",
        language="French",
        device_map="cpu",
    )
    first_backend = QwenTTSBackend(config)
    second_backend = QwenTTSBackend(config)

    first_backend.synthesize(
        text="Bonjour",
        model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
        voice_id="Vivian",
        response_format="wav",
        reference_audio_base64=None,
    )
    second_backend.synthesize(
        text="Rebonjour",
        model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
        voice_id="Vivian",
        response_format="wav",
        reference_audio_base64=None,
    )

    assert FakeQwenModel.from_pretrained_calls == 1
