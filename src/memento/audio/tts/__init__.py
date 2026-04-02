"""Text-to-speech and playback building blocks."""

from .playback import (
    AudioOutputDevice,
    DecodedAudio,
    PlaybackResult,
    SoundDeviceOutput,
    SpeakerConfig,
    SpeakerPlayer,
    decode_audio_bytes,
)
from .synthesis import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    QwenTTSBackend,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SpeechSynthesizer,
    SynthesizedSpeech,
    TextToSpeechBackend,
    TextToSpeechBackendResult,
    TextToSpeechConfig,
)
from .voice import (
    VoiceExperienceMetrics,
    VoiceExperienceTargets,
    VoiceResponsePipeline,
    VoiceResponseResult,
)

__all__ = [
    "AudioOutputDevice",
    "DEFAULT_QWEN_TTS_LANGUAGE",
    "DEFAULT_QWEN_TTS_MODEL_NAME",
    "DEFAULT_QWEN_TTS_SPEAKER",
    "DecodedAudio",
    "PlaybackResult",
    "QwenTTSBackend",
    "SoundDeviceOutput",
    "SpeakerConfig",
    "SpeakerPlayer",
    "SUPPORTED_TTS_RESPONSE_FORMATS",
    "SpeechSynthesizer",
    "SynthesizedSpeech",
    "TextToSpeechBackend",
    "TextToSpeechBackendResult",
    "TextToSpeechConfig",
    "VoiceExperienceMetrics",
    "VoiceExperienceTargets",
    "VoiceResponsePipeline",
    "VoiceResponseResult",
    "decode_audio_bytes",
]
