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
    DEFAULT_VOXTRAL_MODEL_NAME,
    SUPPORTED_TTS_RESPONSE_FORMATS,
    SpeechSynthesizer,
    SynthesizedSpeech,
    TextToSpeechBackend,
    TextToSpeechBackendResult,
    TextToSpeechConfig,
    VoxtralTTSBackend,
)
from .voice import (
    VoiceExperienceMetrics,
    VoiceExperienceTargets,
    VoiceResponsePipeline,
    VoiceResponseResult,
)

__all__ = [
    "AudioOutputDevice",
    "DEFAULT_VOXTRAL_MODEL_NAME",
    "DecodedAudio",
    "PlaybackResult",
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
    "VoxtralTTSBackend",
    "decode_audio_bytes",
]
