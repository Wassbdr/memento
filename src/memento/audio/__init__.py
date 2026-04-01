"""Audio building blocks for the Memento voice stack."""

from .capture import (
    AudioInputDevice,
    CaptureHealthReport,
    CaptureHealthSnapshot,
    MicrophoneConfig,
    RealTimeMicrophone,
)
from .models import AudioFrame
from .transcription import (
    SegmentTranscription,
    WhisperBackend,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriptionPipeline,
    WhisperTranscriber,
)
from .vad import EnergyVAD, SpeechSegment, VoiceActivityConfig, VoiceActivityDecision

__all__ = [
    "AudioFrame",
    "AudioInputDevice",
    "CaptureHealthReport",
    "CaptureHealthSnapshot",
    "EnergyVAD",
    "MicrophoneConfig",
    "RealTimeMicrophone",
    "SegmentTranscription",
    "SpeechSegment",
    "VoiceActivityConfig",
    "VoiceActivityDecision",
    "WhisperBackend",
    "WhisperBackendResult",
    "WhisperConfig",
    "WhisperTranscriber",
    "WhisperTranscriptionPipeline",
]
