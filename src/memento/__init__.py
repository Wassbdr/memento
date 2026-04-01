"""Memento package."""

from .audio import (
    AudioFrame,
    CaptureHealthReport,
    CaptureHealthSnapshot,
    EnergyVAD,
    MicrophoneConfig,
    RealTimeMicrophone,
    SegmentTranscription,
    SpeechSegment,
    VoiceActivityConfig,
    VoiceActivityDecision,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriber,
    WhisperTranscriptionPipeline,
)
from .todo import ProjectScope, build_default_scope

__all__ = [
    "AudioFrame",
    "CaptureHealthReport",
    "CaptureHealthSnapshot",
    "EnergyVAD",
    "MicrophoneConfig",
    "ProjectScope",
    "RealTimeMicrophone",
    "SegmentTranscription",
    "SpeechSegment",
    "VoiceActivityConfig",
    "VoiceActivityDecision",
    "WhisperBackendResult",
    "WhisperConfig",
    "WhisperTranscriber",
    "WhisperTranscriptionPipeline",
    "build_default_scope",
]
