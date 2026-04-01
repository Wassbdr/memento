"""Audio building blocks for the Memento voice stack."""

from .capture import (
    AudioInputDevice,
    CaptureHealthReport,
    CaptureHealthSnapshot,
    MicrophoneConfig,
    RealTimeMicrophone,
)
from .io import WavAudio, load_wav_bytes, speech_segment_from_wav_bytes, write_wav_file
from .models import AudioFrame
from .transcription import (
    FasterWhisperBackend,
    SegmentTranscription,
    TranscribedWord,
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
    "FasterWhisperBackend",
    "MicrophoneConfig",
    "RealTimeMicrophone",
    "SegmentTranscription",
    "SpeechSegment",
    "TranscribedWord",
    "VoiceActivityConfig",
    "VoiceActivityDecision",
    "WavAudio",
    "WhisperBackend",
    "WhisperBackendResult",
    "WhisperConfig",
    "WhisperTranscriber",
    "WhisperTranscriptionPipeline",
    "load_wav_bytes",
    "speech_segment_from_wav_bytes",
    "write_wav_file",
]
