"""Speech-to-text building blocks."""

from .capture import (
    AudioInputDevice,
    CaptureHealthReport,
    CaptureHealthSnapshot,
    MicrophoneConfig,
    RealTimeMicrophone,
)
from .io import WavAudio, load_wav_bytes, speech_segment_from_wav_bytes, write_wav_file
from .live import InputDeviceInfo, SoundDeviceInput, StreamingSpeechSegmenter, list_input_devices
from .models import AudioFrame
from .transcription import (
    OpenAIWhisperBackend,
    SegmentTranscription,
    TranscribedWord,
    WhisperBackend,
    WhisperBackendResult,
    WhisperConfig,
    WhisperTranscriptionPipeline,
    WhisperTranscriber,
    is_cuda_runtime_error,
    is_missing_ffmpeg_error,
    torch_cuda_available,
)
from .vad import EnergyVAD, SpeechSegment, VoiceActivityConfig, VoiceActivityDecision

__all__ = [
    "AudioFrame",
    "AudioInputDevice",
    "CaptureHealthReport",
    "CaptureHealthSnapshot",
    "EnergyVAD",
    "InputDeviceInfo",
    "MicrophoneConfig",
    "OpenAIWhisperBackend",
    "RealTimeMicrophone",
    "SegmentTranscription",
    "SoundDeviceInput",
    "SpeechSegment",
    "StreamingSpeechSegmenter",
    "TranscribedWord",
    "VoiceActivityConfig",
    "VoiceActivityDecision",
    "WavAudio",
    "WhisperBackend",
    "WhisperBackendResult",
    "WhisperConfig",
    "WhisperTranscriber",
    "WhisperTranscriptionPipeline",
    "is_cuda_runtime_error",
    "is_missing_ffmpeg_error",
    "list_input_devices",
    "torch_cuda_available",
    "load_wav_bytes",
    "speech_segment_from_wav_bytes",
    "write_wav_file",
]
