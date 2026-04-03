"""CLI entry point for the live end-to-end runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from memento.audio import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    MicrophoneConfig,
    SpeakerConfig,
    TextToSpeechConfig,
    VoiceActivityConfig,
    WhisperConfig,
    torch_cuda_available,
)
from memento.conversation import (
    DEFAULT_MINISTRAL_MODEL_NAME,
    ConversationConfig,
    OpenAICompatibleBackendConfig,
)

from .bootstrap import build_live_runtime, load_snapshot_from_json_file
from .config import RuntimeConfig
from .observability import FileRuntimeObserver


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the argument parser used by `python -m memento.runtime`."""

    parser = argparse.ArgumentParser(description="Run the live Memento voice assistant.")
    parser.add_argument("--snapshot-file", required=True, help="Path to a patient snapshot JSON file.")
    parser.add_argument("--patient-id", help="Patient id to serve. Defaults to the snapshot patient id.")
    parser.add_argument("--llm-base-url", default=os.getenv("MEMENTO_LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("MEMENTO_LLM_API_KEY"))
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-model", default=DEFAULT_MINISTRAL_MODEL_NAME)
    parser.add_argument("--microphone-device", default="default")
    parser.add_argument("--speaker-device", default="default")
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--frame-duration-ms", type=int, default=30)
    parser.add_argument("--speech-threshold", type=float, default=0.04)
    parser.add_argument("--noise-floor", type=float, default=0.008)
    parser.add_argument("--min-speech-frames", type=int, default=2)
    parser.add_argument("--min-silence-frames", type=int, default=2)
    parser.add_argument("--pre-roll-frames", type=int, default=1)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--whisper-fp16", action="store_true")
    parser.add_argument("--tts-model", default=DEFAULT_QWEN_TTS_MODEL_NAME)
    parser.add_argument("--tts-speaker", default=DEFAULT_QWEN_TTS_SPEAKER)
    parser.add_argument("--tts-language", default=DEFAULT_QWEN_TTS_LANGUAGE)
    parser.add_argument("--tts-instruction", default=None)
    parser.add_argument("--tts-device-map", default="auto")
    parser.add_argument("--max-history-messages", type=int, default=6)
    parser.add_argument("--min-transcript-chars", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional finite loop bound for smoke tests.")
    parser.add_argument(
        "--observability-dir",
        default=os.getenv("MEMENTO_OBSERVABILITY_DIR"),
        help="Optional directory used to persist runtime logs, alerts, errors and metrics.",
    )
    parser.add_argument(
        "--disable-interrupt-on-speech",
        action="store_true",
        help="Do not interrupt current playback when new user speech is detected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live runtime from CLI arguments."""

    args = build_argument_parser().parse_args(argv)
    snapshot = load_snapshot_from_json_file(args.snapshot_file)
    patient_id = args.patient_id or snapshot.patient.patient_id
    whisper_device = args.whisper_device or ("cuda" if torch_cuda_available() else "cpu")
    observer = None
    if args.observability_dir:
        observer = FileRuntimeObserver(Path(args.observability_dir))

    runtime = build_live_runtime(
        snapshot=snapshot,
        runtime_config=RuntimeConfig(
            patient_id=patient_id,
            max_history_messages=args.max_history_messages,
            min_transcript_chars=args.min_transcript_chars,
            interrupt_playback_on_user_speech=not args.disable_interrupt_on_speech,
        ),
        conversation_config=ConversationConfig(model_name=args.llm_model),
        llm_backend_config=OpenAICompatibleBackendConfig(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout_seconds=args.llm_timeout_seconds,
        ),
        whisper_config=WhisperConfig(
            model_name=args.whisper_model,
            device=whisper_device,
            fp16=args.whisper_fp16,
        ),
        microphone_config=MicrophoneConfig(
            device_name=args.microphone_device,
            sample_rate_hz=args.sample_rate_hz,
            frame_duration_ms=args.frame_duration_ms,
        ),
        vad_config=VoiceActivityConfig(
            speech_threshold=args.speech_threshold,
            noise_floor=args.noise_floor,
            min_speech_frames=args.min_speech_frames,
            min_silence_frames=args.min_silence_frames,
            pre_roll_frames=args.pre_roll_frames,
        ),
        tts_config=TextToSpeechConfig(
            model_name=args.tts_model,
            voice_id=args.tts_speaker,
            language=args.tts_language,
            instruction=args.tts_instruction,
            device_map=args.tts_device_map,
            response_format="wav",
        ),
        speaker_config=SpeakerConfig(device_name=args.speaker_device),
        observer=observer,
    )
    try:
        runtime.run_forever(max_frames=args.max_frames)
    finally:
        runtime.close()
    return 0
