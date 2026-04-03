"""Runtime configuration for the live end-to-end assistant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution settings for the live voice runtime."""

    patient_id: str
    max_history_messages: int = 6
    min_transcript_chars: int = 2
    interrupt_playback_on_user_speech: bool = True

    def __post_init__(self) -> None:
        normalized_patient_id = self.patient_id.strip()
        if not normalized_patient_id:
            raise ValueError("patient_id must not be empty")
        if self.max_history_messages < 0:
            raise ValueError("max_history_messages must be non-negative")
        if self.min_transcript_chars < 0:
            raise ValueError("min_transcript_chars must be non-negative")

        object.__setattr__(self, "patient_id", normalized_patient_id)
