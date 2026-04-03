"""Typed outputs and events emitted by the live runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from memento.audio import SegmentTranscription, VoiceResponseResult
from memento.conversation import ConversationMessage, ConversationResponse


@dataclass(frozen=True)
class RuntimeTurn:
    """One completed user-to-assistant exchange."""

    patient_id: str
    user_text: str
    assistant_text: str
    transcription: SegmentTranscription
    response: ConversationResponse
    voice_response: VoiceResponseResult


@dataclass(frozen=True)
class RuntimeEvent:
    """Small structured event emitted while the runtime loop is active."""

    event_type: str
    patient_id: str
    session_id: str = ""
    turn_id: str | None = None
    recorded_at: str = ""
    level: str = "info"
    detail: str = ""
    payload: dict[str, object] | None = None


RuntimeEventHandler = Callable[[RuntimeEvent], None]
ConversationHistory = tuple[ConversationMessage, ...]
