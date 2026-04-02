"""Conversation model contracts shared by the orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


DEFAULT_MINISTRAL_MODEL_NAME = "Ministral 3 8B"
SUPPORTED_CONVERSATION_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class ConversationMessage:
    """One normalized chat message passed to a conversation model."""

    role: str
    content: str

    def __post_init__(self) -> None:
        normalized_role = self.role.strip().lower()
        if normalized_role not in SUPPORTED_CONVERSATION_ROLES:
            raise ValueError(
                f"role must be one of {', '.join(SUPPORTED_CONVERSATION_ROLES)}"
            )

        normalized_content = self.content.strip()
        if not normalized_content:
            raise ValueError("content must not be empty")

        object.__setattr__(self, "role", normalized_role)
        object.__setattr__(self, "content", normalized_content)


@dataclass(frozen=True)
class ConversationGeneration:
    """Normalized text generation returned by a conversation backend."""

    text: str
    model_name: str
    latency_ms: float | None = None
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        normalized_text = self.text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")

        normalized_model_name = self.model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative when provided")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative when provided")
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative when provided")

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "model_name", normalized_model_name)
        object.__setattr__(self, "finish_reason", self.finish_reason.strip())


class ConversationModelBackend(Protocol):
    """Backend contract for one chat-style text generation model."""

    def generate(
        self,
        messages: tuple[ConversationMessage, ...],
        *,
        model_name: str,
        temperature: float,
    ) -> ConversationGeneration:
        """Generate one assistant reply from a normalized prompt."""
