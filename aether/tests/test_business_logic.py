from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

from app.api import endpoints
from app.models.schemas import MeetingStartPayload, WebhookPayload
from app.services import memory_agent

# ---------------------------------------------------------------------------
# 1) Vapi payload extraction logic
# ---------------------------------------------------------------------------


def _build_webhook_payload(raw: dict) -> WebhookPayload:
    return WebhookPayload.model_validate(raw)


def test_extract_first_valid_tool_call_perfect_payload() -> None:
    payload = _build_webhook_payload(
        {
            "message": {
                "type": "tool-calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "call-1",
                            "arguments": {
                                "fact_to_remember": "Sarah needs the survey",
                                "category": "meeting",
                            },
                        },
                    }
                ],
            }
        }
    )

    tool_call_id, fact, category = memory_agent._extract_first_valid_tool_call(payload)

    assert tool_call_id == "call-1"
    assert fact == "Sarah needs the survey"
    assert category == "meeting"


def test_extract_first_valid_tool_call_stringified_json_arguments() -> None:
    payload = _build_webhook_payload(
        {
            "message": {
                "type": "tool-calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "call-2",
                            "arguments": '{"fact_to_remember":"Budget review at 4 PM","category":"work"}',
                        },
                    }
                ],
            }
        }
    )

    tool_call_id, fact, category = memory_agent._extract_first_valid_tool_call(payload)

    assert tool_call_id == "call-2"
    assert fact == "Budget review at 4 PM"
    assert category == "work"


def test_extract_first_valid_tool_call_target_tool_in_second_item() -> None:
    payload = _build_webhook_payload(
        {
            "message": {
                "type": "tool-calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "check_calendar"},
                        "toolCall": {
                            "id": "calendar-1",
                            "arguments": {"fact_to_remember": "ignore me", "category": "general"},
                        },
                    },
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "call-3",
                            "arguments": {
                                "fact_to_remember": "Send proposal before Friday",
                                "category": "task",
                            },
                        },
                    },
                ],
            }
        }
    )

    tool_call_id, fact, category = memory_agent._extract_first_valid_tool_call(payload)

    assert tool_call_id == "call-3"
    assert fact == "Send proposal before Friday"
    assert category == "task"


def test_extract_first_valid_tool_call_missing_target_tool_graceful() -> None:
    payload = _build_webhook_payload(
        {
            "message": {
                "type": "tool-calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "check_calendar"},
                        "toolCall": {"id": "calendar-2", "arguments": {"foo": "bar"}},
                    }
                ],
            }
        }
    )

    tool_call_id, fact, category = memory_agent._extract_first_valid_tool_call(payload)

    assert tool_call_id is None
    assert fact is None
    assert category == "general"


# ---------------------------------------------------------------------------
# 2) Proactive string generation logic
# ---------------------------------------------------------------------------


def test_build_and_format_proactive_whisper_string() -> None:
    whisper = memory_agent._build_meeting_whisper(
        attendee_name="Sarah",
        selected_memory="Sarah needs the survey",
    )
    formatted = memory_agent._format_proactive_whisper_transcript(whisper)

    assert whisper == "Réunion avec Sarah. Rappel : Sarah needs the survey"
    assert (
        formatted
        == '[CHUCHOTEMENT PROACTIF INJECTÉ: "Réunion avec Sarah. Rappel : Sarah needs the survey"]'
    )


def test_trigger_meeting_start_uses_mem0_match_and_formats_transcript() -> None:
    payload = MeetingStartPayload(attendee_name="Sarah")

    mock_client = Mock()
    mock_client.get_all.return_value = {
        "results": [
            {
                "id": "m-1",
                "text": "Sarah needs the survey",
                "category": "meeting",
                "timestamp": "2026-03-14T10:00:00Z",
            }
        ]
    }

    with (
        patch("app.services.memory_agent._get_mem0_client", return_value=mock_client),
        patch("app.services.memory_agent._append_transcript") as append_mock,
    ):
        result = asyncio.run(endpoints.trigger_meeting_start(payload))

    assert result.status == "success"
    assert result.whisper_generated == "Réunion avec Sarah. Rappel : Sarah needs the survey"
    append_mock.assert_called_once_with(
        "assistant",
        '[CHUCHOTEMENT PROACTIF INJECTÉ: "Réunion avec Sarah. Rappel : Sarah needs the survey"]',
    )


def test_trigger_meeting_start_fallback_when_mem0_empty() -> None:
    payload = MeetingStartPayload(attendee_name="Sarah")

    mock_client = Mock()
    mock_client.get_all.return_value = {"results": []}

    with (
        patch("app.services.memory_agent._get_mem0_client", return_value=mock_client),
        patch("app.services.memory_agent._append_transcript") as append_mock,
    ):
        result = asyncio.run(endpoints.trigger_meeting_start(payload))

    fallback = "On lui a demandé les résultats du sondage il y a 4 heures."
    assert result.status == "success"
    assert result.whisper_generated == f"Réunion avec Sarah. Rappel : {fallback}"
    append_mock.assert_called_once_with(
        "assistant",
        f'[CHUCHOTEMENT PROACTIF INJECTÉ: "Réunion avec Sarah. Rappel : {fallback}"]',
    )


# ---------------------------------------------------------------------------
# 3) Memory normalization and sanitization logic
# ---------------------------------------------------------------------------


def test_sanitize_preview_removes_control_chars_and_truncates() -> None:
    noisy = "Hello\x00\x01\n\tWorld" + (" A" * 120)

    sanitized = memory_agent._sanitize_preview(noisy, limit=40)

    assert "\x00" not in sanitized
    assert "\n" not in sanitized
    assert "\t" not in sanitized
    assert len(sanitized) <= 40
    assert sanitized.endswith("...")


def test_sanitize_preview_handles_empty_string() -> None:
    assert memory_agent._sanitize_preview("", limit=30) == ""


def test_normalize_memories_handles_mixed_shapes() -> None:
    raw = {
        "results": [
            {
                "id": 101,
                "text": "Primary memory",
                "category": "voice",
                "timestamp": "2026-03-14T10:00:00Z",
            },
            {
                "memory_id": "mid-2",
                "memory": "From alternate key",
                "metadata": {"category": "work"},
                "created_at": "2026-03-14T10:10:00Z",
            },
            {
                "uuid": "u-3",
                "content": "From content key",
                "updated_at": "2026-03-14T10:20:00Z",
            },
            {
                "id": "skip-empty",
                "text": "",
            },
        ]
    }

    normalized = memory_agent._normalize_memories(raw)

    assert len(normalized) == 3

    assert normalized[0].id == "101"
    assert normalized[0].text == "Primary memory"
    assert normalized[0].category == "voice"
    assert normalized[0].timestamp == "2026-03-14T10:00:00Z"

    assert normalized[1].id == "mid-2"
    assert normalized[1].text == "From alternate key"
    assert normalized[1].category == "work"
    assert normalized[1].timestamp == "2026-03-14T10:10:00Z"

    assert normalized[2].id == "u-3"
    assert normalized[2].text == "From content key"
    assert normalized[2].category == "general"
    assert normalized[2].timestamp == "2026-03-14T10:20:00Z"


def test_normalize_memories_graceful_on_non_list_payload() -> None:
    assert memory_agent._normalize_memories(None) == []
    assert memory_agent._normalize_memories({"results": "invalid"}) == []
