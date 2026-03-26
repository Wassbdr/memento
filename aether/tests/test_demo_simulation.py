from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.config import DEMO_USER_ID, settings
from app.main import app
from app.services import memory_agent


class DemoMem0Client:
    def __init__(self) -> None:
        self.memories: list[dict[str, Any]] = []

    def add(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        content = ""
        if messages and isinstance(messages[0], dict):
            content_value = messages[0].get("content")
            if isinstance(content_value, str):
                content = content_value

        metadata = kwargs.get("metadata") if isinstance(kwargs, dict) else None
        category = "general"
        if isinstance(metadata, dict):
            category_value = metadata.get("category")
            if isinstance(category_value, str) and category_value.strip():
                category = category_value.strip()

        self.memories.insert(
            0,
            {
                "id": f"m{len(self.memories) + 1}",
                "text": content,
                "category": category,
                "timestamp": "2026-03-14T00:00:00Z",
            },
        )
        return {"id": "ok"}

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        if "filters" in kwargs and kwargs["filters"].get("user_id") == DEMO_USER_ID:
            return {"results": list(self.memories)}
        if kwargs.get("user_id") == DEMO_USER_ID:
            return {"results": list(self.memories)}
        raise ValueError("invalid filters")

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        valid_filters = "filters" in kwargs and kwargs["filters"].get("user_id") == DEMO_USER_ID
        valid_user_id = kwargs.get("user_id") == DEMO_USER_ID
        if valid_filters or valid_user_id:
            self.memories.clear()
            return {"status": "ok"}
        raise ValueError("invalid delete filters")


def _reset_state() -> None:
    with memory_agent._transcript_lock:
        memory_agent._transcript.clear()
    with memory_agent._reports_lock:
        memory_agent._call_reports.clear()


def test_demo_simulation_end_to_end(monkeypatch: Any) -> None:
    _reset_state()
    fake_client = DemoMem0Client()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)

    save_memory_payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "save_user_memory"},
                    "toolCall": {
                        "id": "demo_tc_1",
                        "arguments": {
                            "fact_to_remember": "Paul asked for the survey results before the meeting",
                            "category": "meeting_context",
                        },
                    },
                }
            ],
        }
    }
    save_memory_response = client.post(
        "/api/v1/webhook/save_memory",
        json=save_memory_payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert save_memory_response.status_code == 200
    assert save_memory_response.json() == {
        "results": [{"toolCallId": "demo_tc_1", "result": "Mémoire enregistrée avec succès."}]
    }

    vision_response = client.post(
        "/api/v1/vision/capture",
        json={"image_description": "Whiteboard note says present survey results to Paul"},
    )
    assert vision_response.status_code == 200
    assert vision_response.json() == {
        "status": "success",
        "message": "Contexte visuel extrait et enregistré.",
    }

    context_response = client.get("/api/v1/user/context")
    assert context_response.status_code == 200
    context_payload = context_response.json()
    assert context_payload["status"] == "success"
    assert "Paul" in context_payload["injected_prompt"]

    whisper_response = client.post(
        "/api/v1/trigger/meeting_start",
        json={"attendee_name": "Paul"},
    )
    assert whisper_response.status_code == 200
    whisper_payload = whisper_response.json()
    assert whisper_payload["status"] == "success"
    assert "Réunion avec Paul. Rappel :" in whisper_payload["whisper_generated"]

    calendar_payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "check_calendar"},
                    "toolCall": {"id": "demo_tc_calendar", "arguments": {}},
                }
            ],
        }
    }
    calendar_response = client.post(
        "/api/v1/webhook/check_calendar",
        json=calendar_payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert calendar_response.status_code == 200
    assert calendar_response.json()["results"][0]["toolCallId"] == "demo_tc_calendar"

    call_ended_response = client.post(
        "/api/v1/webhook/call_ended",
        json={
            "transcript": [
                {"role": "user", "text": "Do not let me forget the survey update for Paul"},
                {"role": "assistant", "text": "I will keep that in memory."},
            ],
            "metrics": {"duration_seconds": 95},
        },
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert call_ended_response.status_code == 200
    assert call_ended_response.json()["status"] == "success"

    reports_response = client.get("/api/v1/reports")
    assert reports_response.status_code == 200
    reports_payload = reports_response.json()
    assert reports_payload["status"] == "success"
    assert len(reports_payload["reports"]) >= 1

    transcript_response = client.get("/api/v1/transcript")
    assert transcript_response.status_code == 200
    transcript_payload = transcript_response.json()
    assert transcript_payload["status"] == "success"
    assert any("[Mémoire enregistrée :" in item["text"] for item in transcript_payload["messages"])
    assert any("[Contexte visuel enregistré :" in item["text"] for item in transcript_payload["messages"])
    assert any("[CHUCHOTEMENT PROACTIF INJECTÉ:" in item["text"] for item in transcript_payload["messages"])

    forget_response = client.delete("/api/v1/memories/forget")
    assert forget_response.status_code == 200
    assert forget_response.json() == {
        "status": "success",
        "message": "Toutes les mémoires ont été supprimées. État réinitialisé.",
    }

    memories_after_forget_response = client.get("/api/v1/memories")
    assert memories_after_forget_response.status_code == 200
    assert memories_after_forget_response.json()["data"] == []

    context_after_forget_response = client.get("/api/v1/user/context")
    assert context_after_forget_response.status_code == 200
    assert "Contexte connu : aucun" in context_after_forget_response.json()["injected_prompt"]
