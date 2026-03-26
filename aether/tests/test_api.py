from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.config import DEMO_USER_ID, settings
from app.main import app
from app.services import memory_agent


class FakeMem0Client:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.memories: list[dict[str, Any]] = [
            {
                "id": "m1",
                "text": "User likes tea",
                "category": "preferences",
                "timestamp": "2026-03-14T00:00:00Z",
            }
        ]

    def add(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.add_calls.append({"messages": messages, "kwargs": kwargs})
        if messages and messages[0].get("content"):
            self.memories.insert(
                0,
                {
                    "id": f"m{len(self.memories) + 1}",
                    "text": messages[0]["content"],
                    "category": "general",
                    "timestamp": "2026-03-14T00:00:01Z",
                },
            )
        return {"id": "ok"}

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        if "filters" in kwargs and kwargs["filters"].get("user_id") == DEMO_USER_ID:
            return {"results": self.memories}
        if kwargs.get("user_id") == DEMO_USER_ID:
            return {"results": self.memories}
        raise ValueError("invalid filters")


def _reset_transcript() -> None:
    with memory_agent._transcript_lock:
        memory_agent._transcript.clear()


def test_get_transcript_empty() -> None:
    _reset_transcript()
    client = TestClient(app)

    response = client.get("/api/v1/transcript")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "messages": []}


def test_save_memory_webhook_and_transcript(monkeypatch: Any) -> None:
    _reset_transcript()
    fake_client = FakeMem0Client()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "save_user_memory"},
                    "toolCall": {
                        "id": "tc_1",
                        "arguments": {
                            "fact_to_remember": "User prefers async standups",
                            "category": "work_preferences",
                        },
                    },
                }
            ],
        }
    }

    response = client.post(
        "/api/v1/webhook/save_memory",
        json=payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [{"toolCallId": "tc_1", "result": "Mémoire enregistrée avec succès."}]
    }

    transcript_response = client.get("/api/v1/transcript")
    assert transcript_response.status_code == 200
    transcript = transcript_response.json()["messages"]
    assert transcript[0]["role"] == "assistant"
    assert "[Mémoire enregistrée :" in transcript[0]["text"]


def test_get_memories(monkeypatch: Any) -> None:
    _reset_transcript()
    fake_client = FakeMem0Client()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    response = client.get("/api/v1/memories")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["data"]) >= 1
    assert payload["data"][0]["text"] == "User likes tea"


def test_webhook_ignores_invalid_calls(monkeypatch: Any) -> None:
    _reset_transcript()
    fake_client = FakeMem0Client()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "other_tool"},
                    "toolCall": {"id": "x", "arguments": {"foo": "bar"}},
                }
            ],
        }
    }

    response = client.post(
        "/api/v1/webhook/save_memory",
        json=payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert response.status_code == 200
    assert response.json() == {"results": []}
