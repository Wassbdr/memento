from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.config import DEMO_USER_ID
from app.main import app
from app.services import memory_agent


class Mem0DeleteClient:
    def __init__(self) -> None:
        self.delete_all_calls: list[dict[str, Any]] = []

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_all_calls.append(kwargs)
        return {"status": "ok"}


class Mem0DeleteFallbackClient:
    def __init__(self) -> None:
        self.delete_all_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def delete_all(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_all_calls.append(kwargs)
        raise RuntimeError("delete_all unsupported")

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_calls.append(kwargs)
        return {"status": "ok"}


class Mem0AddAndSearchClient:
    def __init__(self, memories: list[dict[str, Any]] | None = None) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.memories = memories or []

    def add(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.add_calls.append({"messages": messages, "kwargs": kwargs})
        return {"id": "ok"}

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        if "filters" in kwargs and kwargs["filters"].get("user_id") == DEMO_USER_ID:
            return {"results": self.memories}
        if kwargs.get("user_id") == DEMO_USER_ID:
            return {"results": self.memories}
        raise ValueError("invalid filters")


class Mem0SearchFailClient:
    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mem0 read failure")


def _reset_transcript() -> None:
    with memory_agent._transcript_lock:
        memory_agent._transcript.clear()


def test_forget_memories_success(monkeypatch: Any) -> None:
    fake_client = Mem0DeleteClient()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    response = client.delete("/api/v1/memories/forget")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Toutes les mémoires ont été supprimées. État réinitialisé.",
    }
    assert fake_client.delete_all_calls[0] == {"filters": {"user_id": DEMO_USER_ID}}


def test_forget_memories_fallback_to_delete(monkeypatch: Any) -> None:
    fake_client = Mem0DeleteFallbackClient()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    response = client.delete("/api/v1/memories/forget")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(fake_client.delete_all_calls) == 2
    assert fake_client.delete_all_calls[0] == {"filters": {"user_id": DEMO_USER_ID}}
    assert fake_client.delete_all_calls[1] == {"user_id": DEMO_USER_ID}
    assert fake_client.delete_calls[0] == {"user_id": DEMO_USER_ID}


def test_vision_capture_saves_visual_context(monkeypatch: Any) -> None:
    _reset_transcript()
    fake_client = Mem0AddAndSearchClient()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    response = client.post(
        "/api/v1/vision/capture",
        json={"image_description": "Business card of Paul for Project Alpha"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Contexte visuel extrait et enregistré.",
    }

    assert len(fake_client.add_calls) == 1
    saved_message = fake_client.add_calls[0]["messages"][0]["content"]
    assert saved_message == "Visual context captured: Business card of Paul for Project Alpha"
    assert fake_client.add_calls[0]["kwargs"]["user_id"] == DEMO_USER_ID

    transcript_response = client.get("/api/v1/transcript")
    assert transcript_response.status_code == 200
    transcript = transcript_response.json()["messages"]
    assert "[Contexte visuel enregistré :" in transcript[0]["text"]


def test_meeting_start_generates_whisper_from_matching_memory(monkeypatch: Any) -> None:
    _reset_transcript()
    fake_client = Mem0AddAndSearchClient(
        memories=[
            {
                "id": "m1",
                "text": "Paul requested the latest survey rollout status.",
                "category": "work",
                "timestamp": "2026-03-14T00:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: fake_client)

    client = TestClient(app)
    response = client.post(
        "/api/v1/trigger/meeting_start",
        json={"attendee_name": "Paul"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "Réunion avec Paul. Rappel :" in payload["whisper_generated"]
    assert "Paul requested the latest survey rollout status." in payload["whisper_generated"]

    transcript_response = client.get("/api/v1/transcript")
    transcript = transcript_response.json()["messages"]
    assert "[CHUCHOTEMENT PROACTIF INJECTÉ:" in transcript[0]["text"]


def test_meeting_start_uses_fallback_memory_when_mem0_unavailable(monkeypatch: Any) -> None:
    _reset_transcript()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: Mem0SearchFailClient())

    client = TestClient(app)
    response = client.post(
        "/api/v1/trigger/meeting_start",
        json={"attendee_name": "Nora"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["whisper_generated"] == (
        "Réunion avec Nora. Rappel : On lui a demandé les résultats du sondage il y a 4 heures."
    )
