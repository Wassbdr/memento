from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import memory_agent


class FailingMem0Client:
    def add(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mem0 add failure")

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mem0 read failure")


def _reset_transcript() -> None:
    with memory_agent._transcript_lock:
        memory_agent._transcript.clear()


def test_webhook_returns_success_on_mem0_failure(monkeypatch: Any) -> None:
    _reset_transcript()
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: FailingMem0Client())

    client = TestClient(app)
    payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "save_user_memory"},
                    "toolCall": {
                        "id": "tc_500",
                        "arguments": {
                            "fact_to_remember": "User is remote",
                            "category": "profile",
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
        "results": [{"toolCallId": "tc_500", "result": "Mémoire enregistrée avec succès."}]
    }


def test_memories_returns_500_on_mem0_failure(monkeypatch: Any) -> None:
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: FailingMem0Client())

    client = TestClient(app)
    response = client.get("/api/v1/memories")

    assert response.status_code == 500
    assert "Failed to fetch memories" in response.json()["detail"]


def test_webhook_token_protection(monkeypatch: Any) -> None:
    original_token = settings.webhook_token
    settings.webhook_token = "demo-secret"

    try:
        monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: FailingMem0Client())
        client = TestClient(app)

        payload = {
            "message": {
                "type": "tool_calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "tc_auth",
                            "arguments": {
                                "fact_to_remember": "User likes tea",
                                "category": "preferences",
                            },
                        },
                    }
                ],
            }
        }

        no_token_response = client.post("/api/v1/webhook/save_memory", json=payload)
        assert no_token_response.status_code == 401

        bad_token_response = client.post(
            "/api/v1/webhook/save_memory",
            json=payload,
            headers={"X-Webhook-Token": "wrong"},
        )
        assert bad_token_response.status_code == 401
    finally:
        settings.webhook_token = original_token


def test_check_calendar_webhook_returns_hardcoded_agenda() -> None:
    client = TestClient(app)
    payload = {
        "message": {
            "type": "tool_calls",
            "toolWithToolCallList": [
                {
                    "tool": {"name": "check_calendar"},
                    "toolCall": {"id": "tc_calendar", "arguments": {}},
                }
            ],
        }
    }

    response = client.post(
        "/api/v1/webhook/check_calendar",
        json=payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "toolCallId": "tc_calendar",
                "result": "Agenda du jour : 14h00 — réunion du conseil, 16h00 — rendez-vous dentiste.",
            }
        ]
    }


def test_call_ended_creates_report_and_reports_endpoint() -> None:
    with memory_agent._reports_lock:
        memory_agent._call_reports.clear()

    client = TestClient(app)
    payload = {
        "transcript": [
            {"role": "user", "text": "Plan my week"},
            {"role": "assistant", "text": "I identified several priorities."},
        ],
        "metrics": {"duration_seconds": 120},
    }

    call_ended_response = client.post(
        "/api/v1/webhook/call_ended",
        json=payload,
        headers={"X-Webhook-Token": settings.webhook_token},
    )
    assert call_ended_response.status_code == 200
    assert call_ended_response.json()["status"] == "success"

    reports_response = client.get("/api/v1/reports")
    assert reports_response.status_code == 200
    reports_payload = reports_response.json()
    assert reports_payload["status"] == "success"
    assert len(reports_payload["reports"]) >= 1
    assert reports_payload["reports"][0]["summary"] == "Call completed. 3 action items identified."


def test_context_endpoint_fallback_on_mem0_failure(monkeypatch: Any) -> None:
    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: FailingMem0Client())
    client = TestClient(app)

    response = client.get("/api/v1/user/context")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "Sois proactif sur ces sujets." in payload["injected_prompt"]
