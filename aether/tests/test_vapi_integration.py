from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient

from app.main import app
from app.services import vapi_client


async def _fake_say_in_call(call_id: str, message: str, *, end_call_after: bool = False) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "message": message,
        "end_call_after": end_call_after,
    }


async def _ended_call_details(call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "status": "ended",
        "controlUrl": "https://example.com/control",
    }


async def _active_call_details(call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "status": "in-progress",
        "controlUrl": "https://example.com/control",
    }


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"status": "ok"}
        self.text = "{}"

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.posts: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def post(self, url: str, headers: dict[str, str] | None = None, json: dict[str, Any] | None = None) -> _FakeResponse:
        self.posts.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(payload={"status": "ok"})


def test_vapi_inject_route_returns_success_for_active_call(monkeypatch: Any) -> None:
    monkeypatch.setattr(vapi_client, "say_in_call", _fake_say_in_call)

    client = TestClient(app)
    response = client.post(
        "/api/v1/vapi/inject",
        json={
            "call_id": "call-123",
            "message": "Quiet reminder for Paul.",
            "end_call_after": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Message injecté dans l'appel call-123.",
    }


def test_vapi_inject_route_rejects_empty_call_id() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/vapi/inject",
        json={
            "call_id": "   ",
            "message": "Quiet reminder for Paul.",
            "end_call_after": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "call_id cannot be empty"


def test_say_in_call_rejects_ended_calls(monkeypatch: Any) -> None:
    monkeypatch.setattr(vapi_client, "get_call", _ended_call_details)

    client = TestClient(app)
    response = client.post(
        "/api/v1/vapi/inject",
        json={
            "call_id": "call-ended",
            "message": "Quiet reminder for Paul.",
            "end_call_after": False,
        },
    )

    assert response.status_code == 409
    assert "not active" in response.json()["detail"]


def test_say_in_call_uses_control_url(monkeypatch: Any) -> None:
    fake_client = _FakeAsyncClient()

    monkeypatch.setattr(vapi_client, "get_call", _active_call_details)
    monkeypatch.setattr(vapi_client.httpx, "AsyncClient", lambda *args, **kwargs: fake_client)

    result = asyncio.run(
        vapi_client.say_in_call("call-live", "Injected message", end_call_after=False)
    )

    assert result == {"status": "ok"}
    assert fake_client.posts[0]["url"] == "https://example.com/control"
    assert fake_client.posts[0]["json"] == {
        "type": "say",
        "content": "Injected message",
        "endCallAfterSpoken": False,
    }
