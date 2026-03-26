"""Thin async wrapper around the Vapi REST API.

All functions raise ``fastapi.HTTPException`` on Vapi-side errors so the
calling route handler doesn't need to repeat error-handling boilerplate.
"""
from __future__ import annotations

import httpx
from fastapi import HTTPException

from app.core.config import API_PREFIX, _log_event, settings

_TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    if not settings.vapi_api_key:
        raise HTTPException(
            status_code=503,
            detail="VAPI_API_KEY is not configured — Vapi features are unavailable.",
        )
    return {
        "Authorization": f"Bearer {settings.vapi_api_key}",
        "Content-Type": "application/json",
    }


def _url(path: str) -> str:
    return f"{settings.vapi_base_url.rstrip('/')}/{path.lstrip('/')}"


def _raise_for_status(response: httpx.Response, context: str) -> None:
    if response.is_error:
        _log_event("vapi_api_error", context=context, status=response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Vapi API error [{context}] status={response.status_code}",
        )


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


async def list_calls(limit: int = 10) -> list[dict]:
    """Return the most-recent *limit* calls from the Vapi dashboard."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(_url("/call"), headers=_headers(), params={"limit": limit})
    _raise_for_status(r, "list_calls")
    data = r.json()
    # Vapi returns either a plain list or {"results": [...]}
    if isinstance(data, list):
        return data
    return data.get("results", data.get("calls", []))


async def get_call(call_id: str) -> dict:
    """Return full details for a single call."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(_url(f"/call/{call_id}"), headers=_headers())
    _raise_for_status(r, "get_call")
    return r.json()


# ---------------------------------------------------------------------------
# Assistants
# ---------------------------------------------------------------------------


def _build_assistant_payload(system_prompt: str, name: str = "Memento") -> dict:
    """Construct a full Vapi assistant configuration.

    Tools are bound to this backend's webhook endpoints so Vapi routes
    tool-call results directly to the right handler.
    """
    server_url = (settings.vapi_server_url or "").rstrip("/")
    save_memory_url = f"{server_url}{API_PREFIX}/webhook/save_memory"
    check_calendar_url = f"{server_url}{API_PREFIX}/webhook/check_calendar"
    call_ended_url = f"{server_url}{API_PREFIX}/webhook/call_ended"

    return {
        "name": name,
        "firstMessage": "Bonjour ! Je suis Memento, votre compagnon de mémoire. Comment puis-je vous aider ?",
        "model": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "save_user_memory",
                        "description": (
                            "Call this whenever the user shares a personal fact, preference, "
                            "or piece of information worth remembering for future conversations."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fact_to_remember": {
                                    "type": "string",
                                    "description": "The exact fact or piece of information to store.",
                                },
                                "category": {
                                    "type": "string",
                                    "description": (
                                        "Category label for the memory, e.g. 'personal', "
                                        "'work', 'preferences', 'health'."
                                    ),
                                },
                            },
                            "required": ["fact_to_remember", "category"],
                        },
                    },
                    "server": {"url": save_memory_url},
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_calendar",
                        "description": "Fetch the user's agenda and upcoming appointments for today.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                    "server": {"url": check_calendar_url},
                },
            ],
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "burt",
        },
        "server": {
            "url": call_ended_url,
            "timeoutSeconds": 20,
        },
        "serverMessages": ["end-of-call-report", "status-update"],
    }


async def create_assistant(system_prompt: str, name: str = "Memento") -> dict:
    """Provision a new Vapi assistant wired to this backend."""
    payload = _build_assistant_payload(system_prompt, name)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(_url("/assistant"), headers=_headers(), json=payload)
    _raise_for_status(r, "create_assistant")
    data = r.json()
    _log_event("vapi_assistant_created", assistant_id=data.get("id"), name=name)
    return data


async def update_assistant(assistant_id: str, system_prompt: str, name: str = "Memento") -> dict:
    """Re-configure an existing Vapi assistant (PATCH replaces top-level keys)."""
    payload = _build_assistant_payload(system_prompt, name)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.patch(
            _url(f"/assistant/{assistant_id}"), headers=_headers(), json=payload
        )
    _raise_for_status(r, "update_assistant")
    data = r.json()
    _log_event("vapi_assistant_updated", assistant_id=assistant_id)
    return data


# ---------------------------------------------------------------------------
# Mid-call injection
# ---------------------------------------------------------------------------


_ACTIVE_CALL_STATUSES = {"in-progress", "ringing", "queued", "forwarding", "active"}


async def say_in_call(call_id: str, message: str, *, end_call_after: bool = False) -> dict:
    """Inject *message* into an active call using Vapi's call control URL.

    Vapi exposes a per-call ``controlUrl`` for live call control actions. The
    current docs show "say" as a control action posted to that URL, not a
    REST endpoint under ``api.vapi.ai/call/{id}/say``.
    """
    call = await get_call(call_id)
    status = str(call.get("status") or "").lower()
    if status not in _ACTIVE_CALL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Call {call_id} is not active (status={status or 'unknown'}).",
        )

    control_url = call.get("monitor") and call["monitor"].get("controlUrl")
    if not control_url:
        control_url = call.get("controlUrl")
    if not isinstance(control_url, str) or not control_url.strip():
        raise HTTPException(
            status_code=409,
            detail=f"Call {call_id} does not expose a controlUrl for live control.",
        )

    payload: dict[str, object] = {
        "type": "say",
        "content": message,
        "endCallAfterSpoken": end_call_after,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            control_url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
    _raise_for_status(r, "say_in_call")
    _log_event("vapi_say_injected", call_id=call_id, preview=message[:60])
    return r.json() if r.text else {"status": "ok"}
