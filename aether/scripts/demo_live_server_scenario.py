from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.getenv("AETHER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "demo_live_server_output.json"
WEBHOOK_TOKEN = os.getenv("AETHER_WEBHOOK_TOKEN", "demo-webhook-token")
DEFAULT_SYSTEM_PROMPT = (
    "You are Aether, a proactive meeting copilot. Save useful facts, remember people, "
    "and help the user before they ask."
)


def request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if path.startswith("/api/v1/webhook/"):
        headers["X-Webhook-Token"] = WEBHOOK_TOKEN

    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": True,
                "status": response.status,
                "body": json.loads(body) if body else None,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed: Any = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        return {
            "ok": False,
            "status": exc.code,
            "body": parsed,
        }


def choose_call_id(calls_response: dict[str, Any]) -> str | None:
    body = calls_response.get("body")
    if not isinstance(body, dict):
        return None
    calls = body.get("calls")
    if not isinstance(calls, list):
        return None

    active_statuses = {"in-progress", "ringing", "queued", "forwarding", "active"}
    for call in calls:
        if isinstance(call, dict) and call.get("id") and str(call.get("status", "")).lower() in active_statuses:
            return str(call["id"])
    return None


def main() -> None:
    system_prompt = os.getenv("AETHER_DEMO_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    results: dict[str, Any] = {
        "base_url": BASE_URL,
        "steps": {},
    }

    results["steps"]["health"] = request("GET", "/health")
    results["steps"]["vapi_provision"] = request(
        "POST",
        "/api/v1/vapi/provision",
        {
            "assistant_name": "Aether Demo",
            "system_prompt": system_prompt,
        },
    )
    results["steps"]["save_memory"] = request(
        "POST",
        "/api/v1/webhook/save_memory",
        {
            "message": {
                "type": "tool_calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "demo_live_tc_memory",
                            "arguments": {
                                "fact_to_remember": "Paul wants the survey results before the board meeting.",
                                "category": "meeting_context",
                            },
                        },
                    }
                ],
            }
        },
    )
    results["steps"]["vision_capture"] = request(
        "POST",
        "/api/v1/vision/capture",
        {
            "image_description": "Paul's business card is on the conference table next to the Q1 survey notes.",
        },
    )
    results["steps"]["context"] = request("GET", "/api/v1/user/context")
    results["steps"]["meeting_start"] = request(
        "POST",
        "/api/v1/trigger/meeting_start",
        {"attendee_name": "Paul"},
    )
    results["steps"]["vapi_calls"] = request("GET", "/api/v1/vapi/calls?limit=5")

    call_id = choose_call_id(results["steps"]["vapi_calls"])
    if call_id:
        results["steps"]["vapi_inject"] = request(
            "POST",
            "/api/v1/vapi/inject",
            {
                "call_id": call_id,
                "message": "Quiet reminder: Paul is waiting on the survey results.",
                "end_call_after": False,
            },
        )
    else:
        results["steps"]["vapi_inject"] = {
            "ok": False,
            "status": None,
            "body": {"status": "skipped", "message": "No call id available for injection."},
        }

    results["steps"]["transcript"] = request("GET", "/api/v1/transcript")
    results["steps"]["reports"] = request("GET", "/api/v1/reports")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
