from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "scenario_output.json"
WEBHOOK_TOKEN = os.getenv("AETHER_WEBHOOK_TOKEN", "demo-webhook-token")


def request(method: str, path: str, payload: dict | None = None) -> dict:
    headers: dict[str, str] = {}
    if path.startswith("/api/v1/webhook/"):
        headers["X-Webhook-Token"] = WEBHOOK_TOKEN

    with TestClient(app) as client:
        response = client.request(method, path, json=payload, headers=headers)
    return {
        "status": response.status_code,
        "body": response.json(),
    }


results = {
    "health": request("GET", "/health"),
    "save_memory": request(
        "POST",
        "/api/v1/webhook/save_memory",
        {
            "message": {
                "type": "tool_calls",
                "toolWithToolCallList": [
                    {
                        "tool": {"name": "save_user_memory"},
                        "toolCall": {
                            "id": "scenario_tc_1",
                            "arguments": {
                                "fact_to_remember": "Paul wants the survey results before the board meeting",
                                "category": "meeting_context",
                            },
                        },
                    }
                ],
            }
        },
    ),
    "vision_capture": request(
        "POST",
        "/api/v1/vision/capture",
        {"image_description": "Business card for Paul found on the conference table"},
    ),
    "context": request("GET", "/api/v1/user/context"),
    "meeting_start": request(
        "POST",
        "/api/v1/trigger/meeting_start",
        {"attendee_name": "Paul"},
    ),
    "transcript": request("GET", "/api/v1/transcript"),
    "reports": request("GET", "/api/v1/reports"),
}

OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"wrote {OUTPUT_PATH}")
