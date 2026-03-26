# Global AI - Aether Demo Stack

Aether is a voice-first memory assistant demo built for live presentations and hackathons.

It contains:

- A FastAPI backend for transcript, memory, proactive whisper logic, and Vapi integrations.
- A React/Vite frontend with live polling, voice capture, alert center, and a knowledge graph.

## Repository Layout

- Backend app: [aether/app](aether/app)
  - API routes: [aether/app/api/endpoints.py](aether/app/api/endpoints.py)
  - App bootstrap: [aether/app/main.py](aether/app/main.py)
  - Memory logic: [aether/app/services/memory_agent.py](aether/app/services/memory_agent.py)
  - Vapi client: [aether/app/services/vapi_client.py](aether/app/services/vapi_client.py)
  - Config: [aether/app/core/config.py](aether/app/core/config.py)
- Backend tests: [aether/tests](aether/tests)
- Frontend app: [aether-ui](aether-ui)
  - App state/polling: [aether-ui/src/App.jsx](aether-ui/src/App.jsx)
  - Main UI components: [aether-ui/src/components](aether-ui/src/components)
- Demo runbook: [aether/DEMO_SCENARIO.md](aether/DEMO_SCENARIO.md)

## Core Demo Features

1. Voice capture reminder input (browser SpeechRecognition).
2. Memory persistence and retrieval via webhook + memory store.
3. Proactive context injection prompt for Vapi sessions (ambient copilot behavior).
4. Proactive whisper simulation via meeting-start trigger.
5. Distinct proactive insight rendering in transcript UI.
6. Alert strip, alert timeline, spoken voice alerts, and proactive chime notifications.
7. Hidden timer automation from spoken "remind me in/after" requests.
8. System Alert protocol for urgent proactive guidance.
9. Knowledge graph panel focused on clean voice memories.
10. One-click demo controls in header: `Simulate whisper` and `Demo reset`.

## Proactive Engine

- Backend context prompt now explicitly enforces ambient proactive behavior via [aether/app/api/endpoints.py](aether/app/api/endpoints.py).
- When memory categories include `task`, `meeting`, or `commitment`, the webhook response includes an internal directive to trigger immediate proactive follow-up.
- Proactive alerts include strict tone-shift instructions and require the spoken prefix `System Alert:`.
- Frontend transcript highlights proactive interventions with a dedicated `Aether Insight` card style in [aether-ui/src/components/TranscriptPanel.jsx](aether-ui/src/components/TranscriptPanel.jsx).
- Frontend plays a distinct proactive chime when new proactive actions are detected during transcript polling in [aether-ui/src/App.jsx](aether-ui/src/App.jsx).

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm
- Virtual environment in repository root (recommended): `.venv`

Optional for full external integrations:

- `MEM0_API_KEY`
- `VAPI_API_KEY`
- `VAPI_SERVER_URL`

Required for protected webhook routes:

- `WEBHOOK_TOKEN` (default fallback is `demo-webhook-token`)

## Environment Configuration

Create or update `.env` in [aether](aether):

```env
MEM0_API_KEY=
VAPI_API_KEY=
VAPI_SERVER_URL=
WEBHOOK_TOKEN=demo-webhook-token
EXTERNAL_SERVICES_MODE=auto
CORS_ORIGINS=http://localhost:5173,http://localhost:5174
```

Notes:

- If `WEBHOOK_TOKEN` is empty, backend normalizes to `demo-webhook-token`.
- Frontend webhook calls use `VITE_WEBHOOK_TOKEN` and default to `demo-webhook-token`.

## Run Backend

From [aether](aether):

```bash
cd /home/wassim/repos/global_ai/aether
source ../.venv/bin/activate
PYTHONPATH=/home/wassim/repos/global_ai/aether uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
```

## Run Frontend

From [aether-ui](aether-ui):

```bash
cd /home/wassim/repos/global_ai/aether-ui
npm install
npm run dev
```

Build check:

```bash
cd /home/wassim/repos/global_ai/aether-ui
node_modules/.bin/vite build
```

## Test and Lint

Backend:

```bash
cd /home/wassim/repos/global_ai/aether
source ../.venv/bin/activate
ruff check .
pytest -q
```

Frontend build sanity:

```bash
cd /home/wassim/repos/global_ai/aether-ui
node_modules/.bin/vite build
```

## API Quick Reference

Base prefix: `/api/v1`

Read endpoints:

- `GET /api/v1/transcript`
- `GET /api/v1/memories`
- `GET /api/v1/user/context`
- `GET /api/v1/reports`
- `GET /api/v1/vapi/calls?limit=10`

Mutation endpoints:

- `DELETE /api/v1/transcript`
- `DELETE /api/v1/memories/forget`
- `POST /api/v1/vision/capture`
- `POST /api/v1/trigger/meeting_start`
- `POST /api/v1/vapi/provision`
- `POST /api/v1/vapi/inject`

Webhook-protected endpoints (`X-Webhook-Token` required):

- `POST /api/v1/webhook/save_memory`
- `POST /api/v1/webhook/check_calendar`
- `POST /api/v1/webhook/call_ended`

## Hackathon Demo Flow (Recommended)

1. Start backend and frontend.
2. Click `Demo reset` in UI header.
3. Speak one reminder with a person name (example: Paul).
4. Click `Simulate whisper`.
5. Show:

   - alert strip
   - neural stream event
   - alert center timeline
   - cleaned knowledge graph memory card

## cURL Snippets

Save memory (token-protected):

```bash
curl -s -X POST http://localhost:8000/api/v1/webhook/save_memory \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Token: demo-webhook-token" \
  -d '{
    "message": {
      "type": "tool-calls",
      "toolWithToolCallList": [{
        "tool": {"name": "save_user_memory"},
        "toolCall": {
          "id": "demo-1",
          "arguments": {"fact_to_remember": "Paul prefers async updates", "category": "voice"}
        }
      }]
    }
  }'
```

Trigger proactive whisper:

```bash
curl -s -X POST http://localhost:8000/api/v1/trigger/meeting_start \
  -H "Content-Type: application/json" \
  -d '{"attendee_name": "Paul"}'
```

Reset state:

```bash
curl -s -X DELETE http://localhost:8000/api/v1/memories/forget
curl -s -X DELETE http://localhost:8000/api/v1/transcript
```

## Troubleshooting

Backend not reachable:

- Confirm port usage: `ss -ltnp | grep ':8000'`
- Ensure `uvicorn` uses `app.main:app` from [aether/app/main.py](aether/app/main.py)

Frontend says reconnecting forever:

- Confirm backend health endpoint responds.
- Check browser devtools for failed `/api/v1/*` calls.

Webhook returns 401/503:

- Ensure `WEBHOOK_TOKEN` is set and request includes `X-Webhook-Token`.

Stale demo data:

- Use `Demo reset` or the two DELETE endpoints above.

## Additional Documentation

- End-to-end scripted demo: [aether/DEMO_SCENARIO.md](aether/DEMO_SCENARIO.md)
