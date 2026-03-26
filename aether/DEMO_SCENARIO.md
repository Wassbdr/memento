# Demo Scenario

This scenario is designed for the live FastAPI server running on port 8000.

## Preconditions

- Backend is running.
- `.env` contains real `MEM0_API_KEY` and `VAPI_API_KEY`.
- `.env` contains `WEBHOOK_TOKEN=demo-webhook-token` (or set `AETHER_WEBHOOK_TOKEN`).
- `VAPI_SERVER_URL` points to your public backend URL.
- A live Vapi call is optional. If none exists, the inject step is skipped.

## Run the demo scenario

```bash
cd /home/wassim/repos/global_ai/aether
PYTHONPATH=/home/wassim/repos/global_ai/aether \
/home/wassim/repos/global_ai/.venv/bin/python scripts/demo_live_server_scenario.py
```

## What it does

1. Checks `/health`
2. Provisions a Vapi assistant via `/api/v1/vapi/provision`
3. Saves a memory through the webhook route
4. Captures visual context
5. Builds user context
6. Triggers the meeting-start whisper
7. Lists recent Vapi calls
8. Injects a live whisper if an active call is found
9. Fetches transcript and reports

## Output

Results are written to:

- [demo_live_server_output.json](demo_live_server_output.json)

## Optional

To target a different server URL:

```bash
AETHER_BASE_URL=https://your-server-url \
AETHER_WEBHOOK_TOKEN=demo-webhook-token \
PYTHONPATH=/home/wassim/repos/global_ai/aether \
/home/wassim/repos/global_ai/.venv/bin/python scripts/demo_live_server_scenario.py
```
