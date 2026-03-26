from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.core.config import APP_TITLE, _log_event, logger, settings
from app.services import memory_agent


def _is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    ngrok_module = None
    ngrok_tunnel = None

    use_ngrok = _is_true(os.getenv("USE_NGROK", "false"))
    if use_ngrok:
        try:
            from pyngrok import ngrok as pyngrok

            ngrok_module = pyngrok
        except Exception as exc:
            logger.warning(
                "[INFO] USE_NGROK is enabled but pyngrok is unavailable. "
                "Skipping tunnel startup. error=%s",
                exc,
            )

        if ngrok_module is not None:
            token = (os.getenv("NGROK_AUTHTOKEN") or "").strip()
            if not token:
                logger.warning(
                    "[INFO] USE_NGROK is enabled but NGROK_AUTHTOKEN is missing. "
                    "Skipping tunnel startup."
                )
            else:
                try:
                    port = int(os.getenv("PORT", "8000"))
                    ngrok_module.set_auth_token(token)
                    ngrok_tunnel = ngrok_module.connect(addr=port, bind_tls=True)
                    public_url = getattr(ngrok_tunnel, "public_url", "")
                    logger.info("[INFO] ngrok public URL ---> %s", public_url)
                    _log_event("ngrok_started", port=port, public_url=public_url)
                except Exception as exc:
                    logger.warning(
                        "[INFO] Failed to start ngrok tunnel. "
                        "Continuing without tunnel. error=%s",
                        exc,
                    )

    yield

    if ngrok_module is not None:
        with suppress(Exception):
            public_url = getattr(ngrok_tunnel, "public_url", "")
            if public_url:
                ngrok_module.disconnect(public_url)
            ngrok_module.kill()
            _log_event("ngrok_stopped")

    with memory_agent._mem0_lock:
        with suppress(Exception):
            close_method = getattr(memory_agent._mem0_client, "close", None)
            if callable(close_method):
                close_method()
        memory_agent._mem0_client = None


app = FastAPI(title=APP_TITLE, version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Webhook-Token"],
    allow_credentials=False,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Any) -> Any:
    request_id = uuid4().hex
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_event(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            latency_ms=latency_ms,
        )
        raise

    latency_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-Id"] = request_id
    _log_event(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(router)
