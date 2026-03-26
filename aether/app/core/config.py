from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_environment() -> None:
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        return

    # Fallback for uncommon file variants (for example, ".env "),
    # while explicitly skipping template files.
    for env_file in PROJECT_ROOT.glob(".env*"):
        if env_file.name in {".env", ".env.example"}:
            continue
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file, override=False)
            break


_load_environment()


class Settings(BaseSettings):
    mem0_api_key: str | None = None
    webhook_token: str = "demo-webhook-token"
    cors_origins: str | list[str] = "http://localhost:5173,http://localhost:5174"
    # Controls how external APIs are used:
    # - auto: use real APIs when configured, otherwise fallback where supported
    # - real: force real APIs only (raise if missing config)
    # - mock: use in-memory mocks where supported
    external_services_mode: str = "auto"
    vapi_api_key: str | None = None
    vapi_base_url: str = "https://api.vapi.ai"
    # Public URL of this backend that Vapi will POST webhooks to.
    # Example: https://your-tunnel.ngrok.io
    vapi_server_url: str | None = None
    # Prénom de l’utilisateur principal (démo Alzheimer) pour les phrases proactives vocales.
    memento_viewer_name: str = "Jean"

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> str | list[str]:
        if value is None or value == "":
            return "*"
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parsed = [str(item).strip() for item in value if str(item).strip()]
            return parsed or ["*"]
        return "*"

    @field_validator("external_services_mode", mode="before")
    @classmethod
    def parse_external_services_mode(cls, value: Any) -> str:
        if value is None:
            return "auto"
        mode = str(value).strip().lower()
        if mode in {"auto", "real", "mock"}:
            return mode
        return "auto"

    @field_validator("webhook_token", mode="before")
    @classmethod
    def parse_webhook_token(cls, value: Any) -> str:
        token = "" if value is None else str(value).strip()
        return token or "demo-webhook-token"

    @property
    def cors_origins_list(self) -> list[str]:
        value = self.cors_origins
        if isinstance(value, list):
            return value or ["*"]
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
            return parts or ["*"]
        return ["*"]


settings = Settings()

APP_TITLE = "Memento"
API_PREFIX = "/api/v1"
DEMO_USER_ID = "hackathon_demo_user"
MEMORY_TOOL_NAME = "save_user_memory"
CALENDAR_TOOL_NAME = "check_calendar"
MAX_PREVIEW_LEN = 80

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voice_ai_backend")


def _log_event(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
    logger.info("event=%s %s", event, payload)
