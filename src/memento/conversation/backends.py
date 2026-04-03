"""Concrete conversation backends used by the live runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable
from urllib import error, request

from .generation import ConversationGeneration, ConversationMessage

_DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://127.0.0.1:11434/v1"


@dataclass(frozen=True)
class OpenAICompatibleBackendConfig:
    """Connection settings for one OpenAI-compatible chat backend."""

    base_url: str = _DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = 60.0
    chat_completions_path: str = "/chat/completions"
    extra_headers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        normalized_path = self.chat_completions_path.strip()
        if not normalized_path:
            raise ValueError("chat_completions_path must not be empty")
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path
        normalized_base_url = self.base_url.strip().rstrip("/")
        if normalized_base_url.endswith(normalized_path):
            normalized_base_url = normalized_base_url[: -len(normalized_path)].rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        normalized_headers: list[tuple[str, str]] = []
        for key, value in self.extra_headers:
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if not normalized_key:
                raise ValueError("extra_headers keys must not be empty")
            normalized_headers.append((normalized_key, normalized_value))

        object.__setattr__(self, "base_url", normalized_base_url)
        object.__setattr__(self, "api_key", _normalize_optional_string(self.api_key))
        object.__setattr__(self, "chat_completions_path", normalized_path)
        object.__setattr__(self, "extra_headers", tuple(normalized_headers))


class OpenAICompatibleConversationBackend:
    """Call a local or remote OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        config: OpenAICompatibleBackendConfig | None = None,
        *,
        transport: Callable[[str, bytes, dict[str, str], float], bytes] | None = None,
    ) -> None:
        self._config = config or OpenAICompatibleBackendConfig()
        self._transport = transport or _default_transport

    @property
    def config(self) -> OpenAICompatibleBackendConfig:
        return self._config

    def generate(
        self,
        messages: tuple[ConversationMessage, ...],
        *,
        model_name: str,
        temperature: float,
    ) -> ConversationGeneration:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "temperature": temperature,
            "stream": False,
        }

        try:
            raw_response = self._transport(
                self._chat_completions_url,
                json.dumps(payload).encode("utf-8"),
                self._headers(),
                self._config.timeout_seconds,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} (model={model_name})"
            ) from exc
        try:
            response_payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("conversation backend returned an invalid JSON payload") from exc

        return _generation_from_openai_payload(response_payload, fallback_model_name=model_name)

    def close(self) -> None:
        """Release resources held by the backend."""

    @property
    def _chat_completions_url(self) -> str:
        return f"{self._config.base_url}{self._config.chat_completions_path}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        for key, value in self._config.extra_headers:
            headers[key] = value
        return headers


def _generation_from_openai_payload(
    payload: dict[str, object],
    *,
    fallback_model_name: str,
) -> ConversationGeneration:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("conversation backend response did not include any choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("conversation backend returned an invalid choice payload")

    message_payload = first_choice.get("message")
    if not isinstance(message_payload, dict):
        raise RuntimeError("conversation backend response did not include a valid message")

    text = _extract_message_text(message_payload.get("content"))
    finish_reason = str(first_choice.get("finish_reason", "") or "")

    usage = payload.get("usage")
    prompt_tokens = None
    completion_tokens = None
    if isinstance(usage, dict):
        prompt_tokens = _as_non_negative_int(usage.get("prompt_tokens"))
        completion_tokens = _as_non_negative_int(usage.get("completion_tokens"))

    model_name = str(payload.get("model", "") or fallback_model_name)
    return ConversationGeneration(
        text=text,
        model_name=model_name,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _extract_message_text(content: object) -> str:
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    parts.append(normalized)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                value = str(item.get("text", "")).strip()
                if value:
                    parts.append(value)
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    raise RuntimeError("conversation backend returned an empty assistant message")


def _as_non_negative_int(value: object) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized < 0:
        return None
    return normalized


def _default_transport(
    url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout_seconds: float,
) -> bytes:
    request_payload = request.Request(
        url=url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(request_payload, timeout=timeout_seconds) as response:
            return response.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body or exc.reason
        raise RuntimeError(f"conversation backend HTTP {exc.code} at {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"conversation backend unavailable at {url}: {exc.reason}") from exc


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
