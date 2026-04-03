import json

from memento import (
    ConversationMessage,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)


def test_openai_compatible_backend_posts_chat_completion_payload() -> None:
    calls: list[dict[str, object]] = []

    def fake_transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        calls.append(
            {
                "url": url,
                "payload": json.loads(payload.decode("utf-8")),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return json.dumps(
            {
                "model": "ministral-local",
                "choices": [
                    {
                        "message": {"content": "Claire vient dimanche."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 48,
                    "completion_tokens": 12,
                },
            }
        ).encode("utf-8")

    backend = OpenAICompatibleConversationBackend(
        config=OpenAICompatibleBackendConfig(
            base_url="http://localhost:11434/v1/",
            api_key="secret",
            timeout_seconds=12.5,
        ),
        transport=fake_transport,
    )

    generation = backend.generate(
        (
            ConversationMessage(role="system", content="Tu es Memento."),
            ConversationMessage(role="user", content="Qui vient dimanche ?"),
        ),
        model_name="Ministral 3 8B",
        temperature=0.2,
    )

    assert calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert calls[0]["payload"] == {
        "model": "Ministral 3 8B",
        "messages": [
            {"role": "system", "content": "Tu es Memento."},
            {"role": "user", "content": "Qui vient dimanche ?"},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["timeout"] == 12.5
    assert generation.text == "Claire vient dimanche."
    assert generation.model_name == "ministral-local"
    assert generation.finish_reason == "stop"
    assert generation.prompt_tokens == 48
    assert generation.completion_tokens == 12


def test_openai_compatible_backend_supports_content_arrays() -> None:
    def fake_transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Bonjour Rose."},
                                {"type": "ignored", "text": "n/a"},
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

    backend = OpenAICompatibleConversationBackend(transport=fake_transport)

    generation = backend.generate(
        (ConversationMessage(role="user", content="Bonjour ?"),),
        model_name="Ministral 3 8B",
        temperature=0.0,
    )

    assert generation.text == "Bonjour Rose."
    assert generation.model_name == "Ministral 3 8B"
