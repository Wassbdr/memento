from __future__ import annotations

from typing import Any

import pytest

from app.services import memory_agent


def test_extract_related_person_names_finds_lea_and_jean() -> None:
    text = "Wassim est le père de Léa, la petite-fille de Jean."
    related = memory_agent.extract_related_person_names(text, "Wassim")
    names_lower = {n.lower() for n in related}
    assert "léa" in names_lower
    assert "jean" in names_lower


def test_search_memories_for_contextual_network_links_network_memories(monkeypatch: Any) -> None:
    raw = [
        {
            "id": "1",
            "text": "Wassim est le père de Léa, la petite-fille de Jean.",
            "category": "family",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "id": "2",
            "text": "Léa adore les pivoines.",
            "category": "voice",
            "timestamp": "2026-01-02T00:00:00Z",
        },
        {
            "id": "3",
            "text": "Autre sujet sans lien.",
            "category": "general",
            "timestamp": "2026-01-03T00:00:00Z",
        },
    ]

    class FakeClient:
        def get_all(self, **kwargs: object) -> dict[str, list[dict[str, str]]]:
            return {"results": raw}

    monkeypatch.setattr(memory_agent, "_get_mem0_client", lambda: FakeClient())

    out = memory_agent.search_memories_for_contextual_network("Wassim")
    texts = {item["text"] for item in out}
    assert "Wassim est le père de Léa, la petite-fille de Jean." in texts
    assert "Léa adore les pivoines." in texts
    assert "Autre sujet sans lien." not in texts


@pytest.mark.parametrize(
    ("identified", "facts", "viewer", "expected_sub"),
    [
        (
            "Wassim",
            ["Wassim est le père de Léa.", "Léa adore les pivoines."],
            "Jean",
            "Jean",
        ),
    ],
)
def test_fallback_proactive_line(
    identified: str,
    facts: list[str],
    viewer: str,
    expected_sub: str,
) -> None:
    line = memory_agent._fallback_proactive_conversation_line(identified, facts, viewer)
    assert expected_sub in line
    assert identified in line
