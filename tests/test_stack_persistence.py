"""Real-stack persistence test (Neo4j + ChromaDB + Ollama from docker compose).

Run with: MEMENTO_STACK_TESTS=1 uv run pytest -m stack
It restarts the compose services, so it is skipped unless explicitly enabled.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.stack

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.getenv("MEMENTO_STACK_TESTS") != "1", reason="set MEMENTO_STACK_TESTS=1 with the stack up")
def test_answer_uses_stored_memory_after_stack_restart() -> None:
    from memento.demo.persistence import persistence_check

    result = persistence_check(REPO_ROOT)

    for term in result["expected_terms"]:
        assert term.lower() in result["before"]["answer"].lower()
        assert term.lower() in result["after"]["answer"].lower()
    assert result["after"]["guard_applied"] is False
    assert {"person:daughter", "routine:sunday_lunch"} & set(result["after"]["retrieved"])
    assert result["passed"] is True
