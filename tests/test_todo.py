from pathlib import Path

from memento import build_default_scope


def test_default_scope_contains_expected_workstreams() -> None:
    scope = build_default_scope()

    assert scope.product_vision
    assert len(scope.workstreams) == 5


def test_markdown_contains_reference_stack_items() -> None:
    scope = build_default_scope()
    markdown = scope.to_markdown()

    assert "Whisper" in markdown
    assert "LlamaIndex" in markdown
    assert "ChromaDB" in markdown
    assert "Neo4j" in markdown
    assert "Ministral 3 8B" in markdown
    assert "Voxtral TTS" in markdown


def test_todo_markdown_file_stays_in_sync() -> None:
    scope = build_default_scope()
    todo_file = Path("TODO.md")

    assert todo_file.read_text() == scope.to_markdown()
