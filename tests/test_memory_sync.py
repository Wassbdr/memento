from dataclasses import replace

import pytest

from memento.memory import InMemoryGraphStore, LlamaIndexSemanticIndex, MemorySyncEngine

from memory_fixtures import build_snapshot


def test_sync_engine_writes_graph_and_semantic_index_then_recalls_context() -> None:
    engine = MemorySyncEngine()
    snapshot = build_snapshot()

    report = engine.sync_snapshot(snapshot)
    recall = engine.recall("rose", "Qui vient le dimanche pour le dejeuner dans la cuisine ?", top_k=2)

    assert report.graph_nodes_written == 6
    assert report.graph_relations_written == 7
    assert report.indexed_documents == 5
    assert recall.hits
    assert recall.hits[0].source_label == "Episode"
    assert "Claire Martin" in recall.hits[0].related_people
    assert "Cuisine" in recall.hits[0].related_places
    assert "joie" in recall.hits[0].related_emotions


def test_sync_engine_deletes_stale_documents_on_resync() -> None:
    engine = MemorySyncEngine()
    first_snapshot = build_snapshot()
    second_snapshot = replace(first_snapshot, episodes=())

    engine.sync_snapshot(first_snapshot)
    report = engine.sync_snapshot(second_snapshot)
    recall = engine.recall("rose", "dejeuner du dimanche", top_k=3, source_labels=("Episode",))

    assert report.deleted_documents == 1
    assert recall.hits == ()


def test_sync_engine_deletes_stale_documents_after_engine_restart() -> None:
    graph_store = InMemoryGraphStore()
    semantic_index = LlamaIndexSemanticIndex()
    first_engine = MemorySyncEngine(graph_store=graph_store, semantic_index=semantic_index)
    second_engine = MemorySyncEngine(graph_store=graph_store, semantic_index=semantic_index)

    first_snapshot = build_snapshot()
    second_snapshot = replace(first_snapshot, episodes=())

    first_engine.sync_snapshot(first_snapshot)
    report = second_engine.sync_snapshot(second_snapshot)
    recall = second_engine.recall("rose", "dejeuner du dimanche", top_k=3, source_labels=("Episode",))

    assert report.deleted_documents == 1
    assert recall.hits == ()


class FailingSemanticIndex:
    def replace_documents(self, patient_id, documents):
        raise RuntimeError("semantic backend unavailable")

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        return ()

    def close(self):
        return None


def test_sync_engine_rolls_back_graph_when_semantic_replace_fails() -> None:
    graph_store = InMemoryGraphStore()
    baseline_engine = MemorySyncEngine(graph_store=graph_store, semantic_index=LlamaIndexSemanticIndex())
    baseline_engine.sync_snapshot(build_snapshot())

    failing_engine = MemorySyncEngine(graph_store=graph_store, semantic_index=FailingSemanticIndex())
    broken_snapshot = replace(build_snapshot(), episodes=())

    with pytest.raises(RuntimeError, match="semantic backend unavailable"):
        failing_engine.sync_snapshot(broken_snapshot)

    restored_graph = graph_store.graph_for_patient("rose")

    assert restored_graph is not None
    assert restored_graph.get_node("episode:dimanche") is not None


def test_sync_engine_context_manager_closes_backends() -> None:
    graph_store = InMemoryGraphStore()
    semantic_index = LlamaIndexSemanticIndex()

    with MemorySyncEngine(graph_store=graph_store, semantic_index=semantic_index) as engine:
        engine.sync_snapshot(build_snapshot())
