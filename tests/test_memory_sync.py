from dataclasses import replace

from memento.memory import MemorySyncEngine

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
