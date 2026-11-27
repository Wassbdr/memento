from memento.demo import build_fictional_snapshot, demo_questions
from memento.memory import MemorySyncEngine, build_memory_graph


def test_fictional_snapshot_is_deterministic_for_a_seed() -> None:
    assert build_fictional_snapshot(7) == build_fictional_snapshot(7)
    assert build_fictional_snapshot(7) != build_fictional_snapshot(8)


def test_fictional_snapshot_has_relatives_places_and_dated_events() -> None:
    snapshot = build_fictional_snapshot()

    assert {person.person_id for person in snapshot.people} >= {"daughter", "son", "grandson"}
    assert len(snapshot.places) >= 3
    assert len(snapshot.episodes) >= 3
    assert all(episode.happened_on for episode in snapshot.episodes)


def test_demo_questions_point_at_existing_graph_nodes() -> None:
    snapshot = build_fictional_snapshot()
    node_ids = {node.node_id for node in build_memory_graph(snapshot).nodes}

    for question in demo_questions(snapshot):
        assert set(question.expected_node_ids) <= node_ids


def test_demo_questions_retrieve_their_expected_nodes_in_memory() -> None:
    snapshot = build_fictional_snapshot()
    engine = MemorySyncEngine()
    engine.sync_snapshot(snapshot)

    for question in demo_questions(snapshot):
        recall = engine.recall(snapshot.patient.patient_id, question.text, top_k=3)
        hit_ids = {hit.source_node_id for hit in recall.hits}
        assert hit_ids & set(question.expected_node_ids), question.question_id
