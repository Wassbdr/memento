from dataclasses import replace
from datetime import datetime

import pytest

from memento.memory import (
    AffectiveState,
    EmotionalState,
    JsonlTransactionLog,
    InMemoryGraphStore,
    InMemoryTransactionLog,
    LlamaIndexSemanticIndex,
    MemoryEpisode,
    MemoryDocument,
    MemoryDocumentProjector,
    MemorySyncEngine,
    PatientProfile,
    PersonProfile,
    RoutineProfile,
    SemanticSearchHit,
    build_memory_graph,
)

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


def test_reorientation_context_uses_knowledge_graph_patient_support_signals() -> None:
    engine = MemorySyncEngine()
    snapshot = build_snapshot()

    engine.sync_snapshot(snapshot)
    context = engine.reorientation_context(
        "rose",
        "Je suis perdue, qui vient le dimanche ?",
        top_k=2,
    )

    assert context.patient_id == "rose"
    assert context.patient_display_name == "Rose Martin"
    assert context.preferred_name == "Mamie Rose"
    assert "Appartement rue des Lilas" in context.anchors
    assert "Rassurer avant de recontextualiser." in context.care_notes

    assert context.trusted_people
    assert context.trusted_people[0].name == "Claire Martin"
    assert context.trusted_people[0].relationship_to_patient == "sa fille"
    assert context.trusted_people[0].emotional_significance == pytest.approx(0.95)

    assert context.routines
    assert context.routines[0].title == "Petit-dejeuner"
    assert context.routines[0].place_name == "Cuisine"
    assert context.routines[0].cue == "La bouilloire siffle"
    assert context.routines[0].temporal_label in {"maintenant", "bientot", "aujourd_hui", "plus_tard", "inconnu"}

    assert context.memory_recall.hits
    assert context.memory_recall.patient_id == "rose"


class ClinicalRankingSemanticIndex:
    def replace_documents(self, patient_id, documents):
        return 0

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        if patient_id != "rose":
            return ()

        hits = (
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="patient:rose",
                    source_node_id="patient:rose",
                    source_label="Patient",
                    text="Rose Martin est parfois confuse.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.5,
            ),
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:dimanche",
                    source_node_id="episode:dimanche",
                    source_label="Episode",
                    text="Dejeuner du dimanche avec Claire dans la cuisine.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.5,
            ),
        )
        return hits[:top_k]

    def close(self):
        return None


def test_recall_applies_explainable_clinical_scoring() -> None:
    engine = MemorySyncEngine(semantic_index=ClinicalRankingSemanticIndex())
    engine.sync_snapshot(build_snapshot())

    recall = engine.recall(
        "rose",
        "Qui est avec moi ?",
        top_k=2,
        reference_datetime=datetime(2024, 1, 7, 11, 50),
    )

    assert recall.hits
    assert recall.hits[0].source_label == "Episode"
    assert recall.hits[0].ranking_score >= recall.hits[1].ranking_score

    breakdown = recall.hits[0].score_breakdown
    assert breakdown is not None
    assert breakdown.final_score == recall.hits[0].ranking_score
    assert "trusted_person_match" in breakdown.signals
    assert "affective_signal" in breakdown.signals


def test_reorientation_context_computes_routine_temporal_priority() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    context = engine.reorientation_context(
        "rose",
        "Quelle est ma routine ?",
        top_k=2,
        reference_datetime=datetime(2024, 1, 8, 7, 50),
    )

    assert context.routines
    breakfast = context.routines[0]
    assert breakfast.minutes_until_next_occurrence == 10
    assert breakfast.temporal_label == "maintenant"


def test_reorientation_context_validates_limits() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    with pytest.raises(ValueError, match="top_k"):
        engine.reorientation_context("rose", "query", top_k=0)

    with pytest.raises(ValueError, match="trusted_people_limit"):
        engine.reorientation_context("rose", "query", trusted_people_limit=0)

    with pytest.raises(ValueError, match="routines_limit"):
        engine.reorientation_context("rose", "query", routines_limit=0)


def test_recall_validates_top_k() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    with pytest.raises(ValueError, match="top_k"):
        engine.recall("rose", "query", top_k=0)


class OrphanAwareSemanticIndex:
    def replace_documents(self, patient_id, documents):
        return 0

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        if patient_id != "rose":
            return ()

        hits = (
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:dimanche",
                    source_node_id="episode:dimanche",
                    source_label="Episode",
                    text="Dejeuner du dimanche avec Claire.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.95,
            ),
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:ghost",
                    source_node_id="episode:ghost",
                    source_label="Episode",
                    text="Souvenir orphelin.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.9,
            ),
        )
        return hits[:top_k]

    def close(self):
        return None


def test_recall_skips_orphan_semantic_hits_instead_of_failing() -> None:
    engine = MemorySyncEngine(semantic_index=OrphanAwareSemanticIndex())
    engine.sync_snapshot(build_snapshot())

    recall = engine.recall("rose", "dejeuner", top_k=3)

    assert len(recall.hits) == 1
    assert recall.hits[0].source_node_id == "episode:dimanche"
    assert recall.dropped_hits == 1
    assert recall.total_semantic_hits == 2


def test_reorientation_context_preserves_recall_drop_metrics() -> None:
    engine = MemorySyncEngine(semantic_index=OrphanAwareSemanticIndex())
    engine.sync_snapshot(build_snapshot())

    context = engine.reorientation_context("rose", "dejeuner", top_k=3)

    assert context.memory_recall.dropped_hits == 1
    assert context.memory_recall.total_semantic_hits == 2


def test_integrity_report_detects_and_repairs_orphan_documents() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    engine.semantic_index.ingest(
        (
            MemoryDocument(
                document_id="episode:rose:orphan",
                source_node_id="episode:missing",
                source_label="Episode",
                text="Souvenir orphelin a nettoyer.",
                metadata={"patient_id": "rose", "source_label": "Episode"},
            ),
        )
    )

    report = engine.integrity_report("rose")
    assert report.orphan_documents == 1
    assert report.repaired_documents == 0

    repaired_report = engine.integrity_report("rose", repair=True)
    assert repaired_report.orphan_documents == 0
    assert repaired_report.repaired_documents == 1


def test_integrity_report_all_scans_all_known_patients() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    lucie_snapshot = replace(
        build_snapshot(),
        patient=PatientProfile(
            patient_id="lucie",
            display_name="Lucie Martin",
            preferred_name="Mamie Lucie",
            care_notes=("Parler doucement.",),
            anchors=("Salon principal",),
        ),
    )
    engine.sync_snapshot(lucie_snapshot)

    reports = engine.integrity_report_all()
    patient_ids = {report.patient_id for report in reports}

    assert patient_ids == {"rose", "lucie"}


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


def test_sync_snapshot_reconciles_conflicts_before_persistence() -> None:
    engine = MemorySyncEngine()
    baseline = build_snapshot()
    conflicting_snapshot = replace(
        baseline,
        people=baseline.people
        + (
            PersonProfile(
                person_id="claire_alt",
                name="Claire Martin",
                relationship_to_patient="sa fille",
                notes="Passe aussi le mardi apres-midi.",
                emotional_significance=0.9,
            ),
        ),
        routines=baseline.routines
        + (
            RoutineProfile(
                routine_id="breakfast_alt",
                title="Petit-dejeuner",
                schedule="Tous les jours a 09:00",
                description="Cafe leger et biscotte.",
                cue="Sortir la tasse bleue",
                support_strategy="Montrer le calendrier du matin.",
                place_id="cuisine",
            ),
        ),
        episodes=baseline.episodes
        + (
            MemoryEpisode(
                episode_id="dimanche_alt",
                title="Dejeuner du dimanche",
                narrative="Claire vient avec des fleurs et du pain frais.",
                happened_on="2024-01-07",
                people_ids=("claire_alt",),
                place_id="cuisine",
                emotions=(AffectiveState(label="joie", valence=0.8, intensity=0.75),),
                tags=("famille", "dimanche"),
            ),
        ),
    )

    report = engine.sync_snapshot(conflicting_snapshot)

    assert report.indexed_documents == 5
    assert report.ingestion_report is not None
    assert report.ingestion_report.merged_people == 1
    assert report.ingestion_report.merged_routines == 1
    assert report.ingestion_report.merged_episodes == 1
    assert "routine-schedule-conflict" in {
        issue.issue_type
        for issue in report.ingestion_report.issues
    }


def test_sync_snapshot_reconciles_aliased_people_references_in_episodes() -> None:
    engine = MemorySyncEngine()
    baseline = build_snapshot()
    conflicted = replace(
        baseline,
        people=baseline.people
        + (
            PersonProfile(
                person_id="claire_alias",
                name="Claire Martin",
                relationship_to_patient="sa fille",
                notes="Alias provenant d'une saisie differente.",
                emotional_significance=0.85,
            ),
        ),
        episodes=baseline.episodes
        + (
            MemoryEpisode(
                episode_id="souvenir_alias",
                title="Promenade de quartier",
                narrative="Claire accompagne Rose jusqu'au parc.",
                happened_on="2024-01-08",
                people_ids=("claire_alias",),
                place_id="cuisine",
                tags=("promenade",),
            ),
        ),
    )

    engine.sync_snapshot(conflicted)
    recall = engine.recall("rose", "promenade", top_k=3)

    assert recall.hits
    assert "Claire Martin" in recall.hits[0].related_people


def test_recall_uses_one_pass_neighbor_hydration() -> None:
    engine = MemorySyncEngine()
    engine.sync_snapshot(build_snapshot())

    graph = engine.graph_store.graph_for_patient("rose")
    assert graph is not None

    calls = {"count": 0}
    original_neighbors = graph.neighbors

    def counting_neighbors(node_id: str):
        calls["count"] += 1
        return original_neighbors(node_id)

    graph.neighbors = counting_neighbors  # type: ignore[method-assign]

    recall = engine.recall("rose", "cuisine", top_k=3)

    assert recall.hits
    assert calls["count"] <= 2


def test_recover_incomplete_transactions_replays_pending_wal(tmp_path) -> None:
    snapshot = build_snapshot()
    graph = build_memory_graph(snapshot)
    documents = MemoryDocumentProjector().project(snapshot)
    wal_path = tmp_path / "memory-wal.jsonl"
    transaction_log = JsonlTransactionLog(path=wal_path)

    transaction_id = transaction_log.begin(
        patient_id="rose",
        graph=graph,
        documents=documents,
    )
    transaction_log.mark_graph_written(transaction_id)

    engine = MemorySyncEngine(
        graph_store=InMemoryGraphStore(),
        semantic_index=LlamaIndexSemanticIndex(),
        transaction_log=JsonlTransactionLog(path=wal_path),
    )

    reports = engine.recover_incomplete_transactions()
    recall = engine.recall("rose", "dejeuner", top_k=2, source_labels=("Episode",))

    assert len(reports) == 1
    assert reports[0].patient_id == "rose"
    assert reports[0].graph_written is True
    assert reports[0].index_written is True
    assert recall.hits


def test_auto_recover_replays_pending_transactions_on_startup(tmp_path) -> None:
    snapshot = build_snapshot()
    graph = build_memory_graph(snapshot)
    documents = MemoryDocumentProjector().project(snapshot)
    wal_path = tmp_path / "memory-wal.jsonl"
    transaction_log = JsonlTransactionLog(path=wal_path)

    transaction_id = transaction_log.begin(
        patient_id="rose",
        graph=graph,
        documents=documents,
    )
    transaction_log.mark_graph_written(transaction_id)

    engine = MemorySyncEngine(
        graph_store=InMemoryGraphStore(),
        semantic_index=LlamaIndexSemanticIndex(),
        transaction_log=JsonlTransactionLog(path=wal_path),
        auto_recover=True,
    )

    recall = engine.recall("rose", "dejeuner", top_k=2, source_labels=("Episode",))

    assert recall.hits


def test_failed_transaction_is_not_left_pending_in_wal() -> None:
    transaction_log = InMemoryTransactionLog()
    graph_store = InMemoryGraphStore()
    baseline_engine = MemorySyncEngine(
        graph_store=graph_store,
        semantic_index=LlamaIndexSemanticIndex(),
        transaction_log=transaction_log,
    )
    baseline_engine.sync_snapshot(build_snapshot())

    failing_engine = MemorySyncEngine(
        graph_store=graph_store,
        semantic_index=FailingSemanticIndex(),
        transaction_log=transaction_log,
    )

    with pytest.raises(RuntimeError, match="semantic backend unavailable"):
        failing_engine.sync_snapshot(replace(build_snapshot(), episodes=()))

    assert transaction_log.pending_transactions() == ()


def test_recall_filters_archived_hits_by_default() -> None:
    engine = MemorySyncEngine()
    archived_snapshot = replace(
        build_snapshot(),
        episodes=(
            replace(
                build_snapshot().episodes[0],
                archived_on="2026-03-20",
            ),
        ),
    )
    engine.sync_snapshot(archived_snapshot)

    recall = engine.recall("rose", "dejeuner", top_k=3, source_labels=("Episode",))

    assert recall.hits == ()
    assert recall.archived_filtered_hits == 1


def test_recall_can_include_archived_hits_when_requested() -> None:
    engine = MemorySyncEngine()
    archived_snapshot = replace(
        build_snapshot(),
        episodes=(
            replace(
                build_snapshot().episodes[0],
                archived_on="2026-03-20",
            ),
        ),
    )
    engine.sync_snapshot(archived_snapshot)

    recall = engine.recall(
        "rose",
        "dejeuner",
        top_k=3,
        source_labels=("Episode",),
        include_archived=True,
    )

    assert recall.hits
    assert recall.archived_filtered_hits == 0


class StalenessAwareSemanticIndex:
    def replace_documents(self, patient_id, documents):
        return 0

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        if patient_id != "rose":
            return ()
        hits = (
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:fresh",
                    source_node_id="episode:fresh",
                    source_label="Episode",
                    text="Souvenir du dejeuner dominical.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.8,
            ),
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:stale",
                    source_node_id="episode:stale",
                    source_label="Episode",
                    text="Souvenir du dejeuner dominical.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.8,
            ),
        )
        return hits[:top_k]

    def close(self):
        return None


def test_recall_applies_staleness_penalty_from_last_validation() -> None:
    engine = MemorySyncEngine(semantic_index=StalenessAwareSemanticIndex())

    base = build_snapshot()
    stale_episode = MemoryEpisode(
        episode_id="stale",
        title="Souvenir stale",
        narrative="Souvenir du dejeuner dominical.",
        happened_on="2024-01-07",
        people_ids=("claire",),
        place_id="cuisine",
        last_validated_on="2020-01-01",
    )
    fresh_episode = MemoryEpisode(
        episode_id="fresh",
        title="Souvenir fresh",
        narrative="Souvenir du dejeuner dominical.",
        happened_on="2024-01-07",
        people_ids=("claire",),
        place_id="cuisine",
        last_validated_on="2026-03-30",
    )
    snapshot = replace(base, episodes=(stale_episode, fresh_episode))
    engine.sync_snapshot(snapshot)

    recall = engine.recall(
        "rose",
        "dejeuner",
        top_k=2,
        reference_datetime=datetime(2026, 4, 3, 9, 0),
    )

    assert recall.hits
    assert recall.hits[0].source_node_id == "episode:fresh"
    assert recall.hits[1].source_node_id == "episode:stale"

    fresh_breakdown = recall.hits[0].score_breakdown
    stale_breakdown = recall.hits[1].score_breakdown
    assert fresh_breakdown is not None
    assert stale_breakdown is not None
    assert stale_breakdown.staleness_penalty > fresh_breakdown.staleness_penalty


class DynamicWeightSemanticIndex:
    def replace_documents(self, patient_id, documents):
        return 0

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        if patient_id != "rose":
            return ()
        hits = (
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="routine:rose:breakfast",
                    source_node_id="routine:breakfast",
                    source_label="Routine",
                    text="Routine du matin pour se rassurer.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.7,
            ),
            SemanticSearchHit(
                document=MemoryDocument(
                    document_id="episode:rose:dimanche",
                    source_node_id="episode:dimanche",
                    source_label="Episode",
                    text="Dejeuner du dimanche avec Claire.",
                    metadata={"patient_id": "rose"},
                ),
                score=0.7,
            ),
        )
        return hits[:top_k]

    def close(self):
        return None


def test_recall_uses_dynamic_weight_profile_with_emotional_context() -> None:
    engine = MemorySyncEngine(semantic_index=DynamicWeightSemanticIndex())
    engine.sync_snapshot(build_snapshot())

    recall = engine.recall(
        "rose",
        "Je suis perdue et angoissee",
        top_k=2,
        reference_datetime=datetime(2026, 4, 3, 22, 30),
        emotional_state=EmotionalState(
            label="agite",
            intensity=0.9,
            confidence=0.9,
            source="voice-tone",
        ),
    )

    assert recall.hits
    breakdown = recall.hits[0].score_breakdown
    assert breakdown is not None
    assert breakdown.weight_profile == "agitation_support"
    assert "emotion_agite" in breakdown.weight_signals
    assert "night_hours" in breakdown.weight_signals


class BatchRecallGraphStore(InMemoryGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0
        self.graph_for_patient_calls = 0

    def graph_for_patient(self, patient_id: str):
        self.graph_for_patient_calls += 1
        return super().graph_for_patient(patient_id)

    def batch_recall_context(self, *, patient_id: str, source_node_ids: tuple[str, ...]):
        self.batch_calls += 1
        graph = super().graph_for_patient(patient_id)
        if graph is None:
            return {
                "patient_found": False,
                "anchor_terms": (),
                "trusted_people": (),
                "contexts": {},
            }

        patient_node = graph.get_node(f"patient:{patient_id}")
        anchors = tuple(patient_node.properties.get("anchors", [])) if patient_node is not None else ()
        trusted = []
        if patient_node is not None:
            for neighbor in graph.neighbors(patient_node.node_id):
                if neighbor.direction != "outgoing":
                    continue
                if neighbor.relation_type != "KNOWS" or neighbor.node.label != "Person":
                    continue
                trusted.append(neighbor.node.display_name.lower())

        contexts = {}
        for source_node_id in source_node_ids:
            node = graph.get_node(source_node_id)
            if node is None:
                continue

            related_people = []
            related_places = []
            related_emotions = []
            related_routines = []
            related_episodes = []
            emotion_intensities = []

            for neighbor in graph.neighbors(source_node_id):
                if neighbor.node.label == "Person":
                    related_people.append(neighbor.node.display_name)
                elif neighbor.node.label == "Place":
                    related_places.append(neighbor.node.display_name)
                elif neighbor.node.label == "Emotion":
                    related_emotions.append(neighbor.node.display_name)
                    intensity = neighbor.node.properties.get("intensity")
                    if isinstance(intensity, (int, float)):
                        emotion_intensities.append(float(intensity))
                elif neighbor.node.label == "Routine":
                    related_routines.append(neighbor.node.display_name)
                elif neighbor.node.label == "Episode":
                    related_episodes.append(neighbor.node.display_name)

            contexts[source_node_id] = {
                "source_label": node.label,
                "source_display_name": node.display_name,
                "source_properties": dict(node.properties),
                "related_people": tuple(sorted(set(related_people))),
                "related_places": tuple(sorted(set(related_places))),
                "related_emotions": tuple(sorted(set(related_emotions))),
                "related_routines": tuple(sorted(set(related_routines))),
                "related_episodes": tuple(sorted(set(related_episodes))),
                "emotion_intensities": tuple(emotion_intensities),
            }

        return {
            "patient_found": True,
            "anchor_terms": anchors,
            "trusted_people": tuple(sorted(set(trusted))),
            "contexts": contexts,
        }


def test_recall_prefers_batch_graph_context_when_available() -> None:
    graph_store = BatchRecallGraphStore()
    engine = MemorySyncEngine(graph_store=graph_store)
    engine.sync_snapshot(build_snapshot())

    graph_store.graph_for_patient_calls = 0
    recall = engine.recall("rose", "dejeuner", top_k=2)

    assert recall.hits
    assert graph_store.batch_calls == 1
    assert graph_store.graph_for_patient_calls == 0
