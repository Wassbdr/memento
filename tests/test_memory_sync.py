from dataclasses import replace
from datetime import datetime

import pytest

from memento.memory import (
    InMemoryGraphStore,
    LlamaIndexSemanticIndex,
    MemoryDocument,
    MemorySyncEngine,
    PatientProfile,
    SemanticSearchHit,
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
