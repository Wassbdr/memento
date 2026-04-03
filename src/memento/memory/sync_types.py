"""Typed payloads returned by memory synchronization and retrieval APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryScoreBreakdown:
    """Explainable scoring components used to rank one memory hit."""

    semantic_score: float
    context_bonus: float
    affective_bonus: float
    trusted_people_bonus: float
    routine_time_bonus: float
    recency_bonus: float
    anchor_bonus: float
    staleness_penalty: float
    final_score: float
    routine_minutes_until: int | None = None
    episode_recency_days: int | None = None
    validation_recency_days: int | None = None
    weight_profile: str = "baseline"
    weight_signals: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryIngestionIssue:
    """One conflict or reconciliation action detected during ingestion."""

    issue_type: str
    entity_ids: tuple[str, ...]
    detail: str
    resolution: str


@dataclass(frozen=True)
class MemoryIngestionReport:
    """Summary of normalization and conflict handling before synchronization."""

    merged_people: int = 0
    merged_places: int = 0
    merged_routines: int = 0
    merged_episodes: int = 0
    issues: tuple[MemoryIngestionIssue, ...] = ()


@dataclass(frozen=True)
class MemorySyncReport:
    """Write-side metrics for one patient synchronization."""

    patient_id: str
    graph_nodes_written: int
    graph_relations_written: int
    indexed_documents: int
    deleted_documents: int
    ingestion_report: MemoryIngestionReport | None = None


@dataclass(frozen=True)
class MemoryTransactionReport:
    """Operational outcome of one graph/index synchronized transaction."""

    patient_id: str
    graph_written: bool
    index_written: bool
    rollback_performed: bool


@dataclass(frozen=True)
class MemoryContextHit:
    """One semantic hit enriched with graph context."""

    source_node_id: str
    source_label: str
    source_display_name: str
    score: float
    summary: str
    related_people: tuple[str, ...]
    related_places: tuple[str, ...]
    related_emotions: tuple[str, ...]
    related_routines: tuple[str, ...]
    related_episodes: tuple[str, ...]
    ranking_score: float = 0.0
    score_breakdown: MemoryScoreBreakdown | None = None


@dataclass(frozen=True)
class MemoryRecall:
    """Read-side answer combining vector retrieval and graph hydration."""

    query: str
    patient_id: str
    hits: tuple[MemoryContextHit, ...]
    dropped_hits: int = 0
    total_semantic_hits: int = 0
    archived_filtered_hits: int = 0


@dataclass(frozen=True)
class TrustedPersonContext:
    """One trusted person extracted from the patient knowledge graph."""

    person_id: str
    name: str
    relationship_to_patient: str
    emotional_significance: float
    notes: str


@dataclass(frozen=True)
class RoutineSupportContext:
    """One reassuring routine with practical support cues."""

    routine_id: str
    title: str
    schedule: str
    cue: str
    support_strategy: str
    place_name: str
    minutes_until_next_occurrence: int | None = None
    temporal_label: str = ""


@dataclass(frozen=True)
class MemoryIntegrityIssue:
    """One integrity issue detected in the memory stack."""

    issue_type: str
    entity_id: str
    detail: str


@dataclass(frozen=True)
class PatientMemoryIntegrityReport:
    """Integrity state for one patient memory graph and semantic projection."""

    patient_id: str
    graph_node_count: int
    graph_relation_count: int
    semantic_document_count: int
    dangling_relations: int
    orphan_documents: int
    repaired_documents: int
    issues: tuple[MemoryIntegrityIssue, ...]


@dataclass(frozen=True)
class PatientReorientationContext:
    """Graph-grounded context used to reassure and reorient one patient."""

    patient_id: str
    patient_display_name: str
    preferred_name: str
    anchors: tuple[str, ...]
    care_notes: tuple[str, ...]
    trusted_people: tuple[TrustedPersonContext, ...]
    routines: tuple[RoutineSupportContext, ...]
    memory_recall: MemoryRecall
