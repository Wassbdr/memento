"""Synchronization rules between the graph memory and semantic index."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import PersonalMemoryGraph, build_memory_graph
from .models import PatientMemorySnapshot
from .semantic import LlamaIndexSemanticIndex, MemoryDocumentProjector


@dataclass(frozen=True)
class MemorySyncReport:
    """Write-side metrics for one patient synchronization."""

    patient_id: str
    graph_nodes_written: int
    graph_relations_written: int
    indexed_documents: int
    deleted_documents: int


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


@dataclass(frozen=True)
class MemoryRecall:
    """Read-side answer combining vector retrieval and graph hydration."""

    query: str
    patient_id: str
    hits: tuple[MemoryContextHit, ...]


class InMemoryGraphStore:
    """Simple patient-scoped graph repository used for synchronization."""

    def __init__(self) -> None:
        self._graphs: dict[str, PersonalMemoryGraph] = {}

    def replace_snapshot(self, snapshot: PatientMemorySnapshot) -> PersonalMemoryGraph:
        graph = build_memory_graph(snapshot)
        self._graphs[snapshot.patient.patient_id] = graph
        return graph

    def graph_for_patient(self, patient_id: str) -> PersonalMemoryGraph | None:
        return self._graphs.get(patient_id)


class MemorySyncEngine:
    """Reference implementation of graph/vector synchronization rules."""

    def __init__(
        self,
        graph_store: InMemoryGraphStore | None = None,
        semantic_index: LlamaIndexSemanticIndex | None = None,
        projector: MemoryDocumentProjector | None = None,
    ) -> None:
        self._graph_store = graph_store or InMemoryGraphStore()
        self._semantic_index = semantic_index or LlamaIndexSemanticIndex()
        self._projector = projector or MemoryDocumentProjector()
        self._document_ids_by_patient: dict[str, tuple[str, ...]] = {}

    @property
    def graph_store(self) -> InMemoryGraphStore:
        return self._graph_store

    @property
    def semantic_index(self) -> LlamaIndexSemanticIndex:
        return self._semantic_index

    def sync_snapshot(self, snapshot: PatientMemorySnapshot) -> MemorySyncReport:
        graph = self._graph_store.replace_snapshot(snapshot)
        documents = self._projector.project(snapshot)

        previous_ids = set(self._document_ids_by_patient.get(snapshot.patient.patient_id, ()))
        current_ids = {document.document_id for document in documents}
        stale_ids = tuple(sorted(previous_ids - current_ids))
        if stale_ids:
            self._semantic_index.delete(stale_ids)

        self._semantic_index.ingest(documents)
        self._document_ids_by_patient[snapshot.patient.patient_id] = tuple(sorted(current_ids))

        return MemorySyncReport(
            patient_id=snapshot.patient.patient_id,
            graph_nodes_written=len(graph.nodes),
            graph_relations_written=len(graph.relations),
            indexed_documents=len(documents),
            deleted_documents=len(stale_ids),
        )

    def recall(
        self,
        patient_id: str,
        query: str,
        *,
        top_k: int = 3,
        source_labels: tuple[str, ...] | None = None,
    ) -> MemoryRecall:
        graph = self._graph_store.graph_for_patient(patient_id)
        if graph is None:
            raise ValueError(f"unknown patient_id: {patient_id}")

        semantic_hits = self._semantic_index.search(
            query,
            top_k=top_k,
            patient_id=patient_id,
            source_labels=source_labels,
        )
        hydrated_hits = tuple(
            self._hydrate_hit(
                graph,
                hit.document.source_node_id,
                hit.document.source_label,
                hit.score,
                hit.document.text,
            )
            for hit in semantic_hits
        )
        hydrated_hits = tuple(
            sorted(
                hydrated_hits,
                key=lambda item: (
                    -(item.score + (_context_weight(item) * 0.05)),
                    -_context_weight(item),
                    item.source_label,
                    item.source_node_id,
                ),
            )
        )
        return MemoryRecall(query=query, patient_id=patient_id, hits=hydrated_hits)

    def _hydrate_hit(
        self,
        graph: PersonalMemoryGraph,
        source_node_id: str,
        source_label: str,
        score: float,
        summary: str,
    ) -> MemoryContextHit:
        node = graph.get_node(source_node_id)
        if node is None:
            raise ValueError(f"source node not found in graph: {source_node_id}")

        related_people: set[str] = set()
        related_places: set[str] = set()
        related_emotions: set[str] = set()
        related_routines: set[str] = set()
        related_episodes: set[str] = set()

        for neighbor in graph.neighbors(source_node_id):
            if neighbor.node.label == "Person":
                related_people.add(neighbor.node.display_name)
            elif neighbor.node.label == "Place":
                related_places.add(neighbor.node.display_name)
            elif neighbor.node.label == "Emotion":
                related_emotions.add(neighbor.node.display_name)
            elif neighbor.node.label == "Routine":
                related_routines.add(neighbor.node.display_name)
            elif neighbor.node.label == "Episode":
                related_episodes.add(neighbor.node.display_name)

        return MemoryContextHit(
            source_node_id=source_node_id,
            source_label=source_label,
            source_display_name=node.display_name,
            score=score,
            summary=summary,
            related_people=tuple(sorted(related_people)),
            related_places=tuple(sorted(related_places)),
            related_emotions=tuple(sorted(related_emotions)),
            related_routines=tuple(sorted(related_routines)),
            related_episodes=tuple(sorted(related_episodes)),
        )


def _context_weight(hit: MemoryContextHit) -> int:
    return (
        len(hit.related_people)
        + len(hit.related_places)
        + len(hit.related_emotions)
        + len(hit.related_routines)
        + len(hit.related_episodes)
    )
