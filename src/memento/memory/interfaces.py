"""Protocol interfaces for pluggable graph and semantic backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from .graph import PersonalMemoryGraph
    from .models import PatientMemorySnapshot
    from .semantic import MemoryDocument, SemanticSearchHit


class BatchRecallNodeContext(TypedDict, total=False):
    """One source-node context payload returned by batch recall graph backends."""

    source_label: str
    source_display_name: str
    source_properties: dict[str, object]
    related_people: tuple[str, ...]
    related_places: tuple[str, ...]
    related_emotions: tuple[str, ...]
    related_routines: tuple[str, ...]
    related_episodes: tuple[str, ...]
    emotion_intensities: tuple[float, ...]


class BatchRecallPayload(TypedDict):
    """Batch graph context payload aligned with sync engine recall expectations."""

    patient_found: bool
    anchor_terms: tuple[str, ...]
    trusted_people: tuple[str, ...]
    contexts: dict[str, BatchRecallNodeContext]


class GraphStore(Protocol):
    """Persistence contract for patient-scoped graph snapshots."""

    def replace_snapshot(self, snapshot: PatientMemorySnapshot) -> PersonalMemoryGraph:
        """Persist one patient snapshot and return the projected graph."""

    def replace_graph(self, patient_id: str, graph: PersonalMemoryGraph) -> PersonalMemoryGraph:
        """Persist one already-projected graph for a patient."""

    def graph_for_patient(self, patient_id: str) -> PersonalMemoryGraph | None:
        """Load one patient graph or return ``None`` if it is unknown."""

    def delete_patient(self, patient_id: str) -> None:
        """Delete the full graph for one patient."""

    def close(self) -> None:
        """Release any underlying resources held by the graph backend."""


@runtime_checkable
class BatchRecallGraphStore(Protocol):
    """Optional graph-store contract for one-shot candidate context hydration."""

    def batch_recall_context(
        self,
        *,
        patient_id: str,
        source_node_ids: tuple[str, ...],
    ) -> BatchRecallPayload:
        """Load patient and candidate-node contexts in one backend call."""


class SemanticIndex(Protocol):
    """Semantic retrieval contract shared by in-memory and external indexes."""

    def ingest(self, documents: tuple[MemoryDocument, ...]) -> None:
        """Upsert projected documents into the semantic index."""

    def delete(self, document_ids: tuple[str, ...]) -> None:
        """Delete documents by their stable identifiers."""

    def replace_documents(self, patient_id: str, documents: tuple[MemoryDocument, ...]) -> int:
        """Replace one patient's document set atomically and return deleted count.

        If an exception is raised, the previous patient-scoped documents must remain
        queryable so the synchronization engine can safely roll back graph writes.
        """

    def close(self) -> None:
        """Release any underlying resources held by the semantic backend."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        patient_id: str | None = None,
        source_labels: tuple[str, ...] | None = None,
    ) -> tuple[SemanticSearchHit, ...]:
        """Search the semantic index and return ranked hits."""
