"""Protocol interfaces for pluggable graph and semantic backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .graph import PersonalMemoryGraph
    from .models import PatientMemorySnapshot
    from .semantic import MemoryDocument, SemanticSearchHit


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


class SemanticIndex(Protocol):
    """Semantic retrieval contract shared by in-memory and external indexes."""

    def ingest(self, documents: tuple[MemoryDocument, ...]) -> None:
        """Upsert projected documents into the semantic index."""

    def delete(self, document_ids: tuple[str, ...]) -> None:
        """Delete documents by their stable identifiers."""

    def replace_documents(self, patient_id: str, documents: tuple[MemoryDocument, ...]) -> int:
        """Replace one patient's document set and return the deleted document count."""

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
