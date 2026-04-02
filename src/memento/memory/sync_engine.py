"""Synchronization engine orchestrating graph and semantic memory layers."""

from __future__ import annotations

from datetime import datetime

from .graph import build_memory_graph
from .graph_store import InMemoryGraphStore
from .integrity import build_patient_integrity_report, orphan_document_ids
from .interfaces import GraphStore, SemanticIndex
from .models import PatientMemorySnapshot
from .recall import build_memory_recall
from .reorientation import build_reorientation_context
from .semantic import LlamaIndexSemanticIndex, MemoryDocument, MemoryDocumentProjector
from .sync_types import MemoryRecall, MemorySyncReport, PatientMemoryIntegrityReport, PatientReorientationContext


class MemorySyncEngine:
    """Reference implementation of graph/vector synchronization rules."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        semantic_index: SemanticIndex | None = None,
        projector: MemoryDocumentProjector | None = None,
    ) -> None:
        self._graph_store = graph_store or InMemoryGraphStore()
        self._semantic_index = semantic_index or LlamaIndexSemanticIndex()
        self._projector = projector or MemoryDocumentProjector()

    @property
    def graph_store(self) -> GraphStore:
        return self._graph_store

    @property
    def semantic_index(self) -> SemanticIndex:
        return self._semantic_index

    def __enter__(self) -> MemorySyncEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        graph_close = getattr(self._graph_store, "close", None)
        if callable(graph_close):
            graph_close()

        semantic_close = getattr(self._semantic_index, "close", None)
        if callable(semantic_close):
            semantic_close()

    def sync_snapshot(self, snapshot: PatientMemorySnapshot) -> MemorySyncReport:
        patient_id = snapshot.patient.patient_id
        previous_graph = self._graph_store.graph_for_patient(patient_id)
        graph = build_memory_graph(snapshot)
        documents = self._projector.project(snapshot)
        deleted_documents = 0

        try:
            persisted_graph = self._graph_store.replace_graph(patient_id, graph)
            deleted_documents = self._semantic_index.replace_documents(patient_id, documents)
        except Exception:
            try:
                if previous_graph is None:
                    self._graph_store.delete_patient(patient_id)
                else:
                    self._graph_store.replace_graph(patient_id, previous_graph)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"memory sync failed for patient {patient_id} and graph rollback also failed"
                ) from rollback_error
            raise

        return MemorySyncReport(
            patient_id=patient_id,
            graph_nodes_written=len(persisted_graph.nodes),
            graph_relations_written=len(persisted_graph.relations),
            indexed_documents=len(documents),
            deleted_documents=deleted_documents,
        )

    def recall(
        self,
        patient_id: str,
        query: str,
        *,
        top_k: int = 3,
        source_labels: tuple[str, ...] | None = None,
        reference_datetime: datetime | None = None,
    ) -> MemoryRecall:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        graph = self._graph_store.graph_for_patient(patient_id)
        if graph is None:
            raise ValueError(f"unknown patient_id: {patient_id}")

        semantic_hits = self._semantic_index.search(
            query,
            top_k=top_k,
            patient_id=patient_id,
            source_labels=source_labels,
        )
        return build_memory_recall(
            query=query,
            patient_id=patient_id,
            graph=graph,
            semantic_hits=semantic_hits,
            reference_datetime=reference_datetime,
        )

    def reorientation_context(
        self,
        patient_id: str,
        query: str,
        *,
        top_k: int = 3,
        trusted_people_limit: int = 3,
        routines_limit: int = 3,
        source_labels: tuple[str, ...] | None = None,
        reference_datetime: datetime | None = None,
    ) -> PatientReorientationContext:
        """Build one knowledge-graph-first context tailored for cognitive support."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if trusted_people_limit <= 0:
            raise ValueError("trusted_people_limit must be positive")
        if routines_limit <= 0:
            raise ValueError("routines_limit must be positive")

        graph = self._graph_store.graph_for_patient(patient_id)
        if graph is None:
            raise ValueError(f"unknown patient_id: {patient_id}")

        semantic_hits = self._semantic_index.search(
            query,
            top_k=top_k,
            patient_id=patient_id,
            source_labels=source_labels,
        )
        recall = build_memory_recall(
            query=query,
            patient_id=patient_id,
            graph=graph,
            semantic_hits=semantic_hits,
            reference_datetime=reference_datetime,
        )

        return build_reorientation_context(
            graph=graph,
            patient_id=patient_id,
            recall=recall,
            trusted_people_limit=trusted_people_limit,
            routines_limit=routines_limit,
            reference_datetime=reference_datetime,
        )

    def integrity_report(
        self,
        patient_id: str,
        *,
        repair: bool = False,
    ) -> PatientMemoryIntegrityReport:
        """Return one integrity report and optionally repair orphan semantic documents."""

        graph = self._graph_store.graph_for_patient(patient_id)
        if graph is None:
            raise ValueError(f"unknown patient_id: {patient_id}")

        semantic_documents = _semantic_documents_for_patient(self._semantic_index, patient_id)
        report = build_patient_integrity_report(
            patient_id=patient_id,
            graph=graph,
            semantic_documents=semantic_documents,
        )

        if not repair:
            return report

        orphan_ids = orphan_document_ids(report)
        if orphan_ids:
            self._semantic_index.delete(orphan_ids)

        repaired_documents = len(orphan_ids)
        repaired_documents_state = _semantic_documents_for_patient(self._semantic_index, patient_id)
        return build_patient_integrity_report(
            patient_id=patient_id,
            graph=graph,
            semantic_documents=repaired_documents_state,
            repaired_documents=repaired_documents,
        )

    def integrity_report_all(self, *, repair: bool = False) -> tuple[PatientMemoryIntegrityReport, ...]:
        """Run integrity checks for all known patients in the current graph store."""

        patient_ids = _graph_store_patient_ids(self._graph_store)
        return tuple(self.integrity_report(patient_id, repair=repair) for patient_id in patient_ids)


def _graph_store_patient_ids(graph_store: GraphStore) -> tuple[str, ...]:
    patient_ids_method = getattr(graph_store, "patient_ids", None)
    if callable(patient_ids_method):
        value = patient_ids_method()
        if isinstance(value, (tuple, list)):
            return tuple(sorted(str(item) for item in value))

    graphs_attribute = getattr(graph_store, "_graphs", None)
    if isinstance(graphs_attribute, dict):
        return tuple(sorted(str(patient_id) for patient_id in graphs_attribute.keys()))

    return ()


def _semantic_documents_for_patient(
    semantic_index: SemanticIndex,
    patient_id: str,
) -> tuple[MemoryDocument, ...]:
    documents_for_patient = getattr(semantic_index, "documents_for_patient", None)
    if callable(documents_for_patient):
        result = documents_for_patient(patient_id)
        if isinstance(result, (tuple, list)):
            return tuple(sorted(result, key=lambda item: item.document_id))

    collection = getattr(semantic_index, "collection", None)
    list_documents = getattr(collection, "list_documents", None)
    if callable(list_documents):
        result = list_documents(metadata_filters={"patient_id": patient_id})
        if isinstance(result, (tuple, list)):
            return tuple(sorted(result, key=lambda item: item.document_id))

    private_documents = getattr(semantic_index, "_documents", None)
    if isinstance(private_documents, dict):
        result: list[MemoryDocument] = []
        for document in private_documents.values():
            metadata = getattr(document, "metadata", {})
            if not isinstance(metadata, dict):
                continue
            if metadata.get("patient_id") != patient_id:
                continue
            result.append(document)
        return tuple(sorted(result, key=lambda item: item.document_id))

    return ()
