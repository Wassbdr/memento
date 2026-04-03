"""Synchronization engine orchestrating graph and semantic memory layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .emotion import EmotionalState, EmotionalStateDetector, RuleBasedEmotionalStateDetector
from .graph import PersonalMemoryGraph, build_memory_graph
from .graph_store import InMemoryGraphStore
from .ingestion import reconcile_snapshot
from .integrity import build_patient_integrity_report, orphan_document_ids
from .interfaces import BatchRecallGraphStore, BatchRecallPayload, GraphStore, SemanticIndex
from .models import PatientMemorySnapshot
from .recall import RecallNodeContext, build_memory_recall
from .reorientation import build_reorientation_context
from .semantic import LlamaIndexSemanticIndex, MemoryDocument, MemoryDocumentProjector
from .transaction_log import InMemoryTransactionLog, MemoryTransactionLog
from .sync_types import (
    MemoryRecall,
    MemorySyncReport,
    MemoryTransactionReport,
    PatientMemoryIntegrityReport,
    PatientReorientationContext,
)


class MemorySyncEngine:
    """Reference implementation of graph/vector synchronization rules."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        semantic_index: SemanticIndex | None = None,
        projector: MemoryDocumentProjector | None = None,
        transaction_log: MemoryTransactionLog | None = None,
        emotion_detector: EmotionalStateDetector | None = None,
        *,
        auto_recover: bool = False,
    ) -> None:
        self._graph_store = graph_store or InMemoryGraphStore()
        self._semantic_index = semantic_index or LlamaIndexSemanticIndex()
        self._projector = projector or MemoryDocumentProjector()
        self._transaction_log = transaction_log or InMemoryTransactionLog()
        self._emotion_detector = emotion_detector or RuleBasedEmotionalStateDetector()

        if auto_recover:
            self.recover_incomplete_transactions()

    @property
    def graph_store(self) -> GraphStore:
        return self._graph_store

    @property
    def semantic_index(self) -> SemanticIndex:
        return self._semantic_index

    @property
    def transaction_log(self) -> MemoryTransactionLog:
        return self._transaction_log

    @property
    def emotion_detector(self) -> EmotionalStateDetector | None:
        return self._emotion_detector

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

        transaction_log_close = getattr(self._transaction_log, "close", None)
        if callable(transaction_log_close):
            transaction_log_close()

    def sync_snapshot(self, snapshot: PatientMemorySnapshot) -> MemorySyncReport:
        normalized_snapshot, ingestion_report = reconcile_snapshot(snapshot)
        patient_id = normalized_snapshot.patient.patient_id
        graph = build_memory_graph(normalized_snapshot)
        documents = self._projector.project(normalized_snapshot)

        with _MemoryTransaction(
            graph_store=self._graph_store,
            semantic_index=self._semantic_index,
            patient_id=patient_id,
            graph=graph,
            documents=documents,
            transaction_log=self._transaction_log,
        ) as transaction:
            persisted_graph = transaction.write_graph()
            deleted_documents = transaction.write_index()
            transaction.commit()

        return MemorySyncReport(
            patient_id=patient_id,
            graph_nodes_written=len(persisted_graph.nodes),
            graph_relations_written=len(persisted_graph.relations),
            indexed_documents=len(documents),
            deleted_documents=deleted_documents,
            ingestion_report=ingestion_report,
        )

    def recover_incomplete_transactions(self) -> tuple[MemoryTransactionReport, ...]:
        """Replay prepared WAL transactions to restore graph/index consistency."""

        reports: list[MemoryTransactionReport] = []
        for pending in self._transaction_log.pending_transactions():
            try:
                self._graph_store.replace_graph(pending.patient_id, pending.graph)
                self._transaction_log.mark_graph_written(pending.transaction_id)

                self._semantic_index.replace_documents(pending.patient_id, pending.documents)
                self._transaction_log.mark_index_written(pending.transaction_id)

                self._transaction_log.mark_committed(pending.transaction_id)
                reports.append(
                    MemoryTransactionReport(
                        patient_id=pending.patient_id,
                        graph_written=True,
                        index_written=True,
                        rollback_performed=False,
                    )
                )
            except Exception as error:
                self._transaction_log.mark_failed(
                    pending.transaction_id,
                    error=f"recovery failed: {error}",
                )
                raise RuntimeError(
                    f"failed to recover memory transaction {pending.transaction_id}"
                ) from error

        return tuple(reports)

    def recall(
        self,
        patient_id: str,
        query: str,
        *,
        top_k: int = 3,
        source_labels: tuple[str, ...] | None = None,
        reference_datetime: datetime | None = None,
        include_archived: bool = False,
        emotional_state: EmotionalState | None = None,
    ) -> MemoryRecall:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        effective_emotional_state = emotional_state or _detect_emotional_state(
            detector=self._emotion_detector,
            text=query,
        )

        semantic_hits = self._semantic_index.search(
            query,
            top_k=top_k,
            patient_id=patient_id,
            source_labels=source_labels,
        )

        source_node_ids = tuple(hit.document.source_node_id for hit in semantic_hits)
        batch_payload = _graph_store_batch_recall_payload(
            graph_store=self._graph_store,
            patient_id=patient_id,
            source_node_ids=source_node_ids,
        )
        if batch_payload is not None:
            if not batch_payload.patient_found:
                raise ValueError(f"unknown patient_id: {patient_id}")
            if _batch_payload_is_complete(batch_payload, source_node_ids):
                return build_memory_recall(
                    query=query,
                    patient_id=patient_id,
                    graph=None,
                    semantic_hits=semantic_hits,
                    reference_datetime=reference_datetime,
                    include_archived=include_archived,
                    emotional_state=effective_emotional_state,
                    prefetched_contexts=batch_payload.contexts,
                    anchor_terms=batch_payload.anchor_terms,
                    trusted_people=set(batch_payload.trusted_people),
                )

        graph = self._graph_store.graph_for_patient(patient_id)
        if graph is None:
            raise ValueError(f"unknown patient_id: {patient_id}")

        return build_memory_recall(
            query=query,
            patient_id=patient_id,
            graph=graph,
            semantic_hits=semantic_hits,
            reference_datetime=reference_datetime,
            include_archived=include_archived,
            emotional_state=effective_emotional_state,
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
        include_archived: bool = False,
        emotional_state: EmotionalState | None = None,
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

        effective_emotional_state = emotional_state or _detect_emotional_state(
            detector=self._emotion_detector,
            text=query,
        )

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
            include_archived=include_archived,
            emotional_state=effective_emotional_state,
        )

        return build_reorientation_context(
            graph=graph,
            patient_id=patient_id,
            recall=recall,
            trusted_people_limit=trusted_people_limit,
            routines_limit=routines_limit,
            reference_datetime=reference_datetime,
            include_archived=include_archived,
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


@dataclass(frozen=True)
class _BatchRecallPayload:
    patient_found: bool
    contexts: dict[str, RecallNodeContext]
    anchor_terms: tuple[str, ...]
    trusted_people: tuple[str, ...]


def _graph_store_batch_recall_payload(
    *,
    graph_store: GraphStore,
    patient_id: str,
    source_node_ids: tuple[str, ...],
) -> _BatchRecallPayload | None:
    if not isinstance(graph_store, BatchRecallGraphStore):
        return None

    try:
        payload: BatchRecallPayload = graph_store.batch_recall_context(
            patient_id=patient_id,
            source_node_ids=source_node_ids,
        )
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    raw_contexts = payload.get("contexts", {})
    contexts: dict[str, RecallNodeContext] = {}
    if isinstance(raw_contexts, dict):
        for source_node_id, raw_context in raw_contexts.items():
            if not isinstance(raw_context, dict):
                continue
            node_id = str(source_node_id)
            source_properties = raw_context.get("source_properties")
            if not isinstance(source_properties, dict):
                source_properties = {}

            contexts[node_id] = RecallNodeContext(
                source_node_id=node_id,
                source_label=str(raw_context.get("source_label", "Unknown")),
                source_display_name=str(raw_context.get("source_display_name", node_id)),
                source_properties=dict(source_properties),
                related_people=_as_string_tuple(raw_context.get("related_people")),
                related_places=_as_string_tuple(raw_context.get("related_places")),
                related_emotions=_as_string_tuple(raw_context.get("related_emotions")),
                related_routines=_as_string_tuple(raw_context.get("related_routines")),
                related_episodes=_as_string_tuple(raw_context.get("related_episodes")),
                emotion_intensities=_as_float_tuple(raw_context.get("emotion_intensities")),
            )

    return _BatchRecallPayload(
        patient_found=bool(payload.get("patient_found", True)),
        contexts=contexts,
        anchor_terms=_as_string_tuple(payload.get("anchor_terms")),
        trusted_people=_as_string_tuple(payload.get("trusted_people")),
    )


def _batch_payload_is_complete(
    payload: _BatchRecallPayload,
    source_node_ids: tuple[str, ...],
) -> bool:
    expected_ids = {
        str(source_node_id).strip()
        for source_node_id in source_node_ids
        if str(source_node_id).strip()
    }
    if not expected_ids:
        return True

    available_ids = set(payload.contexts.keys())
    return expected_ids.issubset(available_ids)


def _as_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(
            text.strip()
            for text in (str(item) for item in value)
            if text.strip()
        )
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _as_float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    result: list[float] = []
    for item in value:
        try:
            cast_value = float(item)
        except (TypeError, ValueError):
            continue
        result.append(max(0.0, min(1.0, cast_value)))
    return tuple(result)


def _detect_emotional_state(
    *,
    detector: EmotionalStateDetector | None,
    text: str,
) -> EmotionalState | None:
    if detector is None:
        return None
    return detector.detect(text)


class _MemoryTransaction:
    """Explicit two-phase write helper with graph rollback semantics."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        semantic_index: SemanticIndex,
        patient_id: str,
        graph: PersonalMemoryGraph,
        documents: tuple[MemoryDocument, ...],
        transaction_log: MemoryTransactionLog,
    ) -> None:
        self._graph_store = graph_store
        self._semantic_index = semantic_index
        self._patient_id = patient_id
        self._graph = graph
        self._documents = documents
        self._transaction_log = transaction_log
        self._previous_graph = self._graph_store.graph_for_patient(patient_id)
        self._transaction_id = self._transaction_log.begin(
            patient_id=patient_id,
            graph=graph,
            documents=documents,
        )
        self._persisted_graph = None
        self._deleted_documents = 0
        self._graph_written = False
        self._index_written = False
        self._committed = False
        self._rollback_performed = False

    def __enter__(self) -> _MemoryTransaction:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and self._committed:
            return

        if not self._graph_written:
            error_message = "transaction exited before graph write"
            if exc is not None:
                error_message = str(exc)
            self._transaction_log.mark_failed(
                self._transaction_id,
                error=error_message,
            )
            return

        self._rollback_performed = True
        try:
            if self._previous_graph is None:
                self._graph_store.delete_patient(self._patient_id)
            else:
                self._graph_store.replace_graph(self._patient_id, self._previous_graph)
        except Exception as rollback_error:
            self._transaction_log.mark_failed(
                self._transaction_id,
                error=f"rollback failed: {rollback_error}",
            )
            raise RuntimeError(
                f"memory sync failed for patient {self._patient_id} and graph rollback also failed"
            ) from rollback_error

        error_message = "transaction rolled back"
        if exc is not None:
            error_message = str(exc)
        self._transaction_log.mark_rolled_back(
            self._transaction_id,
            error=error_message,
        )

    def write_graph(self):
        self._persisted_graph = self._graph_store.replace_graph(self._patient_id, self._graph)
        self._graph_written = True
        self._transaction_log.mark_graph_written(self._transaction_id)
        return self._persisted_graph

    def write_index(self) -> int:
        self._deleted_documents = self._semantic_index.replace_documents(self._patient_id, self._documents)
        self._index_written = True
        self._transaction_log.mark_index_written(self._transaction_id)
        return self._deleted_documents

    def commit(self) -> MemoryTransactionReport:
        if not self._graph_written or not self._index_written:
            raise RuntimeError("memory transaction commit requires graph and index writes")
        self._committed = True
        self._transaction_log.mark_committed(self._transaction_id)
        return self.report

    @property
    def report(self) -> MemoryTransactionReport:
        return MemoryTransactionReport(
            patient_id=self._patient_id,
            graph_written=self._graph_written,
            index_written=self._index_written,
            rollback_performed=self._rollback_performed,
        )
