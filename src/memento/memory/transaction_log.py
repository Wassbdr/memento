"""Write-ahead transaction logging for graph/index synchronization."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Protocol
from uuid import uuid4

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None

from .graph import MemoryNode, MemoryRelation, PersonalMemoryGraph
from .semantic import MemoryDocument

_PENDING_STATES = {"prepared", "graph_written", "index_written"}


@dataclass(frozen=True)
class PendingMemoryTransaction:
    """One incomplete transaction that can be replayed on recovery."""

    transaction_id: str
    patient_id: str
    state: str
    graph: PersonalMemoryGraph
    documents: tuple[MemoryDocument, ...]


class MemoryTransactionLog(Protocol):
    """Storage contract for write-ahead memory synchronization events."""

    def begin(
        self,
        *,
        patient_id: str,
        graph: PersonalMemoryGraph,
        documents: tuple[MemoryDocument, ...],
    ) -> str:
        """Record one prepared transaction and return its identifier."""

    def mark_graph_written(self, transaction_id: str) -> None:
        """Mark one transaction after graph write succeeds."""

    def mark_index_written(self, transaction_id: str) -> None:
        """Mark one transaction after semantic index write succeeds."""

    def mark_committed(self, transaction_id: str) -> None:
        """Mark one transaction as committed."""

    def mark_rolled_back(self, transaction_id: str, *, error: str = "") -> None:
        """Mark one transaction as rolled back."""

    def mark_failed(self, transaction_id: str, *, error: str) -> None:
        """Mark one transaction as failed before commit."""

    def pending_transactions(self) -> tuple[PendingMemoryTransaction, ...]:
        """Return transactions that are safe to replay."""

    def close(self) -> None:
        """Release underlying resources."""


@dataclass
class _InMemoryTransactionRecord:
    patient_id: str
    graph: PersonalMemoryGraph
    documents: tuple[MemoryDocument, ...]
    state: str
    error: str = ""


class InMemoryTransactionLog:
    """Ephemeral transaction log mostly used for tests and local dev."""

    def __init__(self) -> None:
        self._records: dict[str, _InMemoryTransactionRecord] = {}

    def begin(
        self,
        *,
        patient_id: str,
        graph: PersonalMemoryGraph,
        documents: tuple[MemoryDocument, ...],
    ) -> str:
        transaction_id = _new_transaction_id()
        self._records[transaction_id] = _InMemoryTransactionRecord(
            patient_id=patient_id,
            graph=graph,
            documents=documents,
            state="prepared",
        )
        return transaction_id

    def mark_graph_written(self, transaction_id: str) -> None:
        self._set_state(transaction_id, state="graph_written")

    def mark_index_written(self, transaction_id: str) -> None:
        self._set_state(transaction_id, state="index_written")

    def mark_committed(self, transaction_id: str) -> None:
        self._set_state(transaction_id, state="committed")

    def mark_rolled_back(self, transaction_id: str, *, error: str = "") -> None:
        self._set_state(transaction_id, state="rolled_back", error=error)

    def mark_failed(self, transaction_id: str, *, error: str) -> None:
        self._set_state(transaction_id, state="failed", error=error)

    def pending_transactions(self) -> tuple[PendingMemoryTransaction, ...]:
        pending: list[PendingMemoryTransaction] = []
        for transaction_id, record in sorted(self._records.items()):
            if record.state not in _PENDING_STATES:
                continue
            pending.append(
                PendingMemoryTransaction(
                    transaction_id=transaction_id,
                    patient_id=record.patient_id,
                    state=record.state,
                    graph=record.graph,
                    documents=record.documents,
                )
            )
        return tuple(pending)

    def close(self) -> None:
        return None

    def _set_state(self, transaction_id: str, *, state: str, error: str = "") -> None:
        record = self._records.get(transaction_id)
        if record is None:
            return
        record.state = state
        record.error = error


class JsonlTransactionLog:
    """Durable JSONL write-ahead log for crash recovery across process restarts."""

    def __init__(
        self,
        *,
        path: str | Path,
        compact_threshold_bytes: int = 5_000_000,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._compact_threshold_bytes = max(compact_threshold_bytes, 0)

    @property
    def path(self) -> Path:
        return self._path

    def begin(
        self,
        *,
        patient_id: str,
        graph: PersonalMemoryGraph,
        documents: tuple[MemoryDocument, ...],
    ) -> str:
        transaction_id = _new_transaction_id()
        self._append_event(
            {
                "event": "prepared",
                "transaction_id": transaction_id,
                "patient_id": patient_id,
                "timestamp": _utc_timestamp(),
                "graph": _serialize_graph(graph),
                "documents": [_serialize_document(document) for document in documents],
            }
        )
        return transaction_id

    def mark_graph_written(self, transaction_id: str) -> None:
        self._append_state_event(transaction_id, event="graph_written")

    def mark_index_written(self, transaction_id: str) -> None:
        self._append_state_event(transaction_id, event="index_written")

    def mark_committed(self, transaction_id: str) -> None:
        self._append_state_event(transaction_id, event="committed")

    def mark_rolled_back(self, transaction_id: str, *, error: str = "") -> None:
        self._append_state_event(transaction_id, event="rolled_back", error=error)

    def mark_failed(self, transaction_id: str, *, error: str) -> None:
        self._append_state_event(transaction_id, event="failed", error=error)

    def pending_transactions(self) -> tuple[PendingMemoryTransaction, ...]:
        with _advisory_lock(self._lock_path):
            states = _jsonl_states(self._path)
            if self._should_compact_locked():
                _write_states_snapshot(self._path, states)

        pending: list[PendingMemoryTransaction] = []
        for transaction_id in sorted(states.keys()):
            state = states[transaction_id]
            if state["event"] not in _PENDING_STATES:
                continue
            graph_payload = state.get("graph")
            documents_payload = state.get("documents")
            if not isinstance(graph_payload, dict) or not isinstance(documents_payload, list):
                continue
            pending.append(
                PendingMemoryTransaction(
                    transaction_id=transaction_id,
                    patient_id=str(state.get("patient_id", "")),
                    state=str(state["event"]),
                    graph=_deserialize_graph(graph_payload),
                    documents=tuple(
                        _deserialize_document(document_payload)
                        for document_payload in documents_payload
                        if isinstance(document_payload, dict)
                    ),
                )
            )
        return tuple(pending)

    def close(self) -> None:
        return None

    def compact(self) -> None:
        """Compact the WAL by keeping only the latest state per transaction."""

        with _advisory_lock(self._lock_path):
            states = _jsonl_states(self._path)
            _write_states_snapshot(self._path, states)

    def _append_state_event(self, transaction_id: str, *, event: str, error: str = "") -> None:
        payload: dict[str, object] = {
            "event": event,
            "transaction_id": transaction_id,
            "timestamp": _utc_timestamp(),
        }
        if error:
            payload["error"] = error
        self._append_event(payload)

    def _append_event(self, payload: dict[str, object]) -> None:
        with _advisory_lock(self._lock_path):
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
            if self._should_compact_locked():
                states = _jsonl_states(self._path)
                _write_states_snapshot(self._path, states)

    def _should_compact_locked(self) -> bool:
        if self._compact_threshold_bytes <= 0:
            return False
        if not self._path.exists():
            return False
        return self._path.stat().st_size >= self._compact_threshold_bytes


def _jsonl_states(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}

    states: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            transaction_id = event.get("transaction_id")
            if not isinstance(transaction_id, str) or not transaction_id.strip():
                continue

            previous = states.get(transaction_id)
            if previous is None:
                states[transaction_id] = dict(event)
                continue

            merged = dict(previous)
            merged.update(event)
            states[transaction_id] = merged
    return states


def _write_states_snapshot(path: Path, states: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for transaction_id in sorted(states.keys()):
            handle.write(json.dumps(states[transaction_id], ensure_ascii=True))
            handle.write("\n")
    temporary_path.replace(path)


def _new_transaction_id() -> str:
    return str(uuid4())


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _serialize_graph(graph: PersonalMemoryGraph) -> dict[str, object]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "label": node.label,
                "properties": dict(node.properties),
            }
            for node in graph.nodes
        ],
        "relations": [
            {
                "source_id": relation.source_id,
                "relation_type": relation.relation_type,
                "target_id": relation.target_id,
                "properties": dict(relation.properties),
            }
            for relation in graph.relations
        ],
    }


def _deserialize_graph(payload: dict[str, object]) -> PersonalMemoryGraph:
    raw_nodes = payload.get("nodes", [])
    raw_relations = payload.get("relations", [])

    nodes: list[MemoryNode] = []
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            nodes.append(
                MemoryNode(
                    node_id=str(raw_node.get("node_id", "")),
                    label=str(raw_node.get("label", "")),
                    properties=dict(raw_node.get("properties", {})),
                )
            )

    relations: list[MemoryRelation] = []
    if isinstance(raw_relations, list):
        for raw_relation in raw_relations:
            if not isinstance(raw_relation, dict):
                continue
            relations.append(
                MemoryRelation(
                    source_id=str(raw_relation.get("source_id", "")),
                    relation_type=str(raw_relation.get("relation_type", "")),
                    target_id=str(raw_relation.get("target_id", "")),
                    properties=dict(raw_relation.get("properties", {})),
                )
            )

    return PersonalMemoryGraph(nodes=tuple(nodes), relations=tuple(relations))


def _serialize_document(document: MemoryDocument) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "source_node_id": document.source_node_id,
        "source_label": document.source_label,
        "text": document.text,
        "metadata": dict(document.metadata),
    }


def _deserialize_document(payload: dict[str, object]) -> MemoryDocument:
    return MemoryDocument(
        document_id=str(payload.get("document_id", "")),
        source_node_id=str(payload.get("source_node_id", "")),
        source_label=str(payload.get("source_label", "Unknown")),
        text=str(payload.get("text", "")),
        metadata=dict(payload.get("metadata", {})),
    )


@contextmanager
def _advisory_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
