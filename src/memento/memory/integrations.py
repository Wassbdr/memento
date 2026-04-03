"""Optional production-oriented integrations for external memory backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .graph import (
    MemoryGraphSchema,
    MemoryNode,
    MemoryRelation,
    PersonalMemoryGraph,
    build_memory_graph,
    default_memory_schema,
    neo4j_key_properties,
)
from .models import PatientMemorySnapshot
from .semantic import MemoryDocument, SemanticSearchHit

try:
    from neo4j import GraphDatabase
    _NEO4J_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    GraphDatabase = None
    _NEO4J_IMPORT_ERROR = exc

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _CHROMA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    chromadb = None
    SentenceTransformerEmbeddingFunction = None
    _CHROMA_IMPORT_ERROR = exc


class Neo4jGraphStore:
    """Neo4j-backed graph store for patient memory snapshots."""

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        schema: MemoryGraphSchema | None = None,
        ensure_schema: bool = True,
        driver: Any | None = None,
    ) -> None:
        if driver is None and GraphDatabase is None:
            raise RuntimeError(
                "Neo4j support is not available. Install the optional 'memory-backends' dependencies "
                f"or fix the Neo4j environment. Original import error: {_NEO4J_IMPORT_ERROR!r}"
            )

        self._database = database
        self._schema = schema or default_memory_schema()
        self._driver = driver or GraphDatabase.driver(uri, auth=(username, password))

        if ensure_schema:
            self.ensure_schema()

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in _iter_cypher_statements(self._schema.to_neo4j_cypher()):
                session.run(statement).consume()

    def replace_snapshot(self, snapshot: PatientMemorySnapshot) -> PersonalMemoryGraph:
        graph = build_memory_graph(snapshot)
        return self.replace_graph(snapshot.patient.patient_id, graph)

    def replace_graph(self, patient_id: str, graph: PersonalMemoryGraph) -> PersonalMemoryGraph:
        nodes_by_id = {node.node_id: node for node in graph.nodes}

        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._replace_snapshot_tx,
                patient_id,
                graph.nodes,
                graph.relations,
                nodes_by_id,
            )
        return graph

    def delete_patient(self, patient_id: str) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._delete_patient_tx, patient_id)

    def graph_for_patient(self, patient_id: str) -> PersonalMemoryGraph | None:
        with self._driver.session(database=self._database) as session:
            node_records = list(session.run(_LOAD_PATIENT_NODES_QUERY, patient_id=patient_id))
            if not node_records:
                return None

            nodes = tuple(
                _record_to_node(record["labels"], record["properties"])
                for record in node_records
            )
            relation_records = list(session.run(_LOAD_PATIENT_RELATIONS_QUERY, patient_id=patient_id))
            relations = tuple(
                _record_to_relation(
                    relation_type=record["relation_type"],
                    source_labels=record["source_labels"],
                    source_properties=record["source_properties"],
                    target_labels=record["target_labels"],
                    target_properties=record["target_properties"],
                    properties=record["properties"],
                )
                for record in relation_records
            )
        return PersonalMemoryGraph(nodes=nodes, relations=relations)

    def batch_recall_context(
        self,
        *,
        patient_id: str,
        source_node_ids: tuple[str, ...],
    ) -> dict[str, object]:
        """Load patient relevance + source contexts for many node IDs with bounded queries."""

        source_references = _parse_source_references(source_node_ids)

        with self._driver.session(database=self._database) as session:
            patient_record = session.run(
                _LOAD_PATIENT_RECALL_CONTEXT_QUERY,
                patient_id=patient_id,
                trusted_threshold=0.75,
            ).single()
            if patient_record is None:
                return {
                    "patient_found": False,
                    "anchor_terms": (),
                    "trusted_people": (),
                    "contexts": {},
                }

            anchors = _as_clean_string_tuple(patient_record.get("anchors"))
            trusted_people = tuple(
                sorted(
                    {
                        person_name.strip().lower()
                        for person_name in _as_clean_string_tuple(patient_record.get("trusted_people"))
                        if person_name.strip()
                    }
                )
            )

            contexts: dict[str, dict[str, object]] = {}
            if source_references:
                records = list(
                    session.run(
                        _BATCH_RECALL_CONTEXT_QUERY,
                        patient_id=patient_id,
                        sources=source_references,
                    )
                )
                for record in records:
                    source_node_id = str(record.get("source_node_id", "")).strip()
                    if not source_node_id:
                        continue

                    source_label = str(record.get("source_label", "Unknown"))
                    source_properties = dict(record.get("source_properties", {}) or {})
                    neighbors_payload = record.get("neighbors") or []

                    related_people: set[str] = set()
                    related_places: set[str] = set()
                    related_emotions: set[str] = set()
                    related_routines: set[str] = set()
                    related_episodes: set[str] = set()
                    emotion_intensities: list[float] = []

                    if isinstance(neighbors_payload, list):
                        for payload in neighbors_payload:
                            if not isinstance(payload, dict):
                                continue
                            label = str(payload.get("label", ""))
                            display_name = str(payload.get("display_name", "")).strip()
                            if not display_name:
                                continue
                            if label == "Person":
                                related_people.add(display_name)
                            elif label == "Place":
                                related_places.add(display_name)
                            elif label == "Emotion":
                                related_emotions.add(display_name)
                                raw_intensity = payload.get("intensity")
                                try:
                                    intensity = float(raw_intensity)
                                except (TypeError, ValueError):
                                    intensity = None
                                if intensity is not None:
                                    emotion_intensities.append(max(0.0, min(1.0, intensity)))
                            elif label == "Routine":
                                related_routines.add(display_name)
                            elif label == "Episode":
                                related_episodes.add(display_name)

                    contexts[source_node_id] = {
                        "source_label": source_label,
                        "source_display_name": _display_name_from_properties(
                            source_properties,
                            fallback=source_node_id,
                        ),
                        "source_properties": source_properties,
                        "related_people": tuple(sorted(related_people)),
                        "related_places": tuple(sorted(related_places)),
                        "related_emotions": tuple(sorted(related_emotions)),
                        "related_routines": tuple(sorted(related_routines)),
                        "related_episodes": tuple(sorted(related_episodes)),
                        "emotion_intensities": tuple(emotion_intensities),
                    }

        return {
            "patient_found": True,
            "anchor_terms": anchors,
            "trusted_people": trusted_people,
            "contexts": contexts,
        }

    def _replace_snapshot_tx(
        self,
        tx: Any,
        patient_id: str,
        nodes: tuple[MemoryNode, ...],
        relations: tuple[MemoryRelation, ...],
        nodes_by_id: dict[str, MemoryNode],
    ) -> None:
        tx.run("MATCH (n {patient_id: $patient_id}) DETACH DELETE n", patient_id=patient_id).consume()

        for node in nodes:
            tx.run(
                f"CREATE (n:{node.label}) SET n = $properties",
                properties=dict(node.properties),
            ).consume()

        for relation in relations:
            source = nodes_by_id[relation.source_id]
            target = nodes_by_id[relation.target_id]
            relation_properties = dict(relation.properties)
            relation_properties.setdefault("patient_id", patient_id)

            tx.run(
                "MATCH "
                f"(source:{source.label} {_neo4j_match_map('source', source.label)}), "
                f"(target:{target.label} {_neo4j_match_map('target', target.label)}) "
                f"CREATE (source)-[r:{relation.relation_type}]->(target) "
                "SET r = $properties",
                properties=relation_properties,
                **_neo4j_identity_parameters("source", source),
                **_neo4j_identity_parameters("target", target),
            ).consume()

    def _delete_patient_tx(self, tx: Any, patient_id: str) -> None:
        tx.run("MATCH (n {patient_id: $patient_id}) DETACH DELETE n", patient_id=patient_id).consume()


class ChromaSemanticIndex:
    """ChromaDB-backed semantic index with sentence-transformer embeddings."""

    def __init__(
        self,
        *,
        collection_name: str = "memento-memory",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        if client is None and chromadb is None:
            raise RuntimeError(
                "ChromaDB support is not available. Install the optional 'memory-backends' dependencies "
                f"or fix the ChromaDB environment. Original import error: {_CHROMA_IMPORT_ERROR!r}"
            )
        if embedding_function is None and SentenceTransformerEmbeddingFunction is None:
            raise RuntimeError(
                "ChromaDB embedding support is not available. Install the optional 'memory-backends' "
                "dependencies, provide a custom embedding_function, or fix the ChromaDB environment. "
                f"Original import error: {_CHROMA_IMPORT_ERROR!r}"
            )

        self._client = client or _build_chroma_client(
            persist_directory=persist_directory,
            host=host,
            port=port,
            ssl=ssl,
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
            or SentenceTransformerEmbeddingFunction(model_name=embedding_model),
        )

    def ingest(self, documents: tuple[MemoryDocument, ...]) -> None:
        if not documents:
            return

        self._collection.upsert(
            ids=[document.document_id for document in documents],
            documents=[document.text for document in documents],
            metadatas=[_document_metadata(document) for document in documents],
        )

    def delete(self, document_ids: tuple[str, ...]) -> None:
        if not document_ids:
            return
        self._collection.delete(ids=list(document_ids))

    def replace_documents(self, patient_id: str, documents: tuple[MemoryDocument, ...]) -> int:
        previous_documents = self._documents_for_patient(patient_id)
        previous_ids = {document.document_id for document in previous_documents}
        current_ids = {document.document_id for document in documents}
        stale_ids = tuple(sorted(previous_ids - current_ids))

        try:
            if stale_ids:
                self.delete(stale_ids)
            self.ingest(documents)
        except Exception:
            self._collection.delete(where={"patient_id": patient_id})
            if previous_documents:
                self.ingest(previous_documents)
            raise

        return len(stale_ids)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        patient_id: str | None = None,
        source_labels: tuple[str, ...] | None = None,
    ) -> tuple[SemanticSearchHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=_chroma_where_clause(patient_id=patient_id, source_labels=source_labels),
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[SemanticSearchHit] = []
        for document_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            payload = dict(metadata or {})
            source_label = str(payload.get("source_label", "") or "Unknown")
            source_node_id = _safe_source_node_id(
                payload,
                source_label=source_label,
                document_id=str(document_id),
            )
            score = round(1.0 / (1.0 + float(distance or 0.0)), 4)
            hits.append(
                SemanticSearchHit(
                    document=MemoryDocument(
                        document_id=str(document_id),
                        source_node_id=source_node_id,
                        source_label=source_label,
                        text=str(text or ""),
                        metadata=payload,
                    ),
                    score=score,
                )
            )
        return tuple(hits)

    def _documents_for_patient(self, patient_id: str) -> tuple[MemoryDocument, ...]:
        result = self._collection.get(
            where={"patient_id": patient_id},
            include=["documents", "metadatas"],
        )

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        loaded_documents: list[MemoryDocument] = []
        for document_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            payload = dict(metadata or {})
            source_label = str(payload.get("source_label", "") or "Unknown")
            source_node_id = _safe_source_node_id(
                payload,
                source_label=source_label,
                document_id=str(document_id),
            )
            loaded_documents.append(
                MemoryDocument(
                    document_id=str(document_id),
                    source_node_id=source_node_id,
                    source_label=source_label,
                    text=str(text or ""),
                    metadata=payload,
                )
            )
        return tuple(sorted(loaded_documents, key=lambda item: item.document_id))

    def close(self) -> None:
        close_client = getattr(self._client, "close", None)
        if callable(close_client):
            close_client()


_LOAD_PATIENT_NODES_QUERY = """
MATCH (node {patient_id: $patient_id})
RETURN DISTINCT labels(node) AS labels, properties(node) AS properties
"""


_LOAD_PATIENT_RELATIONS_QUERY = """
MATCH ()-[rel {patient_id: $patient_id}]->()
RETURN
    type(rel) AS relation_type,
    labels(startNode(rel)) AS source_labels,
    properties(startNode(rel)) AS source_properties,
    labels(endNode(rel)) AS target_labels,
    properties(endNode(rel)) AS target_properties,
    properties(rel) AS properties
"""


_LOAD_PATIENT_RECALL_CONTEXT_QUERY = """
MATCH (patient:Patient {patient_id: $patient_id})
OPTIONAL MATCH (patient)-[:KNOWS]->(trusted:Person)
RETURN
    patient.patient_id AS patient_id,
    patient.anchors AS anchors,
    [
        person IN collect(trusted)
        WHERE coalesce(person.emotional_significance, 0.0) >= $trusted_threshold
        | toLower(coalesce(person.name, ""))
    ] AS trusted_people
"""


_BATCH_RECALL_CONTEXT_QUERY = """
UNWIND $sources AS source_ref
MATCH (source {patient_id: $patient_id, id: source_ref.id})
WHERE source_ref.label IN labels(source)
OPTIONAL MATCH (source)-[]-(neighbor {patient_id: $patient_id})
WITH source, collect(DISTINCT neighbor) AS neighbors
RETURN
    toLower(head(labels(source))) + ':' + toString(source.id) AS source_node_id,
    head(labels(source)) AS source_label,
    properties(source) AS source_properties,
    [
        neighbor IN neighbors |
        {
            label: head(labels(neighbor)),
            display_name: coalesce(
                neighbor.display_name,
                neighbor.preferred_name,
                neighbor.name,
                neighbor.title,
                neighbor.label,
                toString(neighbor.id)
            ),
            intensity: neighbor.intensity
        }
    ] AS neighbors
"""


def _iter_cypher_statements(cypher: str) -> Iterable[str]:
    for line in cypher.splitlines():
        statement = line.strip()
        if not statement or statement.startswith("//"):
            continue
        yield statement


def _neo4j_match_map(prefix: str, label: str) -> str:
    items = ", ".join(
        f"{property_name}: ${prefix}_{property_name}"
        for property_name in neo4j_key_properties(label)
    )
    return "{" + items + "}"


def _neo4j_identity_parameters(prefix: str, node: MemoryNode) -> dict[str, object]:
    return {
        f"{prefix}_{property_name}": node.properties[property_name]
        for property_name in neo4j_key_properties(node.label)
    }


def _record_to_node(labels: list[str], properties: dict[str, object]) -> MemoryNode:
    label = labels[0]
    node_id = f"{label.lower()}:{properties['id']}"
    return MemoryNode(node_id=node_id, label=label, properties=dict(properties))


def _record_to_relation(
    *,
    relation_type: str,
    source_labels: list[str],
    source_properties: dict[str, object],
    target_labels: list[str],
    target_properties: dict[str, object],
    properties: dict[str, object],
) -> MemoryRelation:
    source_label = source_labels[0]
    target_label = target_labels[0]
    return MemoryRelation(
        source_id=f"{source_label.lower()}:{source_properties['id']}",
        relation_type=relation_type,
        target_id=f"{target_label.lower()}:{target_properties['id']}",
        properties=dict(properties),
    )


def _build_chroma_client(
    *,
    persist_directory: str | None,
    host: str | None,
    port: int,
    ssl: bool,
) -> Any:
    if host is not None:
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)
    if persist_directory is not None:
        return chromadb.PersistentClient(path=persist_directory)
    return chromadb.EphemeralClient()


def _document_metadata(document: MemoryDocument) -> dict[str, object]:
    metadata = dict(document.metadata)
    metadata["source_node_id"] = document.source_node_id
    metadata["source_label"] = document.source_label
    return metadata


def _safe_source_node_id(
    payload: dict[str, object],
    *,
    source_label: str,
    document_id: str,
) -> str:
    raw_source_node_id = payload.pop("source_node_id", None)
    if raw_source_node_id is not None:
        source_node_id = str(raw_source_node_id).strip()
        if source_node_id:
            return source_node_id

    normalized_label = source_label.strip().lower()
    if normalized_label and normalized_label != "unknown":
        return f"{normalized_label}:{document_id}"
    return f"document:{document_id}"


def _chroma_where_clause(
    *,
    patient_id: str | None,
    source_labels: tuple[str, ...] | None,
) -> dict[str, object] | None:
    filters: list[dict[str, object]] = []
    if patient_id is not None:
        filters.append({"patient_id": patient_id})
    if source_labels:
        if len(source_labels) == 1:
            filters.append({"source_label": source_labels[0]})
        else:
            filters.append({"source_label": {"$in": list(source_labels)}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _parse_source_references(source_node_ids: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for source_node_id in source_node_ids:
        text = str(source_node_id).strip()
        if not text or ":" not in text:
            continue
        prefix, entity_id = text.split(":", 1)
        label = _node_label_from_prefix(prefix)
        key = (label, entity_id)
        unique[key] = {"label": label, "id": entity_id}
    return tuple(
        unique[key]
        for key in sorted(unique.keys())
    )


def _node_label_from_prefix(prefix: str) -> str:
    normalized = prefix.strip().lower()
    mapping = {
        "patient": "Patient",
        "person": "Person",
        "place": "Place",
        "routine": "Routine",
        "episode": "Episode",
        "emotion": "Emotion",
    }
    return mapping.get(normalized, normalized.capitalize() or "Unknown")


def _display_name_from_properties(properties: dict[str, object], *, fallback: str) -> str:
    for key in ("display_name", "preferred_name", "name", "title", "label", "id"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _as_clean_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(
            text.strip()
            for text in (str(item) for item in value)
            if text.strip()
        )
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()
