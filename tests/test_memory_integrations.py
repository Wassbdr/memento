from __future__ import annotations

from dataclasses import dataclass

import memento.memory.integrations as integrations_module
import memento.memory.semantic as semantic_module
from memento.memory import ChromaSemanticIndex, LlamaIndexSemanticIndex, MemoryDocumentProjector, Neo4jGraphStore

from memory_fixtures import build_snapshot


class FakeCursor:
    def __init__(self, records=None) -> None:
        self._records = list(records or [])

    def consume(self) -> None:
        return None

    def __iter__(self):
        return iter(self._records)


class FakeNeo4jSession:
    def __init__(self, driver) -> None:
        self._driver = driver

    def __enter__(self) -> FakeNeo4jSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query, **params):
        self._driver.calls.append((query, params))
        if query == integrations_module._LOAD_PATIENT_NODES_QUERY:
            return FakeCursor(self._driver.node_records)
        if query == integrations_module._LOAD_PATIENT_RELATIONS_QUERY:
            return FakeCursor(self._driver.relation_records)
        return FakeCursor()

    def execute_write(self, fn, *args):
        return fn(self, *args)


class FakeNeo4jDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.node_records = []
        self.relation_records = []
        self.closed = False

    def session(self, *, database):
        return FakeNeo4jSession(self)

    def close(self) -> None:
        self.closed = True


class FakeChromaCollection:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, dict[str, object]]] = {}
        self.fail_next_upsert = False

    def upsert(self, *, ids, documents, metadatas) -> None:
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("upsert failed")
        for document_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            self.documents[document_id] = (text, dict(metadata))

    def delete(self, *, ids=None, where=None) -> None:
        if ids is not None:
            for document_id in ids:
                self.documents.pop(document_id, None)
            return

        if where is None:
            return

        to_delete = []
        for document_id, (_, metadata) in self.documents.items():
            if _matches_where_clause(metadata, where):
                to_delete.append(document_id)
        for document_id in to_delete:
            self.documents.pop(document_id, None)

    def query(self, *, query_texts, n_results, where=None, include=None):
        query = query_texts[0].lower()
        matches = []
        for document_id, (text, metadata) in self.documents.items():
            if where is not None and not _matches_where_clause(metadata, where):
                continue
            score = 0.0 if query not in text.lower() else 0.2
            matches.append((document_id, text, metadata, score))

        matches.sort(key=lambda item: item[0])
        matches = matches[:n_results]
        return {
            "ids": [[item[0] for item in matches]],
            "documents": [[item[1] for item in matches]],
            "metadatas": [[dict(item[2]) for item in matches]],
            "distances": [[item[3] for item in matches]],
        }

    def get(self, *, where=None, include=None):
        matches = []
        for document_id, (text, metadata) in sorted(self.documents.items()):
            if where is not None and not _matches_where_clause(metadata, where):
                continue
            matches.append((document_id, text, metadata))
        return {
            "ids": [item[0] for item in matches],
            "documents": [item[1] for item in matches],
            "metadatas": [dict(item[2]) for item in matches],
        }


class FakeChromaClient:
    def __init__(self, collection) -> None:
        self.collection = collection
        self.closed = False

    def get_or_create_collection(self, *, name, embedding_function):
        return self.collection

    def close(self) -> None:
        self.closed = True


def _matches_where_clause(metadata: dict[str, object], where: dict[str, object]) -> bool:
    if "$and" in where:
        return all(_matches_where_clause(metadata, item) for item in where["$and"])
    for key, value in where.items():
        if isinstance(value, dict) and "$in" in value:
            if metadata.get(key) not in value["$in"]:
                return False
            continue
        if metadata.get(key) != value:
            return False
    return True


def test_neo4j_graph_store_replaces_and_loads_graph() -> None:
    driver = FakeNeo4jDriver()
    snapshot = build_snapshot()
    graph_store = Neo4jGraphStore(
        uri="bolt://example",
        username="neo4j",
        password="password",
        ensure_schema=False,
        driver=driver,
    )

    graph = graph_store.replace_snapshot(snapshot)
    assert graph.get_node("episode:dimanche") is not None
    assert any("DETACH DELETE" in query for query, _ in driver.calls)
    assert any("CREATE (n:Episode)" in query for query, _ in driver.calls)

    driver.node_records = [
        {
            "labels": ["Patient"],
            "properties": {
                "patient_id": "rose",
                "id": "rose",
                "display_name": "Rose Martin",
            },
        },
        {
            "labels": ["Episode"],
            "properties": {
                "patient_id": "rose",
                "id": "dimanche",
                "title": "Dejeuner du dimanche",
                "narrative": "Claire apporte une tarte.",
            },
        },
    ]
    driver.relation_records = [
        {
            "relation_type": "REMEMBERS",
            "source_labels": ["Patient"],
            "source_properties": {"patient_id": "rose", "id": "rose", "display_name": "Rose Martin"},
            "target_labels": ["Episode"],
            "target_properties": {
                "patient_id": "rose",
                "id": "dimanche",
                "title": "Dejeuner du dimanche",
                "narrative": "Claire apporte une tarte.",
            },
            "properties": {"patient_id": "rose"},
        }
    ]

    loaded_graph = graph_store.graph_for_patient("rose")

    assert loaded_graph is not None
    assert loaded_graph.get_node("episode:dimanche") is not None
    assert len(loaded_graph.relations) == 1
    assert any("MATCH (node {patient_id: $patient_id})" in query for query, _ in driver.calls)
    assert any("MATCH ()-[rel {patient_id: $patient_id}]->()" in query for query, _ in driver.calls)

    graph_store.close()

    assert driver.closed is True


def test_chroma_semantic_index_rolls_back_failed_replace(monkeypatch) -> None:
    monkeypatch.setattr(integrations_module, "chromadb", object())
    monkeypatch.setattr(integrations_module, "SentenceTransformerEmbeddingFunction", object())
    collection = FakeChromaCollection()
    index = ChromaSemanticIndex(
        client=FakeChromaClient(collection),
        embedding_function=object(),
    )
    snapshot = build_snapshot()
    original_documents = MemoryDocumentProjector().project(snapshot)
    updated_documents = tuple(document for document in original_documents if document.source_label != "Episode")

    index.ingest(original_documents)
    collection.fail_next_upsert = True

    try:
        index.replace_documents("rose", updated_documents)
    except RuntimeError as error:
        assert "upsert failed" in str(error)
    else:
        raise AssertionError("replace_documents should have failed")

    restored_documents = collection.get(where={"patient_id": "rose"})
    restored_ids = set(restored_documents["ids"])

    assert "episode:rose:dimanche" in restored_ids
    assert len(restored_ids) == len(original_documents)


def test_chroma_semantic_index_accepts_injected_client_without_optional_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(integrations_module, "chromadb", None)
    monkeypatch.setattr(integrations_module, "SentenceTransformerEmbeddingFunction", None)

    collection = FakeChromaCollection()
    index = ChromaSemanticIndex(
        client=FakeChromaClient(collection),
        embedding_function=object(),
    )

    documents = MemoryDocumentProjector().project(build_snapshot())
    index.ingest(documents)

    assert collection.documents


def test_chroma_semantic_index_search_filters_hits(monkeypatch) -> None:
    monkeypatch.setattr(integrations_module, "chromadb", object())
    monkeypatch.setattr(integrations_module, "SentenceTransformerEmbeddingFunction", object())
    collection = FakeChromaCollection()
    index = ChromaSemanticIndex(
        client=FakeChromaClient(collection),
        embedding_function=object(),
    )
    documents = MemoryDocumentProjector().project(build_snapshot())

    index.ingest(documents)
    hits = index.search("cuisine", patient_id="rose", source_labels=("Routine", "Episode"), top_k=2)

    assert hits
    assert all(hit.document.source_label in {"Routine", "Episode"} for hit in hits)

    index.close()

    assert index._client.closed is True


@dataclass
class FakeLlamaDocument:
    text: str
    doc_id: str
    metadata: dict[str, object]


@dataclass
class FakeNode:
    metadata: dict[str, object]


@dataclass
class FakeRetrievalHit:
    node: FakeNode
    score: float


class FakeRetriever:
    def __init__(self, documents) -> None:
        self._documents = documents

    def retrieve(self, query):
        hits = []
        for document in self._documents:
            score = 1.0 if query.lower() in document.text.lower() else 0.1
            hits.append(FakeRetrievalHit(node=FakeNode(metadata=dict(document.metadata)), score=score))
        hits.sort(key=lambda item: -item.score)
        return hits


class FakeVectorStoreIndex:
    def __init__(self, documents, embed_model) -> None:
        self._documents = list(documents)
        self._embed_model = embed_model

    @classmethod
    def from_documents(cls, documents, embed_model):
        return cls(documents, embed_model)

    def as_retriever(self, similarity_top_k):
        return FakeRetriever(self._documents[:similarity_top_k])


class FakeHuggingFaceEmbedding:
    def __init__(self, model_name) -> None:
        self.model_name = model_name


def test_llamaindex_semantic_index_uses_real_path_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(semantic_module, "LlamaIndexDocument", FakeLlamaDocument)
    monkeypatch.setattr(semantic_module, "VectorStoreIndex", FakeVectorStoreIndex)
    monkeypatch.setattr(semantic_module, "HuggingFaceEmbedding", FakeHuggingFaceEmbedding)

    index = LlamaIndexSemanticIndex(use_llama_index=True)
    documents = MemoryDocumentProjector().project(build_snapshot())

    deleted_documents = index.replace_documents("rose", documents)
    hits = index.search("cuisine", patient_id="rose", source_labels=("Routine",), top_k=1)

    assert deleted_documents == 0
    assert hits
    assert hits[0].document.source_label == "Routine"
