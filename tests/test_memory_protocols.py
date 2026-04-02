from memento.memory import MemoryDocumentProjector, MemorySyncEngine, SemanticSearchHit, build_memory_graph

from memory_fixtures import build_snapshot


class DuckTypedGraphStore:
    def __init__(self) -> None:
        self._graphs = {}
        self.closed = False

    def replace_snapshot(self, snapshot):
        graph = build_memory_graph(snapshot)
        self._graphs[snapshot.patient.patient_id] = graph
        return graph

    def replace_graph(self, patient_id, graph):
        self._graphs[patient_id] = graph
        return graph

    def delete_patient(self, patient_id):
        self._graphs.pop(patient_id, None)

    def close(self):
        self.closed = True

    def graph_for_patient(self, patient_id):
        return self._graphs.get(patient_id)


class DuckTypedSemanticIndex:
    def __init__(self) -> None:
        self._documents = {}
        self.closed = False

    def ingest(self, documents):
        for document in documents:
            self._documents[document.document_id] = document

    def delete(self, document_ids):
        for document_id in document_ids:
            self._documents.pop(document_id, None)

    def replace_documents(self, patient_id, documents):
        previous_ids = {
            document_id
            for document_id, document in self._documents.items()
            if document.metadata.get("patient_id") == patient_id
        }
        current_ids = {document.document_id for document in documents}

        for document_id in previous_ids - current_ids:
            self._documents.pop(document_id, None)
        for document in documents:
            self._documents[document.document_id] = document

        return len(previous_ids - current_ids)

    def close(self):
        self.closed = True

    def search(self, query, *, top_k=3, patient_id=None, source_labels=None):
        hits = []
        for document in self._documents.values():
            if patient_id is not None and document.metadata.get("patient_id") != patient_id:
                continue
            if source_labels and document.source_label not in source_labels:
                continue
            if query.lower() not in document.text.lower():
                continue
            hits.append(SemanticSearchHit(document=document, score=1.0))
        return tuple(hits[:top_k])


def test_sync_engine_accepts_duck_typed_backends() -> None:
    graph_store = DuckTypedGraphStore()
    semantic_index = DuckTypedSemanticIndex()
    snapshot = build_snapshot()
    engine = MemorySyncEngine(
        graph_store=graph_store,
        semantic_index=semantic_index,
        projector=MemoryDocumentProjector(),
    )

    report = engine.sync_snapshot(snapshot)
    recall = engine.recall("rose", "tarte", top_k=1, source_labels=("Episode",))

    assert report.indexed_documents == 5
    assert recall.hits
    assert recall.hits[0].source_label == "Episode"

    engine.close()

    assert graph_store.closed is True
    assert semantic_index.closed is True
