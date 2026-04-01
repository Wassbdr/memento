from memento.memory import LlamaIndexSemanticIndex, MemoryDocumentProjector

from memory_fixtures import build_snapshot


def test_projector_builds_semantic_documents_for_all_memory_kinds() -> None:
    projector = MemoryDocumentProjector()

    documents = projector.project(build_snapshot())

    labels = {document.source_label for document in documents}
    assert labels == {"Patient", "Person", "Place", "Routine", "Episode"}
    assert any("Claire Martin" in document.text for document in documents)
    assert any(document.to_llamaindex_payload()["id_"].startswith("episode:") for document in documents)


def test_semantic_index_recalls_relevant_routine() -> None:
    index = LlamaIndexSemanticIndex()
    documents = MemoryDocumentProjector().project(build_snapshot())

    index.ingest(documents)
    hits = index.search("Que fait Rose a 08:00 dans la cuisine ?", top_k=2, patient_id="rose")

    assert hits
    assert hits[0].document.source_label == "Routine"
    assert hits[0].document.source_node_id == "routine:breakfast"
