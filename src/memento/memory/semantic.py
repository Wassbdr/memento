"""Semantic indexing for patient memories with Chroma-like storage."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Protocol

from .models import PatientMemorySnapshot

_LLAMA_RETRIEVAL_OVERSAMPLING_FACTOR = 8

try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import Document as LlamaIndexDocument
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:  # pragma: no cover - optional dependency
    HuggingFaceEmbedding = None
    LlamaIndexDocument = None
    VectorStoreIndex = None


@dataclass(frozen=True)
class MemoryDocument:
    """One semantic-searchable document projected from the graph domain."""

    document_id: str
    source_node_id: str
    source_label: str
    text: str
    metadata: dict[str, object]

    def to_llamaindex_payload(self) -> dict[str, object]:
        """Return a dictionary shaped like a minimal LlamaIndex document payload."""

        return {
            "id_": self.document_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    def to_llamaindex_document(self) -> object:
        """Build a real LlamaIndex document when the optional dependency is installed."""

        if LlamaIndexDocument is None:
            raise RuntimeError(
                "LlamaIndex support is not installed. Install the optional 'memory-backends' dependencies."
            )

        metadata = dict(self.metadata)
        metadata["document_id"] = self.document_id
        metadata["source_node_id"] = self.source_node_id
        metadata["source_label"] = self.source_label
        return LlamaIndexDocument(text=self.text, doc_id=self.document_id, metadata=metadata)


@dataclass(frozen=True)
class SemanticSearchHit:
    """One ranked semantic search result."""

    document: MemoryDocument
    score: float


class TextEmbedder(Protocol):
    """Embed text into a vector representation."""

    def embed(self, text: str) -> dict[str, float]:
        """Return one sparse embedding for the provided text."""


class TokenTextEmbedder:
    """Small deterministic embedder suited for local tests and prototypes."""

    def embed(self, text: str) -> dict[str, float]:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if not tokens:
            return {}

        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0

        length = math.sqrt(sum(value * value for value in counts.values()))
        if length == 0:
            return counts
        return {token: value / length for token, value in counts.items()}


class InMemoryChromaCollection:
    """Tiny Chroma-like vector collection for deterministic tests."""

    def __init__(self) -> None:
        self._documents: dict[str, MemoryDocument] = {}
        self._embeddings: dict[str, dict[str, float]] = {}

    def upsert_documents(
        self,
        documents: tuple[MemoryDocument, ...],
        embeddings: tuple[dict[str, float], ...],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        for document, embedding in zip(documents, embeddings, strict=True):
            self._documents[document.document_id] = document
            self._embeddings[document.document_id] = embedding

    def delete_documents(self, document_ids: tuple[str, ...]) -> None:
        for document_id in document_ids:
            self._documents.pop(document_id, None)
            self._embeddings.pop(document_id, None)

    def query(
        self,
        query_embedding: dict[str, float],
        *,
        top_k: int,
        metadata_filters: dict[str, object] | None = None,
        source_labels: tuple[str, ...] | None = None,
    ) -> tuple[SemanticSearchHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        label_filter = set(source_labels or ())
        hits: list[SemanticSearchHit] = []

        for document_id, document in self._documents.items():
            if metadata_filters is not None:
                if any(document.metadata.get(key) != value for key, value in metadata_filters.items()):
                    continue
            if label_filter and document.source_label not in label_filter:
                continue

            embedding = self._embeddings[document_id]
            score = _cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            hits.append(SemanticSearchHit(document=document, score=round(score, 4)))

        hits.sort(key=lambda item: (-item.score, item.document.source_label, item.document.document_id))
        return tuple(hits[:top_k])

    def count(self) -> int:
        return len(self._documents)

    def list_documents(
        self,
        *,
        metadata_filters: dict[str, object] | None = None,
    ) -> tuple[MemoryDocument, ...]:
        documents = []
        for document in self._documents.values():
            if metadata_filters is not None:
                if any(document.metadata.get(key) != value for key, value in metadata_filters.items()):
                    continue
            documents.append(document)
        return tuple(sorted(documents, key=lambda item: item.document_id))


class MemoryDocumentProjector:
    """Project graph-oriented memory records into semantic documents."""

    def project(self, snapshot: PatientMemorySnapshot) -> tuple[MemoryDocument, ...]:
        patient = snapshot.patient
        people_by_id = {person.person_id: person for person in snapshot.people}
        places_by_id = {place.place_id: place for place in snapshot.places}

        documents: list[MemoryDocument] = [
            MemoryDocument(
                document_id=f"patient:{patient.patient_id}",
                source_node_id=f"patient:{patient.patient_id}",
                source_label="Patient",
                text=(
                    f"Patient {patient.display_name}. "
                    f"Nom prefere {patient.preferred_name or patient.display_name}. "
                    f"Repères rassurants {'; '.join(patient.anchors) if patient.anchors else 'aucun precise'}. "
                    f"Notes de soin {'; '.join(patient.care_notes) if patient.care_notes else 'aucune precise'}."
                ),
                metadata={"patient_id": patient.patient_id, "source_label": "Patient"},
            )
        ]

        for person in snapshot.people:
            documents.append(
                MemoryDocument(
                    document_id=f"person:{patient.patient_id}:{person.person_id}",
                    source_node_id=f"person:{person.person_id}",
                    source_label="Person",
                    text=(
                        f"{person.name} est {person.relationship_to_patient} de {patient.display_name}. "
                        f"Notes {person.notes or 'aucune note'}. "
                        f"Importance emotionnelle {person.emotional_significance:.2f}."
                    ),
                    metadata={
                        "patient_id": patient.patient_id,
                        "source_label": "Person",
                        "relationship_to_patient": person.relationship_to_patient,
                    },
                )
            )

        for place in snapshot.places:
            documents.append(
                MemoryDocument(
                    document_id=f"place:{patient.patient_id}:{place.place_id}",
                    source_node_id=f"place:{place.place_id}",
                    source_label="Place",
                    text=(
                        f"Lieu {place.name}. "
                        f"Categorie {place.category}. "
                        f"Contexte {place.notes or 'pas de detail supplementaire'}."
                    ),
                    metadata={
                        "patient_id": patient.patient_id,
                        "source_label": "Place",
                        "category": place.category,
                    },
                )
            )

        for routine in snapshot.routines:
            place_name = places_by_id[routine.place_id].name if routine.place_id is not None else "lieu non precise"
            documents.append(
                MemoryDocument(
                    document_id=f"routine:{patient.patient_id}:{routine.routine_id}",
                    source_node_id=f"routine:{routine.routine_id}",
                    source_label="Routine",
                    text=(
                        f"Routine {routine.title}. "
                        f"Horaire {routine.schedule}. "
                        f"Description {routine.description}. "
                        f"Signal {routine.cue or 'aucun'}. "
                        f"Strategie d'accompagnement {routine.support_strategy or 'aucune'}. "
                        f"Lieu {place_name}."
                    ),
                    metadata={
                        "patient_id": patient.patient_id,
                        "source_label": "Routine",
                        "schedule": routine.schedule,
                    },
                )
            )

        for episode in snapshot.episodes:
            people_names = [people_by_id[person_id].name for person_id in episode.people_ids]
            place_name = places_by_id[episode.place_id].name if episode.place_id is not None else "lieu non precise"
            emotion_names = [emotion.label for emotion in episode.emotions]
            documents.append(
                MemoryDocument(
                    document_id=f"episode:{patient.patient_id}:{episode.episode_id}",
                    source_node_id=f"episode:{episode.episode_id}",
                    source_label="Episode",
                    text=(
                        f"Souvenir {episode.title}. "
                        f"Date {episode.happened_on or 'date inconnue'}. "
                        f"Recit {episode.narrative}. "
                        f"Personnes {'; '.join(people_names) if people_names else 'aucune personne precisee'}. "
                        f"Lieu {place_name}. "
                        f"Emotions {'; '.join(emotion_names) if emotion_names else 'aucune emotion precisee'}. "
                        f"Tags {'; '.join(episode.tags) if episode.tags else 'aucun tag'}."
                    ),
                    metadata={
                        "patient_id": patient.patient_id,
                        "source_label": "Episode",
                        "happened_on": episode.happened_on,
                    },
                )
            )

        return tuple(documents)


class SemanticMemoryIndex:
    """Semantic layer that can feed a ChromaDB-like store from LlamaIndex-style docs."""

    def __init__(
        self,
        collection: InMemoryChromaCollection | None = None,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self._collection = collection or InMemoryChromaCollection()
        self._embedder = embedder or TokenTextEmbedder()

    @property
    def collection(self) -> InMemoryChromaCollection:
        return self._collection

    def ingest(self, documents: tuple[MemoryDocument, ...]) -> None:
        embeddings = tuple(self._embedder.embed(document.text) for document in documents)
        self._collection.upsert_documents(documents, embeddings)

    def delete(self, document_ids: tuple[str, ...]) -> None:
        self._collection.delete_documents(document_ids)

    def replace_documents(self, patient_id: str, documents: tuple[MemoryDocument, ...]) -> int:
        previous_documents = self._collection.list_documents(metadata_filters={"patient_id": patient_id})
        previous_ids = {document.document_id for document in previous_documents}
        current_ids = {document.document_id for document in documents}
        stale_ids = tuple(sorted(previous_ids - current_ids))
        touched_ids = tuple(sorted(previous_ids | current_ids))

        try:
            if stale_ids:
                self.delete(stale_ids)
            self.ingest(documents)
        except Exception:
            if touched_ids:
                self.delete(touched_ids)
            if previous_documents:
                self.ingest(previous_documents)
            raise

        return len(stale_ids)

    def close(self) -> None:
        """Release resources held by the in-memory semantic index."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        patient_id: str | None = None,
        source_labels: tuple[str, ...] | None = None,
    ) -> tuple[SemanticSearchHit, ...]:
        metadata_filters = {"patient_id": patient_id} if patient_id is not None else None
        return self._collection.query(
            self._embedder.embed(query),
            top_k=top_k,
            metadata_filters=metadata_filters,
            source_labels=source_labels,
        )


class LlamaIndexSemanticIndex(SemanticMemoryIndex):
    """Semantic index that can optionally use the real LlamaIndex stack."""

    def __init__(
        self,
        collection: InMemoryChromaCollection | None = None,
        embedder: TextEmbedder | None = None,
        *,
        use_llama_index: bool = False,
        llama_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llama_embed_model: object | None = None,
    ) -> None:
        self._use_llama_index = use_llama_index
        if not use_llama_index:
            super().__init__(collection=collection, embedder=embedder)
            return

        if VectorStoreIndex is None or HuggingFaceEmbedding is None:
            raise RuntimeError(
                "LlamaIndex support is not installed. Install the optional 'memory-backends' dependencies."
            )

        self._documents: dict[str, MemoryDocument] = {}
        self._documents_by_patient: dict[str, dict[str, MemoryDocument]] = {}
        self._llama_index: object | None = None
        self._llama_indexes_by_patient: dict[str, object] = {}
        self._llama_embed_model = llama_embed_model or HuggingFaceEmbedding(model_name=llama_embedding_model)

    def ingest(self, documents: tuple[MemoryDocument, ...]) -> None:
        if not self._use_llama_index:
            super().ingest(documents)
            return

        for document in documents:
            self._documents[document.document_id] = document
        self._rebuild_llama_index()

    def delete(self, document_ids: tuple[str, ...]) -> None:
        if not self._use_llama_index:
            super().delete(document_ids)
            return

        for document_id in document_ids:
            self._documents.pop(document_id, None)
        self._rebuild_llama_index()

    def replace_documents(self, patient_id: str, documents: tuple[MemoryDocument, ...]) -> int:
        if not self._use_llama_index:
            return super().replace_documents(patient_id, documents)

        previous_documents = tuple(
            document
            for document in self._documents.values()
            if document.metadata.get("patient_id") == patient_id
        )
        previous_state = dict(self._documents)
        previous_ids = {document.document_id for document in previous_documents}
        current_ids = {document.document_id for document in documents}

        try:
            for document_id in previous_ids:
                self._documents.pop(document_id, None)
            for document in documents:
                self._documents[document.document_id] = document
            self._rebuild_llama_index()
        except Exception:
            self._documents = previous_state
            self._rebuild_llama_index()
            raise

        return len(previous_ids - current_ids)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        patient_id: str | None = None,
        source_labels: tuple[str, ...] | None = None,
    ) -> tuple[SemanticSearchHit, ...]:
        if not self._use_llama_index:
            return super().search(
                query,
                top_k=top_k,
                patient_id=patient_id,
                source_labels=source_labels,
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        index, corpus_size = self._index_for_search(patient_id)
        if index is None or corpus_size == 0:
            return ()

        retrieval_pool_size = min(
            corpus_size,
            max(top_k * _LLAMA_RETRIEVAL_OVERSAMPLING_FACTOR, top_k),
        )
        retriever = index.as_retriever(similarity_top_k=retrieval_pool_size)
        raw_hits = retriever.retrieve(query)

        hits: list[SemanticSearchHit] = []
        seen_document_ids: set[str] = set()
        for hit in raw_hits:
            metadata = dict(getattr(hit.node, "metadata", {}) or {})
            document_id = str(metadata.get("document_id") or getattr(hit.node, "ref_doc_id", ""))
            if not document_id or document_id in seen_document_ids:
                continue

            document = self._documents.get(document_id)
            if document is None:
                continue
            if patient_id is not None and document.metadata.get("patient_id") != patient_id:
                continue
            if source_labels and document.source_label not in source_labels:
                continue

            score = round(float(hit.score or 0.0), 4)
            hits.append(SemanticSearchHit(document=document, score=score))
            seen_document_ids.add(document_id)
            if len(hits) >= top_k:
                break

        return tuple(hits)

    def _rebuild_llama_index(self) -> None:
        if not self._use_llama_index:
            return
        if not self._documents:
            self._llama_index = None
            self._documents_by_patient = {}
            self._llama_indexes_by_patient = {}
            return

        sorted_documents = sorted(self._documents.values(), key=lambda item: item.document_id)
        self._llama_index = _build_llama_index(sorted_documents, self._llama_embed_model)

        documents_by_patient: dict[str, dict[str, MemoryDocument]] = {}
        for document in sorted_documents:
            patient_id = str(document.metadata.get("patient_id", "")).strip()
            if not patient_id:
                continue
            patient_documents = documents_by_patient.setdefault(patient_id, {})
            patient_documents[document.document_id] = document

        self._documents_by_patient = documents_by_patient
        self._llama_indexes_by_patient = {
            patient_id: _build_llama_index(
                sorted(patient_documents.values(), key=lambda item: item.document_id),
                self._llama_embed_model,
            )
            for patient_id, patient_documents in documents_by_patient.items()
        }

    def _index_for_search(self, patient_id: str | None) -> tuple[object | None, int]:
        if patient_id is None:
            return self._llama_index, len(self._documents)

        patient_documents = self._documents_by_patient.get(patient_id)
        if not patient_documents:
            return None, 0
        return self._llama_indexes_by_patient.get(patient_id), len(patient_documents)


def _build_llama_index(documents: list[MemoryDocument], embed_model: object) -> object:
    llama_documents = [
        document.to_llamaindex_document()
        for document in documents
    ]
    return VectorStoreIndex.from_documents(
        llama_documents,
        embed_model=embed_model,
    )

def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared_tokens = set(left) & set(right)
    if not shared_tokens:
        return 0.0
    return sum(left[token] * right[token] for token in shared_tokens)
