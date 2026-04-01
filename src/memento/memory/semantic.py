"""Semantic indexing for patient memories with Chroma-like storage."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Protocol

from .models import PatientMemorySnapshot


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
    """Small orchestration wrapper named after the intended production retrieval layer."""


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared_tokens = set(left) & set(right)
    if not shared_tokens:
        return 0.0
    return sum(left[token] * right[token] for token in shared_tokens)
