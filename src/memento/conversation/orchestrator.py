"""Conversation orchestration built on top of the memory layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

from memento.memory import EmotionalState, MemorySyncEngine, PatientReorientationContext

from .generation import (
    DEFAULT_MINISTRAL_MODEL_NAME,
    ConversationGeneration,
    ConversationMessage,
    ConversationModelBackend,
)

DEFAULT_CONVERSATION_SYSTEM_INSTRUCTION = (
    "Tu es Memento, un assistant vocal de reassurance pour une personne atteinte "
    "de troubles de la memoire. Reponds en francais avec des phrases courtes, "
    "calmes et concretes. Utilise seulement les informations soutenues par le "
    "contexte memoire fourni. N'invente ni noms, ni dates, ni routines. Si le "
    "contexte ne suffit pas, dis-le explicitement et propose une reorientation "
    "douce. Priorise l'identite de la personne, ses reperes rassurants, ses "
    "proches de confiance et les routines immediates."
)


@dataclass(frozen=True)
class ConversationConfig:
    """Runtime settings controlling retrieval and prompt construction."""

    model_name: str = DEFAULT_MINISTRAL_MODEL_NAME
    temperature: float = 0.2
    top_k: int = 3
    trusted_people_limit: int = 3
    routines_limit: int = 3
    max_prompt_memories: int = 3
    system_instruction: str = DEFAULT_CONVERSATION_SYSTEM_INSTRUCTION

    def __post_init__(self) -> None:
        normalized_model_name = self.model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")

        normalized_instruction = self.system_instruction.strip()
        if not normalized_instruction:
            raise ValueError("system_instruction must not be empty")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.trusted_people_limit <= 0:
            raise ValueError("trusted_people_limit must be positive")
        if self.routines_limit <= 0:
            raise ValueError("routines_limit must be positive")
        if self.max_prompt_memories <= 0:
            raise ValueError("max_prompt_memories must be positive")

        object.__setattr__(self, "model_name", normalized_model_name)
        object.__setattr__(self, "system_instruction", normalized_instruction)


@dataclass(frozen=True)
class RetrievedMemoryEvidence:
    """One retrieved memory item exposed in the conversation trace."""

    source_node_id: str
    source_label: str
    source_display_name: str
    summary: str
    ranking_score: float
    signals: tuple[str, ...]
    related_people: tuple[str, ...]
    related_places: tuple[str, ...]
    related_emotions: tuple[str, ...]
    related_routines: tuple[str, ...]
    related_episodes: tuple[str, ...]


@dataclass(frozen=True)
class ConversationTrace:
    """Prompt and retrieval evidence used for one generated reply."""

    system_prompt: str
    user_prompt: str
    messages: tuple[ConversationMessage, ...]
    retrieved_memories: tuple[RetrievedMemoryEvidence, ...]
    dropped_hits: int
    total_semantic_hits: int


@dataclass(frozen=True)
class ConversationResponse:
    """Final output of the conversation orchestration loop."""

    patient_id: str
    question: str
    answer: str
    generation: ConversationGeneration
    trace: ConversationTrace
    context: PatientReorientationContext


class ConversationOrchestrator:
    """Assemble retrieval, prompt building and model generation for one reply."""

    def __init__(
        self,
        memory_engine: MemorySyncEngine,
        backend: ConversationModelBackend,
        config: ConversationConfig | None = None,
    ) -> None:
        self._memory_engine = memory_engine
        self._backend = backend
        self._config = config or ConversationConfig()

    @property
    def config(self) -> ConversationConfig:
        return self._config

    def close(self) -> None:
        backend_close = getattr(self._backend, "close", None)
        if callable(backend_close):
            backend_close()

    def respond(
        self,
        patient_id: str,
        question: str,
        *,
        conversation_history: tuple[ConversationMessage, ...] = (),
        source_labels: tuple[str, ...] | None = None,
        reference_datetime: datetime | None = None,
        include_archived: bool = False,
        emotional_state: EmotionalState | None = None,
    ) -> ConversationResponse:
        normalized_patient_id = patient_id.strip()
        if not normalized_patient_id:
            raise ValueError("patient_id must not be empty")

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        normalized_history = _normalize_history(conversation_history)
        context = self._memory_engine.reorientation_context(
            normalized_patient_id,
            normalized_question,
            top_k=self._config.top_k,
            trusted_people_limit=self._config.trusted_people_limit,
            routines_limit=self._config.routines_limit,
            source_labels=source_labels,
            reference_datetime=reference_datetime,
            include_archived=include_archived,
            emotional_state=emotional_state,
        )
        system_prompt = self._config.system_instruction
        user_prompt = _build_user_prompt(
            context=context,
            question=normalized_question,
            max_prompt_memories=self._config.max_prompt_memories,
        )
        messages = (
            ConversationMessage(role="system", content=system_prompt),
            *normalized_history,
            ConversationMessage(role="user", content=user_prompt),
        )

        started_at = perf_counter()
        generation = self._backend.generate(
            messages=messages,
            model_name=self._config.model_name,
            temperature=self._config.temperature,
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        if generation.latency_ms is None:
            generation = ConversationGeneration(
                text=generation.text,
                model_name=generation.model_name,
                latency_ms=elapsed_ms,
                finish_reason=generation.finish_reason,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
            )

        trace = ConversationTrace(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            retrieved_memories=_retrieved_memories(context, self._config.max_prompt_memories),
            dropped_hits=context.memory_recall.dropped_hits,
            total_semantic_hits=context.memory_recall.total_semantic_hits,
        )
        return ConversationResponse(
            patient_id=normalized_patient_id,
            question=normalized_question,
            answer=generation.text,
            generation=generation,
            trace=trace,
            context=context,
        )


def _normalize_history(
    conversation_history: tuple[ConversationMessage, ...],
) -> tuple[ConversationMessage, ...]:
    normalized_history: list[ConversationMessage] = []
    for message in conversation_history:
        if message.role == "system":
            raise ValueError("conversation_history must not contain system messages")
        normalized_history.append(message)
    return tuple(normalized_history)


def _build_user_prompt(
    *,
    context: PatientReorientationContext,
    question: str,
    max_prompt_memories: int,
) -> str:
    trusted_people_lines = _trusted_people_lines(context)
    routine_lines = _routine_lines(context)
    memory_lines = _memory_lines(context, max_prompt_memories=max_prompt_memories)

    prompt_sections = [
        "Contexte patient",
        f"Nom affiche: {context.patient_display_name or 'inconnu'}",
        f"Nom prefere: {context.preferred_name or 'inconnu'}",
        f"Repres rassurants: {_join_values(context.anchors)}",
        f"Notes de soin: {_join_values(context.care_notes)}",
        "",
        "Proches de confiance",
        trusted_people_lines,
        "",
        "Routines utiles",
        routine_lines,
        "",
        "Souvenirs recuperes",
        memory_lines,
        "",
        f"Question patient: {question}",
        (
            "Instruction: reponds a la question en t'appuyant d'abord sur les "
            "souvenirs recuperes et les reperes patient. Si une information manque, "
            "dis-le sans inventer et recentre la personne avec douceur."
        ),
    ]
    return "\n".join(prompt_sections).strip()


def _trusted_people_lines(context: PatientReorientationContext) -> str:
    if not context.trusted_people:
        return "Aucun proche de confiance disponible."

    lines = []
    for person in context.trusted_people:
        parts = [person.name]
        if person.relationship_to_patient:
            parts.append(person.relationship_to_patient)
        parts.append(f"importance_emotionnelle={person.emotional_significance:.2f}")
        if person.notes:
            parts.append(f"notes={person.notes}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def _routine_lines(context: PatientReorientationContext) -> str:
    if not context.routines:
        return "Aucune routine disponible."

    lines = []
    for routine in context.routines:
        parts = [routine.title]
        if routine.schedule:
            parts.append(f"horaire={routine.schedule}")
        if routine.temporal_label:
            parts.append(f"temporalite={routine.temporal_label}")
        if routine.place_name:
            parts.append(f"lieu={routine.place_name}")
        if routine.cue:
            parts.append(f"indice={routine.cue}")
        if routine.support_strategy:
            parts.append(f"aide={routine.support_strategy}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def _memory_lines(
    context: PatientReorientationContext,
    *,
    max_prompt_memories: int,
) -> str:
    hits = context.memory_recall.hits[:max_prompt_memories]
    if not hits:
        return "Aucun souvenir pertinent n'a ete retrouve."

    lines = []
    for index, hit in enumerate(hits, start=1):
        parts = [
            f"{index}. [{hit.source_label}] {hit.source_display_name}",
            f"resume={hit.summary}",
            f"score={hit.ranking_score:.4f}",
        ]
        if hit.related_people:
            parts.append(f"personnes={', '.join(hit.related_people)}")
        if hit.related_places:
            parts.append(f"lieux={', '.join(hit.related_places)}")
        if hit.related_emotions:
            parts.append(f"emotions={', '.join(hit.related_emotions)}")
        if hit.related_routines:
            parts.append(f"routines={', '.join(hit.related_routines)}")
        if hit.related_episodes:
            parts.append(f"episodes={', '.join(hit.related_episodes)}")
        if hit.score_breakdown is not None and hit.score_breakdown.signals:
            parts.append(f"indices={', '.join(hit.score_breakdown.signals)}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _join_values(values: tuple[str, ...]) -> str:
    if not values:
        return "aucun"
    return "; ".join(values)


def _retrieved_memories(
    context: PatientReorientationContext,
    max_prompt_memories: int,
) -> tuple[RetrievedMemoryEvidence, ...]:
    evidences = []
    for hit in context.memory_recall.hits[:max_prompt_memories]:
        signals = ()
        if hit.score_breakdown is not None:
            signals = hit.score_breakdown.signals
        evidences.append(
            RetrievedMemoryEvidence(
                source_node_id=hit.source_node_id,
                source_label=hit.source_label,
                source_display_name=hit.source_display_name,
                summary=hit.summary,
                ranking_score=hit.ranking_score,
                signals=signals,
                related_people=hit.related_people,
                related_places=hit.related_places,
                related_emotions=hit.related_emotions,
                related_routines=hit.related_routines,
                related_episodes=hit.related_episodes,
            )
        )
    return tuple(evidences)
