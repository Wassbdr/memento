"""Semantic hit hydration and ranking using the patient knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .emotion import EmotionalState
from .graph import PersonalMemoryGraph
from .semantic import SemanticSearchHit
from .sync_types import MemoryContextHit, MemoryRecall, MemoryScoreBreakdown
from .temporal import (
    days_since_date,
    recency_bonus,
    routine_temporal_label,
    routine_time_bonus,
    validation_staleness_penalty,
    minutes_until_next_occurrence,
)
from .weighting import ResolvedWeightProfile, resolve_weight_profile
_TRUSTED_PERSON_EMOTIONAL_THRESHOLD = 0.75


@dataclass(frozen=True)
class RecallNodeContext:
    """Hydration-ready context for one source node."""

    source_node_id: str
    source_label: str
    source_display_name: str
    source_properties: dict[str, object]
    related_people: tuple[str, ...]
    related_places: tuple[str, ...]
    related_emotions: tuple[str, ...]
    related_routines: tuple[str, ...]
    related_episodes: tuple[str, ...]
    emotion_intensities: tuple[float, ...]


@dataclass(frozen=True)
class _MutableGraphContext:
    source_node_id: str
    source_label: str
    source_display_name: str
    source_properties: dict[str, object]
    related_people: set[str]
    related_places: set[str]
    related_emotions: set[str]
    related_routines: set[str]
    related_episodes: set[str]
    emotion_intensities: list[float]


def build_memory_recall(
    *,
    query: str,
    patient_id: str,
    graph: PersonalMemoryGraph | None,
    semantic_hits: tuple[SemanticSearchHit, ...],
    reference_datetime: datetime | None = None,
    include_archived: bool = False,
    emotional_state: EmotionalState | None = None,
    prefetched_contexts: dict[str, RecallNodeContext] | None = None,
    anchor_terms: tuple[str, ...] | None = None,
    trusted_people: set[str] | None = None,
) -> MemoryRecall:
    """Hydrate semantic hits with graph context and return a ranked recall object.

    Invalid hits are skipped to keep the retrieval path resilient when the semantic
    store contains stale references.
    """

    effective_datetime = reference_datetime or datetime.now()
    resolved_weights = resolve_weight_profile(
        reference_datetime=effective_datetime,
        emotional_state=emotional_state,
    )

    if prefetched_contexts is None:
        if graph is None:
            raise ValueError("graph is required when prefetched_contexts is not provided")
        node_contexts = _build_graph_contexts(
            graph,
            tuple(hit.document.source_node_id for hit in semantic_hits),
        )
    else:
        node_contexts = prefetched_contexts

    if anchor_terms is None or trusted_people is None:
        graph_anchor_terms = ()
        graph_trusted_people: set[str] = set()
        if graph is not None:
            graph_anchor_terms, graph_trusted_people = _patient_relevance_context(graph, patient_id)
        if anchor_terms is None:
            anchor_terms = graph_anchor_terms
        if trusted_people is None:
            trusted_people = graph_trusted_people

    normalized_anchor_terms = tuple(anchor.lower() for anchor in (anchor_terms or ()))
    normalized_trusted_people = {
        person_name.lower()
        for person_name in (trusted_people or set())
    }

    hydrated_hits: list[MemoryContextHit] = []
    dropped_hits = 0
    archived_filtered_hits = 0

    for hit in semantic_hits:
        context = node_contexts.get(hit.document.source_node_id)
        if context is None:
            dropped_hits += 1
            continue

        if not include_archived and _is_archived(context.source_properties):
            archived_filtered_hits += 1
            continue

        context_hit = MemoryContextHit(
            source_node_id=context.source_node_id,
            source_label=context.source_label,
            source_display_name=context.source_display_name,
            score=hit.score,
            summary=hit.document.text,
            related_people=context.related_people,
            related_places=context.related_places,
            related_emotions=context.related_emotions,
            related_routines=context.related_routines,
            related_episodes=context.related_episodes,
            ranking_score=hit.score,
        )

        score_breakdown = _build_score_breakdown(
            context_hit=context_hit,
            source_properties=context.source_properties,
            emotion_intensities=context.emotion_intensities,
            anchor_terms=normalized_anchor_terms,
            trusted_people=normalized_trusted_people,
            reference_datetime=effective_datetime,
            resolved_weights=resolved_weights,
        )
        hydrated_hits.append(
            replace(
                context_hit,
                ranking_score=score_breakdown.final_score,
                score_breakdown=score_breakdown,
            )
        )

    ranked_hits = tuple(sorted(hydrated_hits, key=_rank_key))
    return MemoryRecall(
        query=query,
        patient_id=patient_id,
        hits=ranked_hits,
        dropped_hits=dropped_hits,
        total_semantic_hits=len(semantic_hits),
        archived_filtered_hits=archived_filtered_hits,
    )


def _build_graph_contexts(
    graph: PersonalMemoryGraph,
    source_node_ids: tuple[str, ...],
) -> dict[str, RecallNodeContext]:
    unique_source_ids = tuple(sorted({source_node_id for source_node_id in source_node_ids}))
    if not unique_source_ids:
        return {}

    contexts: dict[str, _MutableGraphContext] = {}
    for source_node_id in unique_source_ids:
        source_node = graph.get_node(source_node_id)
        if source_node is None:
            continue
        contexts[source_node_id] = _MutableGraphContext(
            source_node_id=source_node_id,
            source_label=source_node.label,
            source_display_name=source_node.display_name,
            source_properties=dict(source_node.properties),
            related_people=set(),
            related_places=set(),
            related_emotions=set(),
            related_routines=set(),
            related_episodes=set(),
            emotion_intensities=[],
        )

    if not contexts:
        return {}

    tracked_ids = set(contexts.keys())

    for relation in graph.relations:
        source_node_id = None
        neighbor_node_id = None

        if relation.source_id in tracked_ids:
            source_node_id = relation.source_id
            neighbor_node_id = relation.target_id
        elif relation.target_id in tracked_ids:
            source_node_id = relation.target_id
            neighbor_node_id = relation.source_id

        if source_node_id is None or neighbor_node_id is None:
            continue

        neighbor_node = graph.get_node(neighbor_node_id)
        if neighbor_node is None:
            continue

        if neighbor_node.label == "Person":
            contexts[source_node_id].related_people.add(neighbor_node.display_name)
        elif neighbor_node.label == "Place":
            contexts[source_node_id].related_places.add(neighbor_node.display_name)
        elif neighbor_node.label == "Emotion":
            contexts[source_node_id].related_emotions.add(neighbor_node.display_name)
            intensity = _as_float(neighbor_node.properties.get("intensity"), default=None)
            if intensity is not None:
                contexts[source_node_id].emotion_intensities.append(max(0.0, min(1.0, intensity)))
        elif neighbor_node.label == "Routine":
            contexts[source_node_id].related_routines.add(neighbor_node.display_name)
        elif neighbor_node.label == "Episode":
            contexts[source_node_id].related_episodes.add(neighbor_node.display_name)

    result: dict[str, RecallNodeContext] = {}
    for source_node_id, context in contexts.items():
        result[source_node_id] = RecallNodeContext(
            source_node_id=context.source_node_id,
            source_label=context.source_label,
            source_display_name=context.source_display_name,
            source_properties=context.source_properties,
            related_people=tuple(sorted(context.related_people)),
            related_places=tuple(sorted(context.related_places)),
            related_emotions=tuple(sorted(context.related_emotions)),
            related_routines=tuple(sorted(context.related_routines)),
            related_episodes=tuple(sorted(context.related_episodes)),
            emotion_intensities=tuple(context.emotion_intensities),
        )
    return result


def _build_score_breakdown(
    *,
    context_hit: MemoryContextHit,
    source_properties: dict[str, object],
    emotion_intensities: tuple[float, ...],
    anchor_terms: tuple[str, ...],
    trusted_people: set[str],
    reference_datetime: datetime,
    resolved_weights: ResolvedWeightProfile,
) -> MemoryScoreBreakdown:
    profile = resolved_weights.profile

    semantic_score = round(context_hit.score * profile.semantic_weight, 4)

    context_bonus = round(_context_weight(context_hit) * profile.context_weight_factor, 4)

    affective_bonus = 0.0
    if emotion_intensities:
        affective_bonus = round(
            min(
                sum(emotion_intensities) / len(emotion_intensities) * profile.affective_weight_factor,
                0.2,
            ),
            4,
        )

    trusted_matches = tuple(
        person_name
        for person_name in context_hit.related_people
        if person_name.lower() in trusted_people
    )
    trusted_people_bonus = round(
        min(
            len(trusted_matches) * profile.trusted_person_match_bonus,
            profile.max_trusted_bonus,
        ),
        4,
    )

    routine_minutes_until = None
    routine_bonus = 0.0
    if context_hit.source_label == "Routine":
        routine_minutes_until = minutes_until_next_occurrence(
            _as_clean_string(source_properties.get("schedule")),
            reference_datetime=reference_datetime,
        )
        routine_bonus = round(
            routine_time_bonus(routine_minutes_until) * profile.routine_time_multiplier,
            4,
        )

    episode_recency_days = None
    recency_score = 0.0
    if context_hit.source_label == "Episode":
        episode_recency_days = days_since_date(
            _as_clean_string(source_properties.get("happened_on")),
            reference_date=reference_datetime.date(),
        )
        recency_score = round(
            recency_bonus(episode_recency_days) * profile.recency_multiplier,
            4,
        )

    validation_recency_days = days_since_date(
        _as_clean_string(source_properties.get("last_validated_on")),
        reference_date=reference_datetime.date(),
    )
    staleness_penalty = round(
        validation_staleness_penalty(validation_recency_days) * profile.staleness_penalty_multiplier,
        4,
    )

    summary_lower = context_hit.summary.lower()
    anchor_match_count = sum(1 for anchor in anchor_terms if anchor in summary_lower)
    anchor_bonus = round(
        min(anchor_match_count * profile.anchor_match_bonus, profile.max_anchor_bonus),
        4,
    )

    signals: list[str] = []
    if context_bonus > 0:
        signals.append("graph_context")
    if affective_bonus > 0:
        signals.append("affective_signal")
    if trusted_people_bonus > 0:
        signals.append("trusted_person_match")
    if routine_minutes_until is not None:
        signals.append(f"routine_{routine_temporal_label(routine_minutes_until)}")
    if episode_recency_days is not None:
        signals.append("recent_memory" if episode_recency_days <= 180 else "older_memory")
    if anchor_bonus > 0:
        signals.append("anchor_match")
    if staleness_penalty > 0:
        signals.append("stale_memory")

    final_score = round(
        semantic_score
        + context_bonus
        + affective_bonus
        + trusted_people_bonus
        + routine_bonus
        + recency_score
        + anchor_bonus,
        4,
    )
    final_score = round(
        final_score - staleness_penalty,
        4,
    )

    return MemoryScoreBreakdown(
        semantic_score=semantic_score,
        context_bonus=context_bonus,
        affective_bonus=affective_bonus,
        trusted_people_bonus=trusted_people_bonus,
        routine_time_bonus=routine_bonus,
        recency_bonus=recency_score,
        anchor_bonus=anchor_bonus,
        staleness_penalty=staleness_penalty,
        final_score=final_score,
        routine_minutes_until=routine_minutes_until,
        episode_recency_days=episode_recency_days,
        validation_recency_days=validation_recency_days,
        weight_profile=profile.name,
        weight_signals=resolved_weights.signals,
        signals=tuple(signals),
    )


def _context_weight(hit: MemoryContextHit) -> int:
    return (
        len(hit.related_people)
        + len(hit.related_places)
        + len(hit.related_emotions)
        + len(hit.related_routines)
        + len(hit.related_episodes)
    )


def _rank_key(hit: MemoryContextHit) -> tuple[float, int, str, str]:
    weight = _context_weight(hit)
    return (-hit.ranking_score, -weight, hit.source_label, hit.source_node_id)


def _patient_relevance_context(
    graph: PersonalMemoryGraph,
    patient_id: str,
) -> tuple[tuple[str, ...], set[str]]:
    patient_node = None
    for node in graph.nodes:
        if node.label != "Patient":
            continue
        if str(node.properties.get("patient_id", "")) == patient_id:
            patient_node = node
            break

    if patient_node is None:
        return (), set()

    anchors = tuple(
        anchor.lower()
        for anchor in _as_string_tuple(patient_node.properties.get("anchors"))
    )
    trusted_people: set[str] = set()
    for neighbor in graph.neighbors(patient_node.node_id):
        if neighbor.direction != "outgoing":
            continue
        if neighbor.relation_type != "KNOWS" or neighbor.node.label != "Person":
            continue
        emotional_significance = _as_float(
            neighbor.node.properties.get("emotional_significance"),
            default=0.0,
        )
        if emotional_significance < _TRUSTED_PERSON_EMOTIONAL_THRESHOLD:
            continue
        trusted_people.add(neighbor.node.display_name.lower())

    return anchors, trusted_people


def _is_archived(source_properties: dict[str, object]) -> bool:
    archived_on = _as_clean_string(source_properties.get("archived_on"))
    return bool(archived_on)


def _as_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(
            item.strip()
            for item in (str(raw_item) for raw_item in value)
            if item.strip()
        )
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _as_float(value: object, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
