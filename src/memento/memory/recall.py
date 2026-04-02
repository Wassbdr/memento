"""Semantic hit hydration and ranking using the patient knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .graph import PersonalMemoryGraph
from .semantic import SemanticSearchHit
from .sync_types import MemoryContextHit, MemoryRecall, MemoryScoreBreakdown
from .temporal import (
    days_since_date,
    recency_bonus,
    routine_temporal_label,
    routine_time_bonus,
    minutes_until_next_occurrence,
)

_CONTEXT_WEIGHT_FACTOR = 0.03
_AFFECTIVE_WEIGHT_FACTOR = 0.2
_TRUSTED_PERSON_MATCH_BONUS = 0.05
_MAX_TRUSTED_BONUS = 0.15
_ANCHOR_MATCH_BONUS = 0.06
_MAX_ANCHOR_BONUS = 0.18
_TRUSTED_PERSON_EMOTIONAL_THRESHOLD = 0.75


@dataclass(frozen=True)
class _HydratedHit:
    context_hit: MemoryContextHit
    source_properties: dict[str, object]
    emotion_intensities: tuple[float, ...]


def build_memory_recall(
    *,
    query: str,
    patient_id: str,
    graph: PersonalMemoryGraph,
    semantic_hits: tuple[SemanticSearchHit, ...],
    reference_datetime: datetime | None = None,
) -> MemoryRecall:
    """Hydrate semantic hits with graph context and return a ranked recall object.

    Invalid hits are skipped to keep the retrieval path resilient when the semantic
    store contains stale references.
    """

    effective_datetime = reference_datetime or datetime.now()
    anchor_terms, trusted_people = _patient_relevance_context(graph, patient_id)

    hydrated_hits: list[MemoryContextHit] = []
    dropped_hits = 0

    for hit in semantic_hits:
        try:
            hydrated_hit = _hydrate_hit(
                    graph,
                    hit.document.source_node_id,
                    hit.document.source_label,
                    hit.score,
                    hit.document.text,
            )
            score_breakdown = _build_score_breakdown(
                context_hit=hydrated_hit.context_hit,
                source_properties=hydrated_hit.source_properties,
                emotion_intensities=hydrated_hit.emotion_intensities,
                anchor_terms=anchor_terms,
                trusted_people=trusted_people,
                reference_datetime=effective_datetime,
            )
            hydrated_hits.append(
                replace(
                    hydrated_hit.context_hit,
                    ranking_score=score_breakdown.final_score,
                    score_breakdown=score_breakdown,
                )
            )
        except ValueError:
            dropped_hits += 1

    ranked_hits = tuple(sorted(hydrated_hits, key=_rank_key))
    return MemoryRecall(
        query=query,
        patient_id=patient_id,
        hits=ranked_hits,
        dropped_hits=dropped_hits,
        total_semantic_hits=len(semantic_hits),
    )


def _hydrate_hit(
    graph: PersonalMemoryGraph,
    source_node_id: str,
    source_label: str,
    score: float,
    summary: str,
) -> _HydratedHit:
    node = graph.get_node(source_node_id)
    if node is None:
        raise ValueError(f"source node not found in graph: {source_node_id}")

    related_people: set[str] = set()
    related_places: set[str] = set()
    related_emotions: set[str] = set()
    related_routines: set[str] = set()
    related_episodes: set[str] = set()
    emotion_intensities: list[float] = []

    for neighbor in graph.neighbors(source_node_id):
        if neighbor.node.label == "Person":
            related_people.add(neighbor.node.display_name)
        elif neighbor.node.label == "Place":
            related_places.add(neighbor.node.display_name)
        elif neighbor.node.label == "Emotion":
            related_emotions.add(neighbor.node.display_name)
            intensity = _as_float(neighbor.node.properties.get("intensity"), default=None)
            if intensity is not None:
                emotion_intensities.append(max(0.0, min(1.0, intensity)))
        elif neighbor.node.label == "Routine":
            related_routines.add(neighbor.node.display_name)
        elif neighbor.node.label == "Episode":
            related_episodes.add(neighbor.node.display_name)

    return _HydratedHit(
        context_hit=MemoryContextHit(
            source_node_id=source_node_id,
            source_label=source_label,
            source_display_name=node.display_name,
            score=score,
            summary=summary,
            related_people=tuple(sorted(related_people)),
            related_places=tuple(sorted(related_places)),
            related_emotions=tuple(sorted(related_emotions)),
            related_routines=tuple(sorted(related_routines)),
            related_episodes=tuple(sorted(related_episodes)),
            ranking_score=score,
        ),
        source_properties=dict(node.properties),
        emotion_intensities=tuple(emotion_intensities),
    )


def _build_score_breakdown(
    *,
    context_hit: MemoryContextHit,
    source_properties: dict[str, object],
    emotion_intensities: tuple[float, ...],
    anchor_terms: tuple[str, ...],
    trusted_people: set[str],
    reference_datetime: datetime,
) -> MemoryScoreBreakdown:
    semantic_score = round(context_hit.score, 4)

    context_bonus = round(_context_weight(context_hit) * _CONTEXT_WEIGHT_FACTOR, 4)

    affective_bonus = 0.0
    if emotion_intensities:
        affective_bonus = round(min(sum(emotion_intensities) / len(emotion_intensities) * _AFFECTIVE_WEIGHT_FACTOR, 0.2), 4)

    trusted_matches = tuple(
        person_name
        for person_name in context_hit.related_people
        if person_name.lower() in trusted_people
    )
    trusted_people_bonus = round(
        min(len(trusted_matches) * _TRUSTED_PERSON_MATCH_BONUS, _MAX_TRUSTED_BONUS),
        4,
    )

    routine_minutes_until = None
    routine_bonus = 0.0
    if context_hit.source_label == "Routine":
        routine_minutes_until = minutes_until_next_occurrence(
            _as_clean_string(source_properties.get("schedule")),
            reference_datetime=reference_datetime,
        )
        routine_bonus = round(routine_time_bonus(routine_minutes_until), 4)

    episode_recency_days = None
    recency_score = 0.0
    if context_hit.source_label == "Episode":
        episode_recency_days = days_since_date(
            _as_clean_string(source_properties.get("happened_on")),
            reference_date=reference_datetime.date(),
        )
        recency_score = round(recency_bonus(episode_recency_days), 4)

    summary_lower = context_hit.summary.lower()
    anchor_match_count = sum(1 for anchor in anchor_terms if anchor in summary_lower)
    anchor_bonus = round(min(anchor_match_count * _ANCHOR_MATCH_BONUS, _MAX_ANCHOR_BONUS), 4)

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

    return MemoryScoreBreakdown(
        semantic_score=semantic_score,
        context_bonus=context_bonus,
        affective_bonus=affective_bonus,
        trusted_people_bonus=trusted_people_bonus,
        routine_time_bonus=routine_bonus,
        recency_bonus=recency_score,
        anchor_bonus=anchor_bonus,
        final_score=final_score,
        routine_minutes_until=routine_minutes_until,
        episode_recency_days=episode_recency_days,
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
