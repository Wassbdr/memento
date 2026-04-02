"""Knowledge-graph-first context extraction for patient reorientation."""

from __future__ import annotations

from datetime import datetime

from .graph import PersonalMemoryGraph
from .sync_types import (
    MemoryRecall,
    PatientReorientationContext,
    RoutineSupportContext,
    TrustedPersonContext,
)
from .temporal import minutes_until_next_occurrence, routine_temporal_label


def build_reorientation_context(
    *,
    graph: PersonalMemoryGraph,
    patient_id: str,
    recall: MemoryRecall,
    trusted_people_limit: int,
    routines_limit: int,
    reference_datetime: datetime | None = None,
) -> PatientReorientationContext:
    """Build one reorientation context from graph entities and recall evidence."""

    effective_datetime = reference_datetime or datetime.now()

    patient_node = _patient_node(graph, patient_id)
    if patient_node is None:
        raise ValueError(f"patient node not found in graph: {patient_id}")

    trusted_people = _trusted_people_for_patient(
        graph,
        patient_node.node_id,
        limit=trusted_people_limit,
    )
    routines = _routines_for_patient(
        graph,
        patient_node.node_id,
        limit=routines_limit,
        reference_datetime=effective_datetime,
    )

    return PatientReorientationContext(
        patient_id=patient_id,
        patient_display_name=patient_node.display_name,
        preferred_name=_as_clean_string(patient_node.properties.get("preferred_name")),
        anchors=_as_string_tuple(patient_node.properties.get("anchors")),
        care_notes=_as_string_tuple(patient_node.properties.get("care_notes")),
        trusted_people=trusted_people,
        routines=routines,
        memory_recall=recall,
    )


def _patient_node(graph: PersonalMemoryGraph, patient_id: str):
    for node in graph.nodes:
        if node.label != "Patient":
            continue
        if str(node.properties.get("patient_id", "")) == patient_id:
            return node
    return None


def _trusted_people_for_patient(
    graph: PersonalMemoryGraph,
    patient_node_id: str,
    *,
    limit: int,
) -> tuple[TrustedPersonContext, ...]:
    people: list[TrustedPersonContext] = []

    for neighbor in graph.neighbors(patient_node_id):
        if neighbor.direction != "outgoing":
            continue
        if neighbor.relation_type != "KNOWS" or neighbor.node.label != "Person":
            continue

        properties = neighbor.node.properties
        people.append(
            TrustedPersonContext(
                person_id=str(properties.get("id", "")),
                name=neighbor.node.display_name,
                relationship_to_patient=_as_clean_string(properties.get("relationship_to_patient")),
                emotional_significance=_as_float(properties.get("emotional_significance"), default=0.0),
                notes=_as_clean_string(properties.get("notes")),
            )
        )

    people.sort(
        key=lambda item: (
            -item.emotional_significance,
            item.relationship_to_patient,
            item.name,
        )
    )
    return tuple(people[:limit])


def _routines_for_patient(
    graph: PersonalMemoryGraph,
    patient_node_id: str,
    *,
    limit: int,
    reference_datetime: datetime,
) -> tuple[RoutineSupportContext, ...]:
    routines: list[RoutineSupportContext] = []

    for neighbor in graph.neighbors(patient_node_id):
        if neighbor.direction != "outgoing":
            continue
        if neighbor.relation_type != "FOLLOWS_ROUTINE" or neighbor.node.label != "Routine":
            continue

        routine_node = neighbor.node
        place_name = _routine_place_name(graph, routine_node.node_id)
        properties = routine_node.properties
        schedule = _as_clean_string(properties.get("schedule"))
        minutes_until = minutes_until_next_occurrence(
            schedule,
            reference_datetime=reference_datetime,
        )
        routines.append(
            RoutineSupportContext(
                routine_id=str(properties.get("id", "")),
                title=routine_node.display_name,
                schedule=schedule,
                cue=_as_clean_string(properties.get("cue")),
                support_strategy=_as_clean_string(properties.get("support_strategy")),
                place_name=place_name,
                minutes_until_next_occurrence=minutes_until,
                temporal_label=routine_temporal_label(minutes_until),
            )
        )

    routines.sort(
        key=lambda item: (
            item.minutes_until_next_occurrence is None,
            item.minutes_until_next_occurrence if item.minutes_until_next_occurrence is not None else 1_000_000,
            item.schedule,
            item.title,
            item.routine_id,
        )
    )
    return tuple(routines[:limit])


def _routine_place_name(graph: PersonalMemoryGraph, routine_node_id: str) -> str:
    for neighbor in graph.neighbors(routine_node_id):
        if neighbor.direction != "outgoing":
            continue
        if neighbor.relation_type == "HAPPENS_AT" and neighbor.node.label == "Place":
            return neighbor.node.display_name
    return ""


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


def _as_clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
