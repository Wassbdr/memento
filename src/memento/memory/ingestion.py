"""Snapshot validation and reconciliation prior to memory synchronization."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Iterable

from .models import (
    AffectiveState,
    MemoryEpisode,
    PatientMemorySnapshot,
    PersonProfile,
    PlaceProfile,
    RoutineProfile,
)
from .sync_types import MemoryIngestionIssue, MemoryIngestionReport


def reconcile_snapshot(
    snapshot: PatientMemorySnapshot,
) -> tuple[PatientMemorySnapshot, MemoryIngestionReport]:
    """Normalize one incoming snapshot and reconcile obvious conflicts.

    This step is intentionally deterministic and conservative:
    - merge duplicate people and places based on semantic keys,
    - merge routines with same title and flag schedule conflicts,
    - merge episodes with same title/date and unify references.
    """

    people, people_aliases, people_issues, merged_people = _reconcile_people(snapshot.people)
    places, place_aliases, place_issues, merged_places = _reconcile_places(snapshot.places)
    routines, routine_issues, merged_routines = _reconcile_routines(
        snapshot.routines,
        place_aliases=place_aliases,
    )
    episodes, episode_issues, merged_episodes = _reconcile_episodes(
        snapshot.episodes,
        people_aliases=people_aliases,
        place_aliases=place_aliases,
    )

    reconciled_snapshot = PatientMemorySnapshot(
        patient=snapshot.patient,
        people=tuple(people),
        places=tuple(places),
        routines=tuple(routines),
        episodes=tuple(episodes),
    )

    report = MemoryIngestionReport(
        merged_people=merged_people,
        merged_places=merged_places,
        merged_routines=merged_routines,
        merged_episodes=merged_episodes,
        issues=tuple(people_issues + place_issues + routine_issues + episode_issues),
    )
    return reconciled_snapshot, report


def _reconcile_people(
    people: tuple[PersonProfile, ...],
) -> tuple[list[PersonProfile], dict[str, str], list[MemoryIngestionIssue], int]:
    key_to_index: dict[tuple[str, str], int] = {}
    merged_people: list[PersonProfile] = []
    aliases: dict[str, str] = {}
    issues: list[MemoryIngestionIssue] = []
    merged_count = 0

    for person in people:
        key = (_normalize_text(person.name), _normalize_text(person.relationship_to_patient))
        index = key_to_index.get(key)
        if index is None:
            key_to_index[key] = len(merged_people)
            merged_people.append(person)
            aliases[person.person_id] = person.person_id
            continue

        merged_count += 1
        existing = merged_people[index]
        aliases[person.person_id] = existing.person_id
        merged_people[index] = PersonProfile(
            person_id=existing.person_id,
            name=_prefer_richer_text(existing.name, person.name),
            relationship_to_patient=_prefer_richer_text(
                existing.relationship_to_patient,
                person.relationship_to_patient,
            ),
            notes=_merge_free_text(existing.notes, person.notes),
            emotional_significance=max(existing.emotional_significance, person.emotional_significance),
        )
        issues.append(
            MemoryIngestionIssue(
                issue_type="duplicate-person-merged",
                entity_ids=(existing.person_id, person.person_id),
                detail=(
                    f"Person profiles share the same semantic identity ({person.name} / "
                    f"{person.relationship_to_patient})."
                ),
                resolution=f"Merged into {existing.person_id}",
            )
        )

    return merged_people, aliases, issues, merged_count


def _reconcile_places(
    places: tuple[PlaceProfile, ...],
) -> tuple[list[PlaceProfile], dict[str, str], list[MemoryIngestionIssue], int]:
    key_to_index: dict[tuple[str, str], int] = {}
    merged_places: list[PlaceProfile] = []
    aliases: dict[str, str] = {}
    issues: list[MemoryIngestionIssue] = []
    merged_count = 0

    for place in places:
        key = (_normalize_text(place.name), _normalize_text(place.category))
        index = key_to_index.get(key)
        if index is None:
            key_to_index[key] = len(merged_places)
            merged_places.append(place)
            aliases[place.place_id] = place.place_id
            continue

        merged_count += 1
        existing = merged_places[index]
        aliases[place.place_id] = existing.place_id
        merged_places[index] = PlaceProfile(
            place_id=existing.place_id,
            name=_prefer_richer_text(existing.name, place.name),
            category=_prefer_richer_text(existing.category, place.category),
            notes=_merge_free_text(existing.notes, place.notes),
        )
        issues.append(
            MemoryIngestionIssue(
                issue_type="duplicate-place-merged",
                entity_ids=(existing.place_id, place.place_id),
                detail=f"Place profiles overlap on ({place.name}, {place.category}).",
                resolution=f"Merged into {existing.place_id}",
            )
        )

    return merged_places, aliases, issues, merged_count


def _reconcile_routines(
    routines: tuple[RoutineProfile, ...],
    *,
    place_aliases: dict[str, str],
) -> tuple[list[RoutineProfile], list[MemoryIngestionIssue], int]:
    key_to_index: dict[str, int] = {}
    merged_routines: list[RoutineProfile] = []
    issues: list[MemoryIngestionIssue] = []
    merged_count = 0

    for routine in routines:
        normalized_place_id = _normalize_optional_id(routine.place_id, aliases=place_aliases)
        normalized_routine = replace(routine, place_id=normalized_place_id)

        key = _normalize_text(normalized_routine.title)
        index = key_to_index.get(key)
        if index is None:
            key_to_index[key] = len(merged_routines)
            merged_routines.append(normalized_routine)
            continue

        merged_count += 1
        existing = merged_routines[index]
        has_schedule_conflict = _normalize_text(existing.schedule) != _normalize_text(normalized_routine.schedule)
        if has_schedule_conflict:
            issues.append(
                MemoryIngestionIssue(
                    issue_type="routine-schedule-conflict",
                    entity_ids=(existing.routine_id, normalized_routine.routine_id),
                    detail=(
                        f"Routine '{existing.title}' has contradictory schedules: "
                        f"'{existing.schedule}' vs '{normalized_routine.schedule}'."
                    ),
                    resolution="Kept the richer routine payload and preserved one routine id.",
                )
            )

        merged_routines[index] = _merge_routine(existing, normalized_routine)
        issues.append(
            MemoryIngestionIssue(
                issue_type="duplicate-routine-merged",
                entity_ids=(existing.routine_id, normalized_routine.routine_id),
                detail=f"Routine '{existing.title}' appears multiple times in snapshot.",
                resolution=f"Merged into {existing.routine_id}",
            )
        )

    return merged_routines, issues, merged_count


def _reconcile_episodes(
    episodes: tuple[MemoryEpisode, ...],
    *,
    people_aliases: dict[str, str],
    place_aliases: dict[str, str],
) -> tuple[list[MemoryEpisode], list[MemoryIngestionIssue], int]:
    key_to_index: dict[tuple[str, str], int] = {}
    merged_episodes: list[MemoryEpisode] = []
    issues: list[MemoryIngestionIssue] = []
    merged_count = 0

    for episode in episodes:
        remapped_people = _dedupe_preserve_order(
            people_aliases.get(person_id, person_id)
            for person_id in episode.people_ids
        )
        remapped_place_id = _normalize_optional_id(episode.place_id, aliases=place_aliases)
        normalized_episode = replace(
            episode,
            people_ids=remapped_people,
            place_id=remapped_place_id,
        )

        key = (
            _normalize_text(normalized_episode.title),
            _normalize_text(normalized_episode.happened_on),
        )
        index = key_to_index.get(key)
        if index is None:
            key_to_index[key] = len(merged_episodes)
            merged_episodes.append(normalized_episode)
            continue

        merged_count += 1
        existing = merged_episodes[index]
        merged_episodes[index] = _merge_episode(existing, normalized_episode)
        issues.append(
            MemoryIngestionIssue(
                issue_type="duplicate-episode-merged",
                entity_ids=(existing.episode_id, normalized_episode.episode_id),
                detail=(
                    f"Episode '{existing.title}' appears multiple times for date "
                    f"'{existing.happened_on or 'unknown'}'."
                ),
                resolution=f"Merged into {existing.episode_id}",
            )
        )

    return merged_episodes, issues, merged_count


def _merge_routine(primary: RoutineProfile, candidate: RoutineProfile) -> RoutineProfile:
    primary_score = _routine_information_score(primary)
    candidate_score = _routine_information_score(candidate)
    preferred_schedule = primary.schedule if primary_score >= candidate_score else candidate.schedule

    return RoutineProfile(
        routine_id=primary.routine_id,
        title=_prefer_richer_text(primary.title, candidate.title),
        schedule=preferred_schedule,
        description=_prefer_richer_text(primary.description, candidate.description),
        cue=_prefer_richer_text(primary.cue, candidate.cue),
        support_strategy=_prefer_richer_text(primary.support_strategy, candidate.support_strategy),
        place_id=primary.place_id or candidate.place_id,
        last_validated_on=_latest_iso_date(primary.last_validated_on, candidate.last_validated_on),
        archived_on=_latest_iso_date(primary.archived_on, candidate.archived_on),
    )


def _merge_episode(primary: MemoryEpisode, candidate: MemoryEpisode) -> MemoryEpisode:
    emotions_by_label: dict[str, AffectiveState] = {
        _normalize_text(emotion.label): emotion
        for emotion in primary.emotions
    }
    for emotion in candidate.emotions:
        key = _normalize_text(emotion.label)
        current = emotions_by_label.get(key)
        if current is None or emotion.intensity > current.intensity:
            emotions_by_label[key] = emotion

    return MemoryEpisode(
        episode_id=primary.episode_id,
        title=_prefer_richer_text(primary.title, candidate.title),
        narrative=_prefer_richer_text(primary.narrative, candidate.narrative),
        happened_on=primary.happened_on or candidate.happened_on,
        people_ids=_dedupe_preserve_order(primary.people_ids + candidate.people_ids),
        place_id=primary.place_id or candidate.place_id,
        emotions=tuple(
            sorted(
                emotions_by_label.values(),
                key=lambda item: _normalize_text(item.label),
            )
        ),
        tags=_dedupe_preserve_order(primary.tags + candidate.tags),
        last_validated_on=_latest_iso_date(primary.last_validated_on, candidate.last_validated_on),
        archived_on=_latest_iso_date(primary.archived_on, candidate.archived_on),
    )


def _routine_information_score(routine: RoutineProfile) -> int:
    score = 0
    if routine.schedule.strip():
        score += 3
    if routine.description.strip():
        score += 2
    if routine.cue.strip():
        score += 1
    if routine.support_strategy.strip():
        score += 1
    if routine.place_id is not None:
        score += 1
    if routine.last_validated_on.strip():
        score += 1
    return score


def _normalize_optional_id(value: str | None, *, aliases: dict[str, str]) -> str | None:
    if value is None:
        return None
    normalized_value = value.strip()
    if not normalized_value:
        return None
    return aliases.get(normalized_value, normalized_value)


def _merge_free_text(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if _normalize_text(left_clean) == _normalize_text(right_clean):
        return left_clean
    return f"{left_clean} | {right_clean}"


def _prefer_richer_text(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if len(right_clean) > len(left_clean):
        return right_clean
    return left_clean


def _dedupe_preserve_order(values: Iterable[object]) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        items.append(text)
    return tuple(items)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _latest_iso_date(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean

    left_date = datetime.strptime(left_clean, "%Y-%m-%d")
    right_date = datetime.strptime(right_clean, "%Y-%m-%d")
    if right_date > left_date:
        return right_clean
    return left_clean
