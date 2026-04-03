"""Domain models for the patient memory graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AffectiveState:
    """One emotional marker attached to a memory episode."""

    label: str
    valence: float
    intensity: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError("valence must be between -1.0 and 1.0")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PatientProfile:
    """Core identity and care preferences for the patient."""

    patient_id: str
    display_name: str
    preferred_name: str = ""
    care_notes: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patient_id.strip():
            raise ValueError("patient_id must not be empty")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty")


@dataclass(frozen=True)
class PersonProfile:
    """One close contact in the patient's personal graph."""

    person_id: str
    name: str
    relationship_to_patient: str
    notes: str = ""
    emotional_significance: float = 0.5

    def __post_init__(self) -> None:
        if not self.person_id.strip():
            raise ValueError("person_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.relationship_to_patient.strip():
            raise ValueError("relationship_to_patient must not be empty")
        if not 0.0 <= self.emotional_significance <= 1.0:
            raise ValueError("emotional_significance must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PlaceProfile:
    """One place that matters in the patient's routines or memories."""

    place_id: str
    name: str
    category: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.place_id.strip():
            raise ValueError("place_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.category.strip():
            raise ValueError("category must not be empty")


@dataclass(frozen=True)
class RoutineProfile:
    """A reassuring routine anchored in time and place."""

    routine_id: str
    title: str
    schedule: str
    description: str
    cue: str = ""
    support_strategy: str = ""
    place_id: str | None = None
    last_validated_on: str = ""
    archived_on: str = ""

    def __post_init__(self) -> None:
        if not self.routine_id.strip():
            raise ValueError("routine_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.schedule.strip():
            raise ValueError("schedule must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        _validate_optional_iso_date(self.last_validated_on, field_name="last_validated_on")
        _validate_optional_iso_date(self.archived_on, field_name="archived_on")


@dataclass(frozen=True)
class MemoryEpisode:
    """A structured autobiographical memory."""

    episode_id: str
    title: str
    narrative: str
    happened_on: str = ""
    people_ids: tuple[str, ...] = ()
    place_id: str | None = None
    emotions: tuple[AffectiveState, ...] = ()
    tags: tuple[str, ...] = ()
    last_validated_on: str = ""
    archived_on: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.narrative.strip():
            raise ValueError("narrative must not be empty")
        if len(set(self.people_ids)) != len(self.people_ids):
            raise ValueError("people_ids must be unique within one episode")
        _validate_optional_iso_date(self.happened_on, field_name="happened_on")
        _validate_optional_iso_date(self.last_validated_on, field_name="last_validated_on")
        _validate_optional_iso_date(self.archived_on, field_name="archived_on")


@dataclass(frozen=True)
class PatientMemorySnapshot:
    """Full patient memory state to synchronize into graph and vector stores."""

    patient: PatientProfile
    people: tuple[PersonProfile, ...] = ()
    places: tuple[PlaceProfile, ...] = ()
    routines: tuple[RoutineProfile, ...] = ()
    episodes: tuple[MemoryEpisode, ...] = ()

    def __post_init__(self) -> None:
        _assert_unique_ids(
            values=tuple(person.person_id for person in self.people),
            message="people must have unique person_id values",
        )
        _assert_unique_ids(
            values=tuple(place.place_id for place in self.places),
            message="places must have unique place_id values",
        )
        _assert_unique_ids(
            values=tuple(routine.routine_id for routine in self.routines),
            message="routines must have unique routine_id values",
        )
        _assert_unique_ids(
            values=tuple(episode.episode_id for episode in self.episodes),
            message="episodes must have unique episode_id values",
        )

        person_ids = {person.person_id for person in self.people}
        place_ids = {place.place_id for place in self.places}

        for routine in self.routines:
            if routine.place_id is not None and routine.place_id not in place_ids:
                raise ValueError(f"routine {routine.routine_id} references an unknown place_id")

        for episode in self.episodes:
            missing_people = [person_id for person_id in episode.people_ids if person_id not in person_ids]
            if missing_people:
                raise ValueError(
                    f"episode {episode.episode_id} references unknown people_ids: {missing_people}"
                )
            if episode.place_id is not None and episode.place_id not in place_ids:
                raise ValueError(f"episode {episode.episode_id} references an unknown place_id")


def _assert_unique_ids(values: tuple[str, ...], message: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(message)


def _validate_optional_iso_date(value: str, *, field_name: str) -> None:
    text = value.strip()
    if not text:
        return
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format") from error
