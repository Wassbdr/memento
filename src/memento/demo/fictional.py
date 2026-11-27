"""Seeded generator for a fictitious patient memory graph used by the demo.

Every person, place and event produced here is invented. The same seed always
produces the same snapshot, so demo questions, answers and timings can be
regenerated from a clean clone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import random

from memento.memory import (
    AffectiveState,
    MemoryEpisode,
    PatientMemorySnapshot,
    PatientProfile,
    PersonProfile,
    PlaceProfile,
    RoutineProfile,
)

DEFAULT_DEMO_SEED = 7
DEMO_PATIENT_ID = "demo-patient"

_PATIENT_FIRST_NAMES = ("Odette", "Simone", "Paulette", "Germaine", "Lucienne")
_FAMILY_NAMES = ("Lemoine", "Garnier", "Fontaine", "Marchal", "Bertin")
_DAUGHTER_NAMES = ("Claire", "Nathalie", "Isabelle", "Sylvie", "Valérie")
_SON_NAMES = ("Philippe", "Thierry", "Laurent", "Pascal", "Olivier")
_GRANDSON_NAMES = ("Hugo", "Théo", "Nathan", "Louis", "Jules")
_NEIGHBOUR_NAMES = ("Monique Perrin", "Josette Caron", "Mireille Roux", "Ginette Colin")
_AIDE_NAMES = ("Samia Haddad", "Inès Morel", "Fatou Diallo", "Julie Masson")
_STREETS = ("rue des Tilleuls", "rue des Acacias", "rue du Moulin", "allée des Peupliers")
_TOWNS = ("Vernay-sur-Loire", "Saint-Aubin-les-Prés", "Montclar", "Villeneuve-du-Lac")
_SEASIDE = ("Port-Marin", "Plage des Sables Blancs", "Anse des Mouettes")


@dataclass(frozen=True)
class DemoQuestion:
    """One spoken demo question and the graph facts a grounded answer should cite."""

    question_id: str
    text: str
    expected_terms: tuple[str, ...]
    expected_node_ids: tuple[str, ...]


def build_fictional_snapshot(seed: int = DEFAULT_DEMO_SEED) -> PatientMemorySnapshot:
    """Return a deterministic, entirely fictitious patient memory snapshot."""

    rng = random.Random(seed)
    first_name = rng.choice(_PATIENT_FIRST_NAMES)
    family_name = rng.choice(_FAMILY_NAMES)
    daughter = rng.choice(_DAUGHTER_NAMES)
    son = rng.choice(_SON_NAMES)
    grandson = rng.choice(_GRANDSON_NAMES)
    neighbour = rng.choice(_NEIGHBOUR_NAMES)
    aide = rng.choice(_AIDE_NAMES)
    street = rng.choice(_STREETS)
    town = rng.choice(_TOWNS)
    seaside = rng.choice(_SEASIDE)

    birthday = date(2026, 5, 1) + timedelta(days=rng.randrange(0, 60))
    grandson_visit = date(2026, 10, 1) + timedelta(days=rng.randrange(0, 40))
    seaside_trip_year = rng.randrange(1972, 1990)

    return PatientMemorySnapshot(
        patient=PatientProfile(
            patient_id=DEMO_PATIENT_ID,
            display_name=f"{first_name} {family_name}",
            preferred_name=first_name,
            care_notes=(
                "Rassurer avant de recontextualiser.",
                "Parler lentement avec des phrases courtes.",
            ),
            anchors=(
                f"Vous habitez votre appartement {street} à {town}.",
                f"{daughter}, votre fille, vient déjeuner le dimanche.",
            ),
        ),
        people=(
            PersonProfile(
                person_id="daughter",
                name=f"{daughter} {family_name}",
                relationship_to_patient="sa fille",
                notes="Vient déjeuner tous les dimanches à midi et apporte un dessert.",
                emotional_significance=0.97,
            ),
            PersonProfile(
                person_id="son",
                name=f"{son} {family_name}",
                relationship_to_patient="son fils",
                notes="Habite loin et téléphone le jeudi soir.",
                emotional_significance=0.9,
            ),
            PersonProfile(
                person_id="grandson",
                name=f"{grandson} {family_name}",
                relationship_to_patient="son petit-fils",
                notes=f"Fils de {son}, étudiant, passe le mercredi après les cours.",
                emotional_significance=0.85,
            ),
            PersonProfile(
                person_id="neighbour",
                name=neighbour,
                relationship_to_patient="sa voisine",
                notes="Habite sur le même palier et prend le café le mardi.",
                emotional_significance=0.6,
            ),
            PersonProfile(
                person_id="aide",
                name=aide,
                relationship_to_patient="son aide à domicile",
                notes="Vient le matin en semaine pour les médicaments.",
                emotional_significance=0.78,
            ),
        ),
        places=(
            PlaceProfile(
                place_id="home",
                name=f"Appartement {street}",
                category="home",
                notes=f"Troisième étage avec ascenseur, à {town}.",
            ),
            PlaceProfile(
                place_id="kitchen",
                name="Cuisine",
                category="home_room",
                notes="Table ronde près de la fenêtre.",
            ),
            PlaceProfile(
                place_id="guinguette",
                name="La Guinguette du Lac",
                category="restaurant",
                notes=f"Restaurant au bord du lac de {town}.",
            ),
            PlaceProfile(
                place_id="seaside",
                name=seaside,
                category="vacation",
                notes="Bord de mer des vacances en famille.",
            ),
        ),
        routines=(
            RoutineProfile(
                routine_id="sunday_lunch",
                title="Déjeuner du dimanche",
                schedule="Le dimanche à 12:30",
                description=f"{daughter} vient déjeuner et apporte un dessert.",
                cue="La table est mise avec la nappe jaune.",
                support_strategy=f"Rappeler que {daughter} arrive à midi et qu'il n'y a rien à préparer.",
                place_id="kitchen",
            ),
            RoutineProfile(
                routine_id="morning_meds",
                title="Médicaments du matin",
                schedule="Tous les jours à 09:00",
                description=f"{aide.split()[0]} aide à prendre les médicaments du pilulier bleu.",
                cue="Le pilulier bleu est sur la table de la cuisine.",
                place_id="kitchen",
            ),
        ),
        episodes=(
            MemoryEpisode(
                episode_id="birthday_80",
                title=f"Les quatre-vingts ans de {first_name}",
                narrative=(
                    f"Toute la famille a fêté les quatre-vingts ans de {first_name} à La Guinguette du Lac. "
                    f"{grandson} a joué de la guitare et {daughter} a apporté un gâteau aux framboises."
                ),
                happened_on=birthday.isoformat(),
                people_ids=("daughter", "son", "grandson"),
                place_id="guinguette",
                emotions=(AffectiveState(label="joie", valence=0.95, intensity=0.85),),
                tags=("famille", "anniversaire"),
            ),
            MemoryEpisode(
                episode_id="grandson_visit",
                title=f"Visite de {grandson}",
                narrative=f"{grandson} est passé après les cours et a réparé la radio de la cuisine.",
                happened_on=grandson_visit.isoformat(),
                people_ids=("grandson",),
                place_id="kitchen",
                emotions=(AffectiveState(label="fierté", valence=0.8, intensity=0.6),),
                tags=("famille", "visite"),
            ),
            MemoryEpisode(
                episode_id="seaside_holidays",
                title=f"Vacances à {seaside}",
                narrative=(
                    f"En {seaside_trip_year}, {first_name} a passé l'été à {seaside} "
                    f"avec ses enfants {daughter} et {son}."
                ),
                happened_on=f"{seaside_trip_year}-08-01",
                people_ids=("daughter", "son"),
                place_id="seaside",
                emotions=(AffectiveState(label="nostalgie", valence=0.6, intensity=0.5),),
                tags=("vacances", "souvenir"),
            ),
        ),
    )


def demo_questions(snapshot: PatientMemorySnapshot) -> tuple[DemoQuestion, ...]:
    """Questions asked aloud in the demo, with the facts the answer must use."""

    people = {person.person_id: person for person in snapshot.people}
    daughter_first = people["daughter"].name.split()[0]
    grandson_first = people["grandson"].name.split()[0]
    return (
        DemoQuestion(
            question_id="q1_sunday",
            text="Qui vient déjeuner avec moi le dimanche ?",
            expected_terms=(daughter_first,),
            expected_node_ids=("person:daughter", "routine:sunday_lunch"),
        ),
        DemoQuestion(
            question_id="q2_grandson",
            text="Comment s'appelle mon petit-fils ?",
            expected_terms=(grandson_first,),
            expected_node_ids=("person:grandson",),
        ),
        DemoQuestion(
            question_id="q3_birthday",
            text="Où est-ce que j'ai fêté mes quatre-vingts ans ?",
            expected_terms=("Guinguette",),
            expected_node_ids=("episode:birthday_80", "place:guinguette"),
        ),
    )
