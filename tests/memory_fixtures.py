from memento.memory import (
    AffectiveState,
    MemoryEpisode,
    PatientMemorySnapshot,
    PatientProfile,
    PersonProfile,
    PlaceProfile,
    RoutineProfile,
)


def build_snapshot() -> PatientMemorySnapshot:
    return PatientMemorySnapshot(
        patient=PatientProfile(
            patient_id="rose",
            display_name="Rose Martin",
            preferred_name="Mamie Rose",
            care_notes=("Rassurer avant de recontextualiser.",),
            anchors=("Appartement rue des Lilas", "Claire vient souvent le dimanche"),
        ),
        people=(
            PersonProfile(
                person_id="claire",
                name="Claire Martin",
                relationship_to_patient="sa fille",
                notes="Vient pour le dejeuner du dimanche.",
                emotional_significance=0.95,
            ),
        ),
        places=(
            PlaceProfile(
                place_id="cuisine",
                name="Cuisine",
                category="home_room",
                notes="Table ronde pres de la fenetre.",
            ),
        ),
        routines=(
            RoutineProfile(
                routine_id="breakfast",
                title="Petit-dejeuner",
                schedule="Tous les jours a 08:00",
                description="The noir et tartines avec confiture.",
                cue="La bouilloire siffle",
                support_strategy="Montrer la tasse rouge pour demarrer la routine.",
                place_id="cuisine",
            ),
        ),
        episodes=(
            MemoryEpisode(
                episode_id="dimanche",
                title="Dejeuner du dimanche",
                narrative="Claire apporte une tarte aux pommes et dejeune dans la cuisine.",
                happened_on="2024-01-07",
                people_ids=("claire",),
                place_id="cuisine",
                emotions=(AffectiveState(label="joie", valence=0.9, intensity=0.8),),
                tags=("famille", "repas"),
            ),
        ),
    )
