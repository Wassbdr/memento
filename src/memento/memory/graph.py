"""Graph schema and Neo4j-friendly exports for patient memory."""

from __future__ import annotations

from dataclasses import dataclass

from .models import MemoryEpisode, PatientMemorySnapshot


@dataclass(frozen=True)
class NodeSchema:
    """Definition of one graph node label."""

    label: str
    key_property: str
    required_properties: tuple[str, ...]


@dataclass(frozen=True)
class RelationSchema:
    """Definition of one graph relation type."""

    relation_type: str
    source_label: str
    target_label: str
    description: str


@dataclass(frozen=True)
class MemoryNode:
    """One graph node instance."""

    node_id: str
    label: str
    properties: dict[str, object]

    @property
    def display_name(self) -> str:
        for key in ("display_name", "preferred_name", "name", "title", "label", "id"):
            value = self.properties.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return self.node_id


@dataclass(frozen=True)
class MemoryRelation:
    """One graph edge instance."""

    source_id: str
    relation_type: str
    target_id: str
    properties: dict[str, object]


@dataclass(frozen=True)
class GraphNeighbor:
    """A neighboring node plus the relation that connects it."""

    relation_type: str
    direction: str
    node: MemoryNode


@dataclass(frozen=True)
class MemoryGraphSchema:
    """Initial graph schema for the patient memory domain."""

    nodes: tuple[NodeSchema, ...]
    relations: tuple[RelationSchema, ...]

    @classmethod
    def default(cls) -> MemoryGraphSchema:
        return cls(
            nodes=(
                NodeSchema("Patient", "id", ("id", "display_name")),
                NodeSchema("Person", "id", ("id", "name", "relationship_to_patient")),
                NodeSchema("Place", "id", ("id", "name", "category")),
                NodeSchema("Routine", "id", ("id", "title", "schedule")),
                NodeSchema("Episode", "id", ("id", "title", "narrative")),
                NodeSchema("Emotion", "id", ("id", "label", "valence", "intensity")),
            ),
            relations=(
                RelationSchema(
                    "KNOWS",
                    "Patient",
                    "Person",
                    "Patient-to-person anchor relation enriched with familiarity and notes.",
                ),
                RelationSchema(
                    "FOLLOWS_ROUTINE",
                    "Patient",
                    "Routine",
                    "Routine that stabilizes the patient in time and space.",
                ),
                RelationSchema(
                    "HAPPENS_AT",
                    "Routine",
                    "Place",
                    "Physical place where the routine usually happens.",
                ),
                RelationSchema(
                    "REMEMBERS",
                    "Patient",
                    "Episode",
                    "Autobiographical memory that can be recalled during a conversation.",
                ),
                RelationSchema(
                    "INVOLVES",
                    "Episode",
                    "Person",
                    "Person seen or mentioned in a memory episode.",
                ),
                RelationSchema(
                    "TOOK_PLACE_AT",
                    "Episode",
                    "Place",
                    "Place where the episode happened.",
                ),
                RelationSchema(
                    "EVOKES",
                    "Episode",
                    "Emotion",
                    "Emotion attached to an episode with affective metadata.",
                ),
            ),
        )

    def to_neo4j_cypher(self) -> str:
        """Render schema constraints and indexes for Neo4j."""

        lines: list[str] = ["// Neo4j schema for Memento memory graph"]
        for node in self.nodes:
            constraint_name = f"{node.label.lower()}_{node.key_property}_unique"
            lines.append(
                "CREATE CONSTRAINT "
                f"{constraint_name} IF NOT EXISTS FOR (n:{node.label}) REQUIRE n.{node.key_property} IS UNIQUE;"
            )
        lines.extend(
            (
                "CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name);",
                "CREATE INDEX place_name IF NOT EXISTS FOR (n:Place) ON (n.name);",
                "CREATE INDEX routine_title IF NOT EXISTS FOR (n:Routine) ON (n.title);",
                "CREATE INDEX episode_title IF NOT EXISTS FOR (n:Episode) ON (n.title);",
            )
        )
        return "\n".join(lines)


class PersonalMemoryGraph:
    """In-memory graph representation used for sync and retrieval."""

    def __init__(self, nodes: tuple[MemoryNode, ...], relations: tuple[MemoryRelation, ...]) -> None:
        self._nodes = {node.node_id: node for node in nodes}
        self._relations = relations

    @property
    def nodes(self) -> tuple[MemoryNode, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes.keys()))

    @property
    def relations(self) -> tuple[MemoryRelation, ...]:
        return self._relations

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> tuple[GraphNeighbor, ...]:
        neighbors: list[GraphNeighbor] = []
        for relation in self._relations:
            if relation.source_id == node_id:
                target = self._nodes[relation.target_id]
                neighbors.append(
                    GraphNeighbor(
                        relation_type=relation.relation_type,
                        direction="outgoing",
                        node=target,
                    )
                )
            elif relation.target_id == node_id:
                source = self._nodes[relation.source_id]
                neighbors.append(
                    GraphNeighbor(
                        relation_type=relation.relation_type,
                        direction="incoming",
                        node=source,
                    )
                )
        return tuple(
            sorted(
                neighbors,
                key=lambda item: (item.node.label, item.node.display_name, item.relation_type, item.direction),
            )
        )

    def to_seed_cypher(self) -> str:
        """Render MERGE statements to seed the current graph into Neo4j."""

        lines = ["// Seed data for Memento memory graph"]
        for node in self.nodes:
            lines.append(
                f"MERGE (n:{node.label} {{id: {_cypher_value(node.node_id.split(':', 1)[1])}}}) "
                f"SET n += {_cypher_value(node.properties)};"
            )
        for relation in self._relations:
            lines.append(
                "MATCH "
                f"(source {{id: {_cypher_value(relation.source_id.split(':', 1)[1])}}}), "
                f"(target {{id: {_cypher_value(relation.target_id.split(':', 1)[1])}}}) "
                f"MERGE (source)-[r:{relation.relation_type}]->(target) "
                f"SET r += {_cypher_value(relation.properties)};"
            )
        return "\n".join(lines)


def build_memory_graph(snapshot: PatientMemorySnapshot) -> PersonalMemoryGraph:
    """Project a typed patient snapshot into a graph."""

    nodes: list[MemoryNode] = []
    relations: list[MemoryRelation] = []

    patient = snapshot.patient
    patient_node_id = _node_key("Patient", patient.patient_id)
    nodes.append(
        MemoryNode(
            node_id=patient_node_id,
            label="Patient",
            properties={
                "id": patient.patient_id,
                "display_name": patient.display_name,
                "preferred_name": patient.preferred_name,
                "care_notes": list(patient.care_notes),
                "anchors": list(patient.anchors),
            },
        )
    )

    for person in snapshot.people:
        person_node_id = _node_key("Person", person.person_id)
        nodes.append(
            MemoryNode(
                node_id=person_node_id,
                label="Person",
                properties={
                    "id": person.person_id,
                    "name": person.name,
                    "relationship_to_patient": person.relationship_to_patient,
                    "notes": person.notes,
                    "emotional_significance": person.emotional_significance,
                },
            )
        )
        relations.append(
            MemoryRelation(
                source_id=patient_node_id,
                relation_type="KNOWS",
                target_id=person_node_id,
                properties={
                    "relationship_to_patient": person.relationship_to_patient,
                    "emotional_significance": person.emotional_significance,
                },
            )
        )

    for place in snapshot.places:
        nodes.append(
            MemoryNode(
                node_id=_node_key("Place", place.place_id),
                label="Place",
                properties={
                    "id": place.place_id,
                    "name": place.name,
                    "category": place.category,
                    "notes": place.notes,
                },
            )
        )

    for routine in snapshot.routines:
        routine_node_id = _node_key("Routine", routine.routine_id)
        nodes.append(
            MemoryNode(
                node_id=routine_node_id,
                label="Routine",
                properties={
                    "id": routine.routine_id,
                    "title": routine.title,
                    "schedule": routine.schedule,
                    "description": routine.description,
                    "cue": routine.cue,
                    "support_strategy": routine.support_strategy,
                },
            )
        )
        relations.append(
            MemoryRelation(
                source_id=patient_node_id,
                relation_type="FOLLOWS_ROUTINE",
                target_id=routine_node_id,
                properties={},
            )
        )
        if routine.place_id is not None:
            relations.append(
                MemoryRelation(
                    source_id=routine_node_id,
                    relation_type="HAPPENS_AT",
                    target_id=_node_key("Place", routine.place_id),
                    properties={},
                )
            )

    for episode in snapshot.episodes:
        _append_episode(
            episode=episode,
            patient_node_id=patient_node_id,
            nodes=nodes,
            relations=relations,
        )

    return PersonalMemoryGraph(nodes=tuple(nodes), relations=tuple(relations))


def default_memory_schema() -> MemoryGraphSchema:
    """Return the default graph schema used by the project."""

    return MemoryGraphSchema.default()


def _append_episode(
    episode: MemoryEpisode,
    patient_node_id: str,
    nodes: list[MemoryNode],
    relations: list[MemoryRelation],
) -> None:
    episode_node_id = _node_key("Episode", episode.episode_id)
    nodes.append(
        MemoryNode(
            node_id=episode_node_id,
            label="Episode",
            properties={
                "id": episode.episode_id,
                "title": episode.title,
                "narrative": episode.narrative,
                "happened_on": episode.happened_on,
                "tags": list(episode.tags),
            },
        )
    )
    relations.append(
        MemoryRelation(
            source_id=patient_node_id,
            relation_type="REMEMBERS",
            target_id=episode_node_id,
            properties={},
        )
    )
    for person_id in episode.people_ids:
        relations.append(
            MemoryRelation(
                source_id=episode_node_id,
                relation_type="INVOLVES",
                target_id=_node_key("Person", person_id),
                properties={},
            )
        )
    if episode.place_id is not None:
        relations.append(
            MemoryRelation(
                source_id=episode_node_id,
                relation_type="TOOK_PLACE_AT",
                target_id=_node_key("Place", episode.place_id),
                properties={},
            )
        )
    for index, emotion in enumerate(episode.emotions, start=1):
        emotion_node_id = _node_key("Emotion", f"{episode.episode_id}:{index}:{emotion.label.lower()}")
        nodes.append(
            MemoryNode(
                node_id=emotion_node_id,
                label="Emotion",
                properties={
                    "id": f"{episode.episode_id}:{index}:{emotion.label.lower()}",
                    "label": emotion.label,
                    "valence": emotion.valence,
                    "intensity": emotion.intensity,
                    "notes": emotion.notes,
                },
            )
        )
        relations.append(
            MemoryRelation(
                source_id=episode_node_id,
                relation_type="EVOKES",
                target_id=emotion_node_id,
                properties={
                    "valence": emotion.valence,
                    "intensity": emotion.intensity,
                },
            )
        )


def _node_key(label: str, raw_id: str) -> str:
    return f"{label.lower()}:{raw_id}"


def _cypher_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, dict):
        items = ", ".join(
            f"{key}: {_cypher_value(item)}" for key, item in sorted(value.items())
        )
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        items = ", ".join(_cypher_value(item) for item in value)
        return "[" + items + "]"
    raise TypeError(f"Unsupported Cypher value: {type(value)!r}")
