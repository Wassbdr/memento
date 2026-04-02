"""Graph store implementations used by the memory synchronization engine."""

from __future__ import annotations

from .graph import PersonalMemoryGraph, build_memory_graph
from .models import PatientMemorySnapshot


class InMemoryGraphStore:
    """Simple patient-scoped graph repository used for synchronization."""

    def __init__(self) -> None:
        self._graphs: dict[str, PersonalMemoryGraph] = {}

    def replace_snapshot(self, snapshot: PatientMemorySnapshot) -> PersonalMemoryGraph:
        graph = build_memory_graph(snapshot)
        return self.replace_graph(snapshot.patient.patient_id, graph)

    def replace_graph(self, patient_id: str, graph: PersonalMemoryGraph) -> PersonalMemoryGraph:
        self._graphs[patient_id] = graph
        return graph

    def delete_patient(self, patient_id: str) -> None:
        self._graphs.pop(patient_id, None)

    def graph_for_patient(self, patient_id: str) -> PersonalMemoryGraph | None:
        return self._graphs.get(patient_id)

    def patient_ids(self) -> tuple[str, ...]:
        """Return known patient identifiers stored in memory."""

        return tuple(sorted(self._graphs.keys()))

    def close(self) -> None:
        """Release resources held by the in-memory graph store."""
