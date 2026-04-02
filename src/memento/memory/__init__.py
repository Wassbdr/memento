"""Memory and context building blocks for the Memento knowledge layer."""

from importlib import import_module

from .graph import (
    GraphNeighbor,
    MemoryGraphSchema,
    MemoryNode,
    MemoryRelation,
    NodeSchema,
    PersonalMemoryGraph,
    RelationSchema,
    build_memory_graph,
    default_memory_schema,
)
from .interfaces import GraphStore, SemanticIndex
from .models import (
    AffectiveState,
    MemoryEpisode,
    PatientMemorySnapshot,
    PatientProfile,
    PersonProfile,
    PlaceProfile,
    RoutineProfile,
)
from .semantic import (
    InMemoryChromaCollection,
    LlamaIndexSemanticIndex,
    MemoryDocument,
    MemoryDocumentProjector,
    SemanticMemoryIndex,
    SemanticSearchHit,
    TextEmbedder,
    TokenTextEmbedder,
)
from .sync import (
    InMemoryGraphStore,
    MemoryContextHit,
    MemoryIntegrityIssue,
    MemoryRecall,
    MemoryScoreBreakdown,
    MemorySyncEngine,
    MemorySyncReport,
    PatientMemoryIntegrityReport,
    PatientReorientationContext,
    RoutineSupportContext,
    TrustedPersonContext,
)

__all__ = [
    "AffectiveState",
    "ChromaSemanticIndex",
    "GraphNeighbor",
    "GraphStore",
    "InMemoryChromaCollection",
    "InMemoryGraphStore",
    "LlamaIndexSemanticIndex",
    "MemoryContextHit",
    "MemoryIntegrityIssue",
    "MemoryDocument",
    "MemoryDocumentProjector",
    "MemoryEpisode",
    "MemoryGraphSchema",
    "MemoryNode",
    "MemoryRecall",
    "MemoryScoreBreakdown",
    "MemoryRelation",
    "MemorySyncEngine",
    "MemorySyncReport",
    "NodeSchema",
    "Neo4jGraphStore",
    "PatientMemorySnapshot",
    "PatientMemoryIntegrityReport",
    "PatientProfile",
    "PersonProfile",
    "PersonalMemoryGraph",
    "PlaceProfile",
    "PatientReorientationContext",
    "RelationSchema",
    "RoutineSupportContext",
    "RoutineProfile",
    "SemanticMemoryIndex",
    "SemanticIndex",
    "SemanticSearchHit",
    "TextEmbedder",
    "TrustedPersonContext",
    "TokenTextEmbedder",
    "build_memory_graph",
    "default_memory_schema",
]


def __getattr__(name: str):
    if name not in {"ChromaSemanticIndex", "Neo4jGraphStore"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".integrations", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
