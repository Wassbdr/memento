"""Memory and context building blocks for the Memento knowledge layer."""

from importlib import import_module

from .emotion import EmotionalState, EmotionalStateDetector, RuleBasedEmotionalStateDetector
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
from .ingestion import reconcile_snapshot
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
    InMemoryTransactionLog,
    JsonlTransactionLog,
    MemoryContextHit,
    MemoryIntegrityIssue,
    MemoryIngestionIssue,
    MemoryIngestionReport,
    MemoryRecall,
    MemoryScoreBreakdown,
    MemorySyncEngine,
    MemorySyncReport,
    MemoryTransactionReport,
    PatientMemoryIntegrityReport,
    PatientReorientationContext,
    RoutineSupportContext,
    TrustedPersonContext,
)
from .weighting import ClinicalWeightProfile

__all__ = [
    "AffectiveState",
    "ClinicalWeightProfile",
    "ChromaSemanticIndex",
    "EmotionalState",
    "EmotionalStateDetector",
    "GraphNeighbor",
    "GraphStore",
    "InMemoryChromaCollection",
    "InMemoryGraphStore",
    "InMemoryTransactionLog",
    "JsonlTransactionLog",
    "LlamaIndexSemanticIndex",
    "MemoryContextHit",
    "MemoryIntegrityIssue",
    "MemoryIngestionIssue",
    "MemoryIngestionReport",
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
    "MemoryTransactionReport",
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
    "RuleBasedEmotionalStateDetector",
    "SemanticMemoryIndex",
    "SemanticIndex",
    "SemanticSearchHit",
    "TextEmbedder",
    "TrustedPersonContext",
    "TokenTextEmbedder",
    "build_memory_graph",
    "default_memory_schema",
    "reconcile_snapshot",
]


def __getattr__(name: str):
    if name not in {"ChromaSemanticIndex", "Neo4jGraphStore"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".integrations", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
