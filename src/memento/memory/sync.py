"""Backward-compatible sync API surface.

The synchronization logic now lives in smaller modules:
- graph_store.py
- recall.py
- reorientation.py
- sync_engine.py
- sync_types.py
"""

from .graph_store import InMemoryGraphStore
from .sync_engine import MemorySyncEngine
from .sync_types import (
    MemoryIntegrityIssue,
    MemoryContextHit,
    MemoryRecall,
    MemoryScoreBreakdown,
    MemorySyncReport,
    PatientMemoryIntegrityReport,
    PatientReorientationContext,
    RoutineSupportContext,
    TrustedPersonContext,
)

__all__ = [
    "InMemoryGraphStore",
    "MemoryIntegrityIssue",
    "MemoryContextHit",
    "MemoryRecall",
    "MemoryScoreBreakdown",
    "MemorySyncEngine",
    "MemorySyncReport",
    "PatientMemoryIntegrityReport",
    "PatientReorientationContext",
    "RoutineSupportContext",
    "TrustedPersonContext",
]
