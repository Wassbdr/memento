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
    MemoryIngestionIssue,
    MemoryIngestionReport,
    MemoryContextHit,
    MemoryRecall,
    MemoryScoreBreakdown,
    MemorySyncReport,
    MemoryTransactionReport,
    PatientMemoryIntegrityReport,
    PatientReorientationContext,
    RoutineSupportContext,
    TrustedPersonContext,
)
from .transaction_log import InMemoryTransactionLog, JsonlTransactionLog

__all__ = [
    "InMemoryGraphStore",
    "InMemoryTransactionLog",
    "JsonlTransactionLog",
    "MemoryIntegrityIssue",
    "MemoryIngestionIssue",
    "MemoryIngestionReport",
    "MemoryContextHit",
    "MemoryRecall",
    "MemoryScoreBreakdown",
    "MemorySyncEngine",
    "MemorySyncReport",
    "MemoryTransactionReport",
    "PatientMemoryIntegrityReport",
    "PatientReorientationContext",
    "RoutineSupportContext",
    "TrustedPersonContext",
]
