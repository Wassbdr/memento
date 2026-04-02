"""Integrity checks and optional repair routines for patient memory state."""

from __future__ import annotations

from .graph import PersonalMemoryGraph
from .semantic import MemoryDocument
from .sync_types import MemoryIntegrityIssue, PatientMemoryIntegrityReport


def build_patient_integrity_report(
    *,
    patient_id: str,
    graph: PersonalMemoryGraph,
    semantic_documents: tuple[MemoryDocument, ...],
    repaired_documents: int = 0,
) -> PatientMemoryIntegrityReport:
    """Build an integrity report by checking graph and semantic consistency."""

    node_ids = {node.node_id for node in graph.nodes}

    issues: list[MemoryIntegrityIssue] = []
    dangling_relations = 0
    for relation in graph.relations:
        missing_parts = []
        if relation.source_id not in node_ids:
            missing_parts.append("missing source")
        if relation.target_id not in node_ids:
            missing_parts.append("missing target")
        if not missing_parts:
            continue

        dangling_relations += 1
        issues.append(
            MemoryIntegrityIssue(
                issue_type="dangling-graph-relation",
                entity_id=f"{relation.source_id}->{relation.relation_type}->{relation.target_id}",
                detail=", ".join(missing_parts),
            )
        )

    orphan_documents = 0
    for document in semantic_documents:
        if document.source_node_id in node_ids:
            continue
        orphan_documents += 1
        issues.append(
            MemoryIntegrityIssue(
                issue_type="orphan-semantic-document",
                entity_id=document.document_id,
                detail=f"source_node_id {document.source_node_id!r} not found in graph",
            )
        )

    return PatientMemoryIntegrityReport(
        patient_id=patient_id,
        graph_node_count=len(graph.nodes),
        graph_relation_count=len(graph.relations),
        semantic_document_count=len(semantic_documents),
        dangling_relations=dangling_relations,
        orphan_documents=orphan_documents,
        repaired_documents=repaired_documents,
        issues=tuple(issues),
    )


def orphan_document_ids(report: PatientMemoryIntegrityReport) -> tuple[str, ...]:
    """Return the semantic document IDs marked as orphan in one report."""

    return tuple(
        issue.entity_id
        for issue in report.issues
        if issue.issue_type == "orphan-semantic-document"
    )
