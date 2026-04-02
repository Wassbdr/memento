from memento.memory import build_memory_graph, default_memory_schema

from memory_fixtures import build_snapshot


def test_default_memory_schema_exports_neo4j_constraints() -> None:
    cypher = default_memory_schema().to_neo4j_cypher()

    assert "CREATE CONSTRAINT patient_patient_id_id_unique" in cypher
    assert "REQUIRE (n.patient_id, n.id) IS UNIQUE;" in cypher
    assert "FOR (n:Episode)" in cypher
    assert "CREATE INDEX routine_title" in cypher


def test_build_memory_graph_contains_nodes_relations_and_seed_cypher() -> None:
    graph = build_memory_graph(build_snapshot())
    seed_cypher = graph.to_seed_cypher()

    assert len(graph.nodes) == 6
    assert len(graph.relations) == 7
    assert graph.get_node("person:claire") is not None
    assert 'MERGE (n:Patient {id: "rose", patient_id: "rose"})' in seed_cypher
    assert 'MERGE (n:Person {id: "claire", patient_id: "rose"})' in seed_cypher
    assert 'SET r += {patient_id: "rose"};' in seed_cypher
    assert "MERGE (source)-[r:REMEMBERS]->(target)" in seed_cypher
