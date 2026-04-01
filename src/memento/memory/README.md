# Memento Memory

Ce package couvre le chantier `2. Memoire et contexte`.

Il fournit trois briques:

- un schema graphe exportable en Cypher pour Neo4j
- une projection semantique vers des documents exploitables par une couche type LlamaIndex
- une collection vectorielle locale de reference, compatible conceptuellement avec ChromaDB

Le package fournit aujourd'hui un prototype local de reference.
Il ne branche pas encore de client Neo4j, ChromaDB ou LlamaIndex en production.

Objets principaux:

- `PatientMemorySnapshot`: etat complet a synchroniser
- `default_memory_schema()`: schema Neo4j initial
- `build_memory_graph(snapshot)`: projection snapshot -> graphe
- `MemoryDocumentProjector`: projection snapshot -> documents semantiques
- `LlamaIndexSemanticIndex`: indexation et recherche semantique
- `MemorySyncEngine`: regles d'ecriture/lecture entre graphe et vecteurs

Cycle recommande:

1. modeliser le patient, ses proches, lieux, routines et souvenirs dans un `PatientMemorySnapshot`
2. appeler `MemorySyncEngine.sync_snapshot(snapshot)`
3. utiliser `MemorySyncEngine.recall(patient_id, query)` pour recuperer des souvenirs semantiques enrichis par le graphe
