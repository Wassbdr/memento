# Memento Memory

Ce package couvre le chantier `2. Memoire et contexte`.

Il fournit trois briques:

- un schema graphe exportable en Cypher pour Neo4j
- une projection semantique vers des documents exploitables par une couche type LlamaIndex
- une collection vectorielle locale de reference, compatible conceptuellement avec ChromaDB

Le package fournit aujourd'hui un prototype local de reference.
Par defaut, il tourne entierement en local, sans dependances externes obligatoires.
Il expose aussi des adaptateurs optionnels `Neo4jGraphStore` et `ChromaSemanticIndex`
pour brancher de vraies backends via l'extra `memory-backends`.
`LlamaIndexSemanticIndex` peut egalement utiliser la vraie pile LlamaIndex via
`use_llama_index=True` quand les dependances optionnelles sont installees.

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

Notes d'integration:

- les backends externes peuvent etre fermes via `close()`
- `MemorySyncEngine` supporte aussi `with ... as engine:` pour fermer proprement ses backends
