# Memento Memory

Ce package couvre le chantier `2. Memoire et contexte`.

Il fournit trois briques:

- un schema graphe exportable en Cypher pour Neo4j
- une projection semantique vers des documents exploitables par une couche type LlamaIndex
- une collection vectorielle locale de reference, compatible conceptuellement avec ChromaDB

## Structure du dossier

Le dossier est organise par responsabilite pour eviter un bloc monolithique:

- `models.py`: modeles metier (patient, proches, routines, souvenirs, emotions)
- `graph.py`: projection knowledge graph + export Cypher
- `semantic.py`: indexation/recherche semantique locale et option LlamaIndex
- `graph_store.py`: stockage graphe en memoire
- `ingestion.py`: validation et reconciliation des conflits d'entree
- `emotion.py`: couche emotionnelle (detecteur injectable)
- `recall.py`: hydration des hits semantiques avec contexte graphe
- `reorientation.py`: extraction de contexte de reassurance patient
- `sync_engine.py`: orchestration sync graphe/index + API publique
- `sync_types.py`: dataclasses de sortie (`MemoryRecall`, `PatientReorientationContext`, ...)
- `sync.py`: facade de compatibilite qui re-exporte l'API sync
- `transaction_log.py`: journal WAL et reprise des transactions inachevees
- `temporal.py`: logique temporelle routines/souvenirs (proximite et recence)
- `weighting.py`: profils de pondération cliniques dynamiques
- `integrity.py`: controles de coherence graphe/index et reparation
- `integrations.py`: adaptateurs optionnels Neo4j et ChromaDB

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

## Fonctionnement de la memoire

Le fonctionnement est organise en deux boucles complementaires.

1. Ecriture (ingestion/synchronisation):
- l'application construit un `PatientMemorySnapshot` (patient, proches, lieux, routines, episodes)
- `reconcile_snapshot(snapshot)` normalise les doublons/alias et signale les conflits detectes
- `MemorySyncEngine.sync_snapshot(snapshot)` projette ce snapshot vers:
	- un graphe de connaissances (`build_memory_graph` + `GraphStore`)
	- un index semantique (`MemoryDocumentProjector` + `SemanticIndex`)
- la synchronisation est transactionnelle au niveau moteur (via transaction explicite graph/index):
	- si l'index semantique echoue, le graphe est restaure
	- chaque tentative est journalisee dans une WAL (write-ahead log)

2. Lecture (rappel/reorientation):
- `MemorySyncEngine.recall(...)` interroge l'index semantique puis hydrate les hits via une passe unique sur les relations graphe
- si le backend expose `batch_recall_context` (Neo4j), la lecture utilise un chargement batch des contextes candidats
- les souvenirs archives sont exclus par defaut (`include_archived=False`)
- un profil de poids clinique dynamique est resolu selon l'heure et l'etat emotionnel
- le ranking final est explicable via `score_breakdown`:
	- score semantique
	- bonus contexte graphe
	- bonus affectif
	- bonus proches de confiance
	- bonus temporalite/recence
	- bonus anchors
	- penalite d'obsolescence (`staleness_penalty`) selon `last_validated_on`
- `MemorySyncEngine.reorientation_context(...)` construit ensuite un contexte patient pret pour la conversation (identite, reperes, proches, routines, souvenirs pertinents)

Cycle recommande:

1. modeliser le patient, ses proches, lieux, routines et souvenirs dans un `PatientMemorySnapshot`
2. appeler `MemorySyncEngine.sync_snapshot(snapshot)`
3. utiliser `MemorySyncEngine.recall(patient_id, query)` pour recuperer des souvenirs semantiques enrichis par le graphe

## Chemin de la donnee

### Chemin ecriture

`PatientMemorySnapshot`
-> `reconcile_snapshot(...)`
-> `build_memory_graph(snapshot)`
-> `GraphStore.replace_graph(...)`
-> `MemoryDocumentProjector.project(snapshot)`
-> `SemanticIndex.replace_documents(...)`

Resultat:
- un etat graphe patient (source de verite relationnelle)
- un etat semantique patient (recherche contextuelle rapide)

### Chemin lecture

`query`
-> `SemanticIndex.search(...)`
-> hydration batch par `PersonalMemoryGraph` (une passe) ou batch Neo4j cible
-> scoring clinique explicable (`score_breakdown`)
-> `MemoryRecall`
-> `PatientReorientationContext`

Resultat:
- hits ordonnes avec justification des signaux utilises
- contexte de reassurance directement exploitable par la couche conversationnelle

Notes d'integration:

- les backends externes peuvent etre fermes via `close()`
- `MemorySyncEngine` supporte aussi `with ... as engine:` pour fermer proprement ses backends

## Reprise apres crash (WAL)

Le moteur supporte un journal de transactions avec deux implementations:

- `InMemoryTransactionLog`: utile en tests/unitaires
- `JsonlTransactionLog`: WAL durable sur disque (reprise apres restart process)

Usage recommande en production:

```python
from memento.memory import JsonlTransactionLog, MemorySyncEngine

engine = MemorySyncEngine(
	transaction_log=JsonlTransactionLog(path="/var/lib/memento/memory-wal.jsonl"),
	auto_recover=True,
)
```

Au demarrage, `auto_recover=True` rejoue les transactions `prepared/graph_written/index_written`
jusqu'au commit, pour remettre graphe et index dans un etat coherent.

## Adaptation emotionnelle temps reel

Le moteur accepte un contexte emotionnel direct (`emotional_state`) ou le detecte via
`RuleBasedEmotionalStateDetector` si aucun contexte n'est fourni.

Ce signal ajuste dynamiquement les poids cliniques:

- agitation/confusion: plus de poids sur routines et anchors
- tristesse: plus de poids sur proches de confiance et affectif
- contexte calme: leger renforcement du signal semantique/recence

Le profil actif et les signaux appliques sont exposes dans:

- `hit.score_breakdown.weight_profile`
- `hit.score_breakdown.weight_signals`

Cette interface est prete a etre branchee sur un detecteur acoustique temps reel.

## Obsolescence et archivage

Les routines et souvenirs supportent:

- `last_validated_on`: date de derniere validation par les aidants
- `archived_on`: date d'archivage sans suppression

Comportement en lecture:

- archives exclus par defaut (`include_archived=False`)
- archives inclus sur demande (`include_archived=True`)
- penalty de stale applique via `staleness_penalty`
- compteur `MemoryRecall.archived_filtered_hits` pour la traçabilite

## Contexte de reorientation (knowledge graph)

La couche memory expose `MemorySyncEngine.reorientation_context(...)` pour produire
un contexte de support cognitif directement ancre dans le graphe patient.

Ce contexte regroupe:

- identite patient (`display_name`, `preferred_name`)
- reperes rassurants (`anchors`) et notes de soin (`care_notes`)
- proches de confiance tries par importance emotionnelle
- routines avec indices pratiques (`cue`, `support_strategy`, lieu)
- rappel semantique enrichi (`MemoryRecall`) pour la question courante

Le ranking des souvenirs est explicable: chaque hit expose un
`score_breakdown` (score semantique, bonus contexte, affectif,
proche de confiance, temporalite, recence et anchors) ainsi que
la liste `signals` qui justifie la priorisation.

Exemple:

```python
from memento.memory import MemorySyncEngine

engine = MemorySyncEngine()
engine.sync_snapshot(snapshot)

context = engine.reorientation_context(
	patient_id="rose",
	query="Je suis perdue, qui vient dimanche ?",
	top_k=3,
)

for person in context.trusted_people:
	print(person.name, person.relationship_to_patient)
```

Cette approche garde le knowledge graph comme source de verite et permet de guider
la reponse conversationnelle avec des signaux utiles a la reassurance.

La recuperation semantique est tolerante aux references orphelines: les hits invalides
sont ignores et exposes via `memory_recall.dropped_hits` pour garder une reponse stable.

## Verification d'integrite

`MemorySyncEngine` expose:

- `integrity_report(patient_id, repair=False)`
- `integrity_report_all(repair=False)`

Ces APIs detectent notamment:

- relations graphe pendantes (noeud source/cible absent)
- documents semantiques orphelins (source_node_id absent du graphe)

Avec `repair=True`, les documents semantiques orphelins sont supprimes
pour retrouver un etat coherent sans interrompre le service.

