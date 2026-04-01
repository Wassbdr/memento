# TODO

## Vision
Construire une prothese cognitive vocale rassurante, sans ecran, capable de capter le contexte du quotidien, de le structurer dans une memoire personnalisee et de repondre naturellement au patient.

## Chantiers

### 1. Capture audio
Stabiliser l'entree micro et l'activation vocale pour obtenir des interactions fiables et non intrusives.

- [x] Integrer le micro temps reel
  Objectif : capturer un flux audio continu exploitable localement.
  Livrable : un module d'acquisition audio avec configuration du device et logs de sante.
- [x] Ajouter la detection de parole (VAD)
  Objectif : ne declencher les traitements que pendant les segments utiles.
  Livrable : une brique VAD testee sur silence, parole et bruit de fond.
- [x] Brancher la transcription Whisper
  Objectif : transformer les tours de parole en texte de maniere robuste.
  Livrable : un pipeline STT documente avec mesure de latence et de qualite.

### 2. Memoire et contexte
Representer la memoire personnelle du patient et relier les evenements, personnes, lieux et emotions.

- [ ] Modeliser le graphe memoire
  Objectif : definir les entites, relations et attributs affectifs du domaine.
  Livrable : un schema initial pour Neo4j avec exemples de noeuds et relations.
- [ ] Mettre en place l'indexation semantique
  Objectif : permettre la recherche contextuelle dans les souvenirs et habitudes.
  Livrable : une couche LlamaIndex connectee a ChromaDB pour l'ingestion et la recherche.
- [ ] Concevoir la strategie de synchronisation
  Objectif : faire coexister recherche vectorielle et graphe de connaissances.
  Livrable : des regles d'ecriture/lecture entre ChromaDB et Neo4j avec tests d'integration.

### 3. Raisonnement conversationnel
Produire des reponses utiles, apaisantes et contextualisees a partir de la memoire du patient.

- [ ] Definir la boucle RAG conversationnelle
  Objectif : assembler recuperation de contexte, prompt et generation de reponse.
  Livrable : un orchestrateur applicatif branche sur Ministral 3 8B.
- [ ] Formuler des garde-fous conversationnels
  Objectif : limiter les hallucinations et adopter un ton rassurant adapte au public cible.
  Livrable : un jeu de prompts systeme et de cas limites documentes.
- [ ] Tracer les decisions du systeme
  Objectif : pouvoir expliquer quelle memoire a ete mobilisee pour chaque reponse.
  Livrable : des logs ou traces de retrieval exploitables pour debug et evaluation.

### 4. Restitution vocale
Transformer les reponses texte en experience vocale fluide, claire et rassurante.

- [ ] Connecter Voxtral TTS
  Objectif : generer une voix naturelle et intelligible.
  Livrable : un adaptateur TTS configurable avec choix de voix et parametres de synthese.
- [ ] Diffuser la reponse sur haut-parleur
  Objectif : assurer une sortie audio stable et a faible latence.
  Livrable : un module de playback avec gestion d'erreurs et interruptions.
- [ ] Mesurer l'experience temps reel
  Objectif : verifier que le cycle complet reste supportable pour un usage quotidien.
  Livrable : des indicateurs de latence bout-en-bout et seuils cibles.

### 5. Produit et confiance
Encadrer les usages, le consentement et la qualite globale du systeme avant d'aller vers des interfaces plus riches.

- [ ] Definir l'apport des aidants et de la famille
  Objectif : structurer la collecte manuelle de souvenirs, routines et profils proches.
  Livrable : un backlog fonctionnel de l'application compagnon et de ses flux de donnees.
- [ ] Cadencer la collecte contextuelle consentie
  Objectif : encadrer les captures automatiques de conversations, visites et emotions.
  Livrable : une politique de consentement, retention et suppression des donnees.
- [ ] Preparer l'extension visuelle future
  Objectif : garder une architecture compatible avec lunettes connectees et reconnaissance faciale.
  Livrable : une note d'architecture listant les points d'extension et les risques.
