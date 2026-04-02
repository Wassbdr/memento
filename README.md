# Memento

Memento est une prothèse cognitive vocale destinée aux patients atteints de la maladie d’Alzheimer. L’idée centrale est simple : donner au patient un accès permanent, à portée de voix, à sa propre mémoire personnelle. Un compagnon vocal qui connaît ses proches, ses habitudes, ses souvenirs et ses routines, et peut y répondre à tout moment de manière naturelle et rassurante — sans écran, sans bouton, sans aucune interaction cognitive requise.
La mémoire du patient est alimentée de deux façons : par les aidants et la famille via une application dédiée, et de manière automatique en captant le contexte des interactions quotidiennes — visites, conversations, émotions — avec le consentement de chacun. Ce contexte vivant est stocké dans un graphe de mémoire personnalisé et affectif, bien au-delà de ce que peut offrir n’importe quel assistant généraliste.
L’objectif n’est pas de guérir, mais de réduire les moments de confusion et d’angoisse, de préserver l’autonomie le plus longtemps possible, et de soulager les aidants. Dans un second temps, une couche visuelle — idéalement via des lunettes connectées — permettra à l’assistant de reconnaître les visages en temps réel et d’enrichir encore davantage le contexte de chaque interaction.

# TODO:
Micro → Whisper + VAD
              ↓
        LlamaIndex
              ↓
   ChromaDB ←→ Neo4j
              ↓
    Ministral 3 8B
              ↓
         Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
              ↓
           Haut-parleur

## Fonctionnement de la memoire

La memoire suit une logique knowledge graph + recherche semantique.

1. Ingestion:
- l'application compose un `PatientMemorySnapshot`
- le snapshot est synchronise via `MemorySyncEngine.sync_snapshot(...)`
- le moteur met a jour en parallele:
    - le graphe patient (relations explicites)
    - l'index semantique (recuperation contextuelle)

2. Recuperation:
- `MemorySyncEngine.recall(...)` retrouve des souvenirs pertinents
- les hits sont reclasses avec un score clinique explicable (`score_breakdown`)
- `MemorySyncEngine.reorientation_context(...)` produit un contexte de reassurance pret a etre injecte dans la reponse vocale

Voir la documentation detaillee de la couche memory:
- `src/memento/memory/README.md`

## Chemin de la donnee

Chemin ecriture:

`PatientMemorySnapshot`
-> projection graphe
-> ecriture `GraphStore`
-> projection documents
-> ecriture `SemanticIndex`

Chemin lecture:

`question patient`
-> recherche semantique
-> enrichment par le graphe
-> scoring explicable
-> contexte de reorientation
-> generation de reponse vocale

## Comment tester

### Installer l'environnement

```bash
uv sync
```

### Lancer les tests unitaires

```bash
uv run python -m pytest
```

### Tester la transcription micro avec Streamlit

```bash
uv run streamlit run experimentations/streamlit_whisper_app.py
```

Ensuite :
- autoriser l'acces au microphone dans le navigateur
- enregistrer une phrase avec le bouton de capture
- cliquer sur `Transcribe`
- lire la transcription mot par mot puis la transcription complete

Notes :
- le depot est maintenant configure par defaut pour `Whisper large-v3`
- au premier lancement, `openai-whisper` peut telecharger le modele choisi
- `large-v3` sur `cpu` fonctionne, mais le chargement initial et l'inference peuvent etre lents
- la capture micro temps reel utilise `sounddevice`
- `ffmpeg` reste necessaire pour les transcriptions depuis un fichier audio, mais le flux micro capture en memoire n'en depend pas
- le mode `cuda` avec `fp16` est recommande si une GPU compatible est disponible
- les modeles `tiny`, `base` ou `small` restent utiles pour des essais rapides
- le projet pinne maintenant `torch` sur l'index PyTorch `cu128` pour eviter qu'un `uv sync` reinstalle un build CPU-only

### Tester le TTS avec Streamlit

Installer d'abord le runtime local recommande pour `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` :

```bash
uv pip install -U qwen-tts
```

Ensuite lancer le labo TTS :

```bash
uv run streamlit run experimentations/streamlit_tts_app.py
```

Notes :
- la doc officielle Qwen recommande le package Python `qwen-tts`
- le modele `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` se charge localement via `Qwen3TTSModel.from_pretrained(...)`
- `Synthese seulement` genere un preview audio dans le navigateur
- `Synthese + server speaker` joue sur les haut-parleurs de la machine qui execute Streamlit
- le front expose `speaker`, `langue` et `instruction` pour le mode `CustomVoice`
- le chemin `wav` reste le plus simple pour les essais locaux

### Utiliser la restitution vocale

Verifier d'abord que `qwen-tts` et ses dependances runtime sont installes.

Exemple minimal :

```python
from memento import (
    DEFAULT_QWEN_TTS_LANGUAGE,
    DEFAULT_QWEN_TTS_MODEL_NAME,
    DEFAULT_QWEN_TTS_SPEAKER,
    QwenTTSBackend,
    SoundDeviceOutput,
    SpeakerPlayer,
    SpeechSynthesizer,
    TextToSpeechConfig,
    VoiceResponsePipeline,
)

tts_config = TextToSpeechConfig(
    model_name=DEFAULT_QWEN_TTS_MODEL_NAME,
    voice_id=DEFAULT_QWEN_TTS_SPEAKER,
    language=DEFAULT_QWEN_TTS_LANGUAGE,
    response_format="wav",
    instruction="Parle calmement et rassure la personne.",
)

pipeline = VoiceResponsePipeline(
    synthesizer=SpeechSynthesizer(
        backend=QwenTTSBackend(config=tts_config),
        config=tts_config,
    ),
    player=SpeakerPlayer(device=SoundDeviceOutput()),
)

result = pipeline.speak("Bonjour Charles, je suis la pour t'aider.")
print(result.metrics.end_to_end_latency_ms, result.meets_targets)
```
