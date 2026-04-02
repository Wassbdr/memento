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
         Voxtral TTS
              ↓
           Haut-parleur

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

### Utiliser la restitution vocale

Configurer d'abord une cle API Mistral :

```bash
export MISTRAL_API_KEY="..."
```

Exemple minimal :

```python
from memento import (
    SoundDeviceOutput,
    SpeakerPlayer,
    SpeechSynthesizer,
    TextToSpeechConfig,
    VoiceResponsePipeline,
    VoxtralTTSBackend,
)

tts_config = TextToSpeechConfig(
    model_name="voxtral-tts-2603",
    voice_id="calm-french-voice",
    response_format="wav",
)

pipeline = VoiceResponsePipeline(
    synthesizer=SpeechSynthesizer(
        backend=VoxtralTTSBackend(config=tts_config),
        config=tts_config,
    ),
    player=SpeakerPlayer(device=SoundDeviceOutput()),
)

result = pipeline.speak("Bonjour Charles, je suis la pour t'aider.")
print(result.metrics.end_to_end_latency_ms, result.meets_targets)
```
