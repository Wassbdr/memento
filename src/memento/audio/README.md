# Memento Audio

Ce dossier regroupe les briques audio reutilisables du projet `memento`.
Le scope actuel couvre deux sous-domaines regroupes dans deux dossiers:

- `stt/` pour la capture micro, le VAD, les structures audio et la transcription
- `tts/` pour la synthese vocale, le playback et les metriques de restitution

Le scope fonctionnel couvre:

- decrire les frames audio et leurs metriques
- capturer du micro en temps reel via un peripherique natif
- detecter les segments de parole avec un VAD energie simple
- transcrire ces segments avec un backend Whisper compatible
- synthese vocale, playback et mesure de latence de restitution

Le package public est expose depuis [`src/memento/audio/__init__.py`](./__init__.py). Les exemples ci-dessous utilisent donc `from memento.audio import ...`.

## Vue d'ensemble

### 1. `stt/models.py`

Contient [`AudioFrame`](./stt/models.py), l'unite de base du package.

- `samples`: tuple de flottants normalises entre `-1.0` et `1.0`
- `sample_rate_hz`: frequence d'echantillonnage
- `channels`: nombre de canaux
- `frame_index`: index logique dans le flux
- `started_at_ms`: timestamp relatif

Metriques derivees:

- `sample_count`
- `duration_ms`
- `peak_level`
- `rms_level`

`AudioFrame` est la structure partagee par la capture, le VAD et la transcription.

### 2. `stt/capture.py`

Definit le contrat minimal pour lire un micro en temps reel.

Objets principaux:

- [`MicrophoneConfig`](./stt/capture.py): validation de la configuration micro
- [`AudioInputDevice`](./stt/capture.py): protocole d'un device avec `open/read/close`
- [`RealTimeMicrophone`](./stt/capture.py): lit des frames de taille fixe
- [`CaptureHealthSnapshot`](./stt/capture.py): metriques frame par frame
- [`CaptureHealthReport`](./stt/capture.py): agregat de session

`RealTimeMicrophone` ne depend pas de `sounddevice`. Il depend uniquement d'un objet qui implemente `AudioInputDevice`, ce qui facilite les tests unitaires.

### 3. `stt/live.py`

Ajoute l'integration avec le vrai micro local.

Objets principaux:

- [`SoundDeviceInput`](./stt/live.py): implementation `AudioInputDevice` basee sur `sounddevice`
- [`list_input_devices`](./stt/live.py): liste les micros disponibles
- [`InputDeviceInfo`](./stt/live.py): metadonnees sur un micro
- [`StreamingSpeechSegmenter`](./stt/live.py): VAD incremental pour un flux temps reel

Ce module sert pour les outils CLI ou les services qui tournent directement sur la machine. Dans le front Streamlit, le micro navigateur passe par `st.audio_input`, ce qui est un chemin different.

### 4. `stt/vad.py`

Fournit un VAD simple base sur le niveau RMS.

Objets principaux:

- [`VoiceActivityConfig`](./stt/vad.py): seuils et fenetres de decision
- [`VoiceActivityDecision`](./stt/vad.py): resultat de classification d'une frame
- [`SpeechSegment`](./stt/vad.py): groupe contigu de frames de parole
- [`EnergyVAD`](./stt/vad.py): classifie une frame ou segmente une sequence

Parametres importants:

- `speech_threshold`: niveau RMS minimal pour considerer de la parole
- `noise_floor`: bruit de fond estime
- `min_speech_frames`: nombre de frames consecutives avant ouverture d'un segment
- `min_silence_frames`: nombre de frames silencieuses avant fermeture
- `pre_roll_frames`: contexte a conserver avant le debut de parole

Le seuil effectif est `max(speech_threshold, noise_floor * 3)`.

### 5. `stt/io.py`

Pont entre bytes WAV et structures du package.

Fonctions:

- [`load_wav_bytes`](./stt/io.py): decode un WAV PCM 16 bits vers `WavAudio`
- [`write_wav_file`](./stt/io.py): ecrit un WAV PCM 16 bits a partir de samples normalises
- [`speech_segment_from_wav_bytes`](./stt/io.py): convertit un WAV entier en un `SpeechSegment`

Limites actuelles:

- uniquement WAV PCM 16 bits en entree
- aucune conversion automatique de format compresse

### 6. `stt/transcription.py`

Definit le pipeline Whisper.

Objets principaux:

- [`WhisperConfig`](./stt/transcription.py): configuration du backend
- [`TranscribedWord`](./stt/transcription.py): mot et timestamps
- [`WhisperBackendResult`](./stt/transcription.py): resultat normalise du backend
- [`WhisperBackend`](./stt/transcription.py): protocole backend
- [`SegmentTranscription`](./stt/transcription.py): resultat final pour un segment
- [`WhisperTranscriber`](./stt/transcription.py): adapte un backend a l'API du projet
- [`WhisperTranscriptionPipeline`](./stt/transcription.py): combine VAD + transcription
- [`OpenAIWhisperBackend`](./stt/transcription.py): implementation avec `openai-whisper`

Helpers utiles:

- `torch_cuda_available()`
- `is_cuda_runtime_error(error)`
- `is_missing_ffmpeg_error(error)`

## Flux de donnees recommandes

### Transcription d'un WAV

1. charger les bytes WAV
2. decoder avec `load_wav_bytes`
3. convertir en frames si un VAD est necessaire
4. segmenter avec `EnergyVAD`
5. transcrire avec `WhisperTranscriber`

Exemple minimal:

```python
from pathlib import Path

from memento.audio import (
    EnergyVAD,
    OpenAIWhisperBackend,
    VoiceActivityConfig,
    WhisperConfig,
    WhisperTranscriber,
    speech_segment_from_wav_bytes,
)

audio_bytes = Path("sample.wav").read_bytes()
segment = speech_segment_from_wav_bytes(audio_bytes)

config = WhisperConfig(
    model_name="large-v3",
    language="fr",
    device="cpu",
    min_segment_duration_ms=0,
)
transcriber = WhisperTranscriber(
    backend=OpenAIWhisperBackend(config=config),
    config=config,
)

result = transcriber.transcribe_segment(segment)
print(result.text if result else "No speech")
```

### Segmentation VAD hors temps reel

```python
from memento.audio import AudioFrame, EnergyVAD, VoiceActivityConfig

frames = (
    AudioFrame(samples=(0.0, 0.0, 0.0, 0.0), sample_rate_hz=100, frame_index=0),
    AudioFrame(samples=(0.08, 0.08, 0.08, 0.08), sample_rate_hz=100, frame_index=1),
    AudioFrame(samples=(0.09, 0.09, 0.09, 0.09), sample_rate_hz=100, frame_index=2),
)

vad = EnergyVAD(
    VoiceActivityConfig(
        speech_threshold=0.05,
        noise_floor=0.01,
        min_speech_frames=2,
        min_silence_frames=2,
        pre_roll_frames=1,
    )
)

segments = vad.segment(frames)
print(len(segments))
```

### Capture micro temps reel

```python
from memento.audio import (
    MicrophoneConfig,
    RealTimeMicrophone,
    SoundDeviceInput,
    StreamingSpeechSegmenter,
)

config = MicrophoneConfig(device_name="default", sample_rate_hz=16000, frame_duration_ms=30)
microphone = RealTimeMicrophone(device=SoundDeviceInput(), config=config)
segmenter = StreamingSpeechSegmenter()

try:
    for frame in microphone.stream(max_frames=50):
        segment = segmenter.push_frame(frame)
        if segment is not None:
            print("Speech from", segment.start_frame_index, "to", segment.end_frame_index)
finally:
    microphone.stop()
```

## Front de test fourni

Le depot contient un front Streamlit de demonstration dans [`experimentations/streamlit_whisper_app.py`](../../../experimentations/streamlit_whisper_app.py).

Il permet de tester:

- import d'un WAV
- capture via le micro du navigateur
- visualisation frame par frame
- segmentation VAD
- transcription Whisper
- diagnostics runtime (`sounddevice`, CUDA)

Lancement:

```bash
uv run streamlit run experimentations/streamlit_whisper_app.py
```

## Tests existants

Les tests couvrent actuellement:

- [`tests/test_capture.py`](../../../tests/test_capture.py)
- [`tests/test_io.py`](../../../tests/test_io.py)
- [`tests/test_live.py`](../../../tests/test_live.py)
- [`tests/test_transcription.py`](../../../tests/test_transcription.py)
- [`tests/test_vad.py`](../../../tests/test_vad.py)

Commande:

```bash
uv run python -m pytest
```

## Dependance et runtime

Le package s'appuie sur:

- `sounddevice` pour le micro natif
- `openai-whisper` pour la transcription
- `torch` pour l'execution CPU ou CUDA
- `streamlit` pour les experimentations UI

Notes pratiques:

- `OpenAIWhisperBackend` fonctionne en direct sur des samples a `16_000 Hz`
- si le sample rate differe, le backend ecrit un WAV temporaire avant transcription
- dans ce cas, `ffmpeg` peut etre requis par `openai-whisper`
- `fp16=True` n'a de sens que sur `cuda`
- les fichiers stereo peuvent exiger un mixage mono cote application avant transcription

### 7. `tts/synthesis.py`

Fournit l'adaptateur texte-vers-voix.

Objets principaux:

- `TextToSpeechConfig`: configuration Voxtral et format de sortie
- `TextToSpeechBackend`: protocole backend
- `TextToSpeechBackendResult`: reponse normalisee du backend
- `SpeechSynthesizer`: facade projet qui mesure la latence de synthese
- `VoxtralTTSBackend`: implementation Mistral/Voxtral avec import paresseux de `mistralai`
- `SynthesizedSpeech`: resultat final exploitable par la couche playback

Le backend attend une cle Mistral via `MISTRAL_API_KEY` par defaut.

### 8. `tts/playback.py`

Fournit la lecture sur haut-parleur.

Objets principaux:

- `DecodedAudio`: audio PCM pret a jouer
- `SpeakerConfig`: configuration de sortie
- `AudioOutputDevice`: protocole device
- `SpeakerPlayer`: facade projet qui decode puis joue une synthese
- `SoundDeviceOutput`: implementation basee sur `sounddevice`
- `PlaybackResult`: metriques de diffusion

Le decodeur supporte actuellement les formats `wav` et `pcm`. Le choix par defaut pour Voxtral est donc `wav`, plus simple a relire localement.

### 9. `tts/voice.py`

Orchestre la reponse vocale de bout en bout.

Objets principaux:

- `VoiceExperienceTargets`: seuils cibles de latence
- `VoiceExperienceMetrics`: indicateurs de synthese et diffusion
- `VoiceResponsePipeline`: pipeline synthese + playback
- `VoiceResponseResult`: agregat final avec verdict sur les seuils

Exemple minimal:

```python
from memento.audio import (
    SoundDeviceOutput,
    SpeakerPlayer,
    SpeechSynthesizer,
    TextToSpeechConfig,
    VoiceResponsePipeline,
    VoxtralTTSBackend,
)

tts_config = TextToSpeechConfig(
    model_name="voxtral-tts-2603",
    response_format="wav",
)

pipeline = VoiceResponsePipeline(
    synthesizer=SpeechSynthesizer(
        backend=VoxtralTTSBackend(config=tts_config),
        config=tts_config,
    ),
    player=SpeakerPlayer(device=SoundDeviceOutput()),
)

result = pipeline.speak("Bonjour, je peux t'aider a retrouver ton manteau.")
print(result.metrics.end_to_end_latency_ms, result.meets_targets)
```

## Limites actuelles

- le backend Voxtral suppose la disponibilite du SDK `mistralai`
- la lecture locale decode uniquement `wav` et `pcm`
- aucun test d'integration reseau n'est execute dans la suite unitaire

- pas de VAD neuronal, uniquement un seuil energie
- pas de resampling interne generaliste
- pas de gestion directe des formats non WAV dans `stt/io.py`
- pas de persistence de session pour les flux live
- pas de post-traitement linguistique de la transcription

## Quand modifier quel module

- tu changes `stt/models.py` si la representation de base d'une frame evolue
- tu changes `stt/capture.py` si la logique de capture ou les metriques de sante changent
- tu changes `stt/live.py` si l'integration micro locale evolue
- tu changes `stt/vad.py` si la segmentation parole/silence change
- tu changes `stt/io.py` si de nouveaux formats ou conversions sont ajoutes
- tu changes `stt/transcription.py` si le backend Whisper ou le pipeline de transcription evoluent
- tu changes `tts/synthesis.py` si l'integration Voxtral ou l'API TTS evoluent
- tu changes `tts/playback.py` si la diffusion haut-parleur ou le decodage audio evoluent
- tu changes `tts/voice.py` si les seuils et metriques de restitution changent
