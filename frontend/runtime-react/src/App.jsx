import { useEffect, useMemo, useRef, useState } from "react";

const LEGACY_LLM_MODEL_LABEL = "Ministral 3 8B";

const DEFAULT_SNAPSHOT = {
  patient: {
    patient_id: "rose",
    display_name: "Rose Martin",
    preferred_name: "Mamie Rose",
    care_notes: [
      "Rassurer avant de recontextualiser.",
      "Parler lentement avec des phrases courtes.",
    ],
    anchors: [
      "Appartement rue des Lilas",
      "Claire vient souvent le dimanche",
    ],
  },
  people: [
    {
      person_id: "claire",
      name: "Claire Martin",
      relationship_to_patient: "sa fille",
      notes: "Passe le dimanche pour le dejeuner.",
      emotional_significance: 0.98,
    },
    {
      person_id: "lucas",
      name: "Lucas",
      relationship_to_patient: "son petit-fils",
      notes: "Appelle en video le mercredi soir.",
      emotional_significance: 0.86,
    },
  ],
  places: [
    {
      place_id: "home",
      name: "Appartement rue des Lilas",
      category: "domicile",
      notes: "Salon lumineux avec les photos de famille sur la commode.",
    },
  ],
  routines: [
    {
      routine_id: "lunch_sunday",
      title: "Dejeuner du dimanche",
      schedule: "dimanche midi",
      description: "Claire vient partager le repas avec Rose.",
      cue: "Mettre la nappe claire sur la table.",
      support_strategy: "Rappeler que Claire arrive apres la matinee.",
      place_id: "home",
    },
  ],
  episodes: [
    {
      episode_id: "ep_family_lunch",
      title: "Repas de famille",
      narrative: "Rose aime les dejeuners calmes avec Claire dans le salon.",
      happened_on: "2026-03-30",
      people_ids: ["claire"],
      place_id: "home",
      emotions: [
        {
          label: "apaisement",
          valence: 0.9,
          intensity: 0.7,
          notes: "La presence de Claire rassure Rose.",
        },
      ],
      tags: ["famille", "repere", "dimanche"],
    },
  ],
};

const DEFAULT_SETTINGS = {
  apiBaseUrl: "http://127.0.0.1:8000",
  snapshotText: JSON.stringify(DEFAULT_SNAPSHOT, null, 2),
  llmBaseUrl: "http://127.0.0.1:11434/v1",
  llmApiKey: "",
  llmTimeoutSeconds: 60,
  llmModel: "",
  temperature: 0.2,
  topK: 3,
  maxPromptMemories: 3,
  whisperModel: "large-v3",
  whisperLanguage: "fr",
  whisperDevice: "cpu",
  whisperFp16: false,
  ttsEnabled: true,
  ttsModel: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  ttsSpeaker: "Vivian",
  ttsLanguage: "French",
  ttsInstruction: "",
  ttsDeviceMap: "auto",
};

function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [turns, setTurns] = useState([]);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState("Pret a ecouter");
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [health, setHealth] = useState(null);
  const [llmModels, setLlmModels] = useState([]);
  const [llmModelsError, setLlmModelsError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechLevel, setSpeechLevel] = useState(0);

  const recorderRef = useRef(null);
  const orbRef = useRef(null);
  const playbackRef = useRef(null);
  const playbackContextRef = useRef(null);
  const playbackAnalyserRef = useRef(null);
  const playbackDataRef = useRef(null);
  const playbackFrameRef = useRef(0);
  const latestAutoPlayIdRef = useRef("");
  const speechLevelRef = useRef(0);
  const orbMotionFrameRef = useRef(0);

  const parsedSnapshot = useMemo(() => {
    try {
      return JSON.parse(settings.snapshotText);
    } catch {
      return null;
    }
  }, [settings.snapshotText]);

  const patient = health?.patient ?? parsedSnapshot?.patient ?? null;
  const patientName =
    patient?.preferredName ||
    patient?.preferred_name ||
    patient?.displayName ||
    patient?.display_name ||
    "Memento";

  useEffect(() => {
    const controller = new AbortController();

    async function fetchHealth() {
      try {
        const response = await fetch(`${settings.apiBaseUrl}/api/health`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        setHealth(payload);
      } catch {
        setHealth(null);
      }
    }

    fetchHealth();
    return () => controller.abort();
  }, [settings.apiBaseUrl]);

  useEffect(() => {
    const tagsUrl = buildOllamaTagsUrl(settings.llmBaseUrl);
    if (!tagsUrl) {
      setLlmModels([]);
      setLlmModelsError("");
      return undefined;
    }

    const controller = new AbortController();

    async function fetchModels() {
      try {
        const response = await fetch(tagsUrl, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        const models = extractModelNames(payload);
        setLlmModels(models);
        setLlmModelsError("");

        if (models.length > 0) {
          setSettings((current) => {
            const normalizedCurrentModel = current.llmModel.trim();
            if (
              normalizedCurrentModel &&
              normalizedCurrentModel !== LEGACY_LLM_MODEL_LABEL
            ) {
              return current;
            }
            return {
              ...current,
              llmModel: models[0],
            };
          });
        }
      } catch (requestError) {
        if (controller.signal.aborted) {
          return;
        }
        setLlmModels([]);
        setLlmModelsError(requestError.message || "Impossible de charger les modeles LLM.");
      }
    }

    fetchModels();
    return () => controller.abort();
  }, [settings.llmBaseUrl]);

  useEffect(() => {
    return () => {
      stopAssistantPlayback();
    };
  }, []);

  useEffect(() => {
    const latestTurn = turns[0];
    if (!latestTurn?.audioUrl || latestTurn.playbackId === latestAutoPlayIdRef.current) {
      return;
    }
    latestAutoPlayIdRef.current = latestTurn.playbackId;
    void playAssistantAudio(latestTurn.audioUrl);
  }, [turns]);

  useEffect(() => {
    speechLevelRef.current = speechLevel;
  }, [speechLevel]);

  useEffect(() => {
    const orbNode = orbRef.current;
    if (!orbNode) {
      return undefined;
    }

    const mode = isSpeaking
      ? "speaking"
      : isSending
        ? "thinking"
        : isRecording
          ? "recording"
          : "idle";
    const startAt = performance.now();

    const tick = (timestamp) => {
      const seconds = (timestamp - startAt) / 1000;
      const liveSpeechLevel = speechLevelRef.current;
      const motionStrength =
        mode === "speaking" ? 0.94 : mode === "thinking" ? 0.72 : mode === "recording" ? 0.56 : 0.22;
      const speechBoost = mode === "speaking" ? liveSpeechLevel * 0.9 : 0;
      const wave = motionStrength + speechBoost;

      const stageX =
        Math.sin(seconds * 0.54 + 0.35) * (2.4 + wave * 2.2) +
        Math.sin(seconds * 1.28 + 2.1) * (0.8 + wave * 0.7);
      const stageY =
        Math.cos(seconds * 0.46 + 0.2) * (3 + wave * 2.6) +
        Math.sin(seconds * 1.12) * (0.6 + wave * 0.5);
      const stageRotate = Math.sin(seconds * 0.52 + 1.3) * (1.6 + wave * 2.1);
      const driftX =
        Math.sin(seconds * 0.88 + 0.4) * (7 + wave * 7) +
        Math.sin(seconds * 2.24 + 0.8) * (1.8 + wave * 2.2);
      const driftY =
        Math.cos(seconds * 0.74 + 1.1) * (6.5 + wave * 6) +
        Math.sin(seconds * 1.82 + 0.3) * (1.5 + wave * 1.8);
      const shellShiftX = driftX * 0.34;
      const shellShiftY = driftY * 0.34;
      const coreShiftX =
        driftX * -0.18 + Math.sin(seconds * 2.86 + 0.9) * (1 + wave * 1.8);
      const coreShiftY =
        driftY * -0.15 + Math.cos(seconds * 2.32 + 0.4) * (1.2 + wave * 1.7);
      const haloScale = 1 + motionStrength * 0.05 + speechBoost * 0.07;
      const stageScale =
        1 +
        (mode === "speaking" ? 0.018 : mode === "thinking" ? 0.012 : 0) +
        liveSpeechLevel * 0.045;
      const glow = 0.52 + motionStrength * 0.34 + speechBoost * 0.2;
      const spectrumTilt = Math.sin(seconds * 1.18 + 0.5) * (3 + wave * 7.5);

      orbNode.style.setProperty("--orb-stage-x", `${stageX.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-stage-y", `${stageY.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-stage-rotate", `${stageRotate.toFixed(2)}deg`);
      orbNode.style.setProperty("--orb-stage-scale", stageScale.toFixed(3));
      orbNode.style.setProperty("--orb-drift-x", `${driftX.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-drift-y", `${driftY.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-shell-shift-x", `${shellShiftX.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-shell-shift-y", `${shellShiftY.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-core-shift-x", `${coreShiftX.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-core-shift-y", `${coreShiftY.toFixed(2)}px`);
      orbNode.style.setProperty("--orb-halo-scale", haloScale.toFixed(3));
      orbNode.style.setProperty("--orb-glow", glow.toFixed(3));
      orbNode.style.setProperty("--orb-wave", wave.toFixed(3));
      orbNode.style.setProperty("--orb-spectrum-tilt", `${spectrumTilt.toFixed(2)}deg`);

      orbMotionFrameRef.current = requestAnimationFrame(tick);
    };

    orbMotionFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (orbMotionFrameRef.current) {
        cancelAnimationFrame(orbMotionFrameRef.current);
        orbMotionFrameRef.current = 0;
      }
    };
  }, [isRecording, isSending, isSpeaking]);

  function stopAssistantPlayback() {
    if (playbackFrameRef.current) {
      cancelAnimationFrame(playbackFrameRef.current);
      playbackFrameRef.current = 0;
    }

    const audio = playbackRef.current;
    if (audio) {
      audio.onplay = null;
      audio.onpause = null;
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      playbackRef.current = null;
    }

    const context = playbackContextRef.current;
    playbackAnalyserRef.current = null;
    playbackDataRef.current = null;
    playbackContextRef.current = null;
    if (context) {
      void context.close().catch(() => {});
    }

    setIsSpeaking(false);
    setSpeechLevel(0);
  }

  function startSpeechMeter() {
    const analyser = playbackAnalyserRef.current;
    const data = playbackDataRef.current;
    if (!analyser || !data) {
      return;
    }

    const tick = () => {
      analyser.getByteFrequencyData(data);
      const average =
        data.reduce((sum, value) => sum + value, 0) / Math.max(1, data.length);
      const normalized = Math.min(1, average / 160);
      setSpeechLevel((current) => current * 0.58 + normalized * 0.42);
      playbackFrameRef.current = requestAnimationFrame(tick);
    };

    tick();
  }

  async function playAssistantAudio(audioUrl) {
    stopAssistantPlayback();

    try {
      const audio = new Audio(audioUrl);
      audio.preload = "auto";
      playbackRef.current = audio;

      const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;
      if (BrowserAudioContext) {
        const context = new BrowserAudioContext();
        const analyser = context.createAnalyser();
        analyser.fftSize = 128;
        const source = context.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(context.destination);

        playbackContextRef.current = context;
        playbackAnalyserRef.current = analyser;
        playbackDataRef.current = new Uint8Array(analyser.frequencyBinCount);

        if (context.state === "suspended") {
          await context.resume();
        }
      }

      audio.onplay = () => {
        setIsSpeaking(true);
        setStatus("Parole en cours");
        startSpeechMeter();
      };
      audio.onpause = () => {
        if (audio.ended) {
          return;
        }
        if (playbackFrameRef.current) {
          cancelAnimationFrame(playbackFrameRef.current);
          playbackFrameRef.current = 0;
        }
        setIsSpeaking(false);
        setSpeechLevel(0);
      };
      audio.onended = () => {
        stopAssistantPlayback();
        setStatus("Pret a ecouter");
      };
      audio.onerror = () => {
        stopAssistantPlayback();
      };

      await audio.play();
    } catch (playbackError) {
      stopAssistantPlayback();
      setError(playbackError.message || "Lecture audio impossible.");
    }
  }

  async function submitTextTurn() {
    if (!question.trim()) {
      setError("Le message texte est vide.");
      setStatus("En attente d'une question");
      return;
    }
    await sendTurn({ mode: "text", userText: question.trim() });
    setQuestion("");
  }

  async function sendTurn({ mode, userText = "", audioBase64 = "" }) {
    const normalizedModel = settings.llmModel.trim();
    if (!normalizedModel) {
      setError("Choisis un modele LLM exact avant d'envoyer.");
      setStatus("Configuration incomplete");
      return;
    }
    if (llmModels.length > 0 && !llmModels.includes(normalizedModel)) {
      setError(
        `Modele introuvable sur le serveur: ${normalizedModel}. Utilise un identifiant exact comme ${llmModels[0]}.`
      );
      setStatus("Configuration invalide");
      return;
    }

    stopAssistantPlayback();
    setIsSending(true);
    setError("");
    setStatus(mode === "voice" ? "Transcription et reponse en cours" : "Generation en cours");

    try {
      const snapshot = JSON.parse(settings.snapshotText);
      const response = await fetch(`${settings.apiBaseUrl}/api/runtime/turn`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mode,
          userText,
          audioBase64,
          snapshot,
          conversationHistory: history,
          llmBaseUrl: settings.llmBaseUrl,
          llmApiKey: settings.llmApiKey,
          llmTimeoutSeconds: Number(settings.llmTimeoutSeconds),
          llmModel: normalizedModel,
          temperature: Number(settings.temperature),
          topK: Number(settings.topK),
          maxPromptMemories: Number(settings.maxPromptMemories),
          whisperModel: settings.whisperModel,
          whisperLanguage: settings.whisperLanguage,
          whisperDevice: settings.whisperDevice,
          whisperFp16: settings.whisperFp16,
          ttsEnabled: settings.ttsEnabled,
          ttsModel: settings.ttsModel,
          ttsSpeaker: settings.ttsSpeaker,
          ttsLanguage: settings.ttsLanguage,
          ttsInstruction: settings.ttsInstruction,
          ttsDeviceMap: settings.ttsDeviceMap,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      const nextTurn = {
        ...payload.turn,
        playbackId: createPlaybackId(),
        audioUrl:
          payload.turn.audioBase64 && payload.turn.audioMimeType
            ? `data:${payload.turn.audioMimeType};base64,${payload.turn.audioBase64}`
            : null,
      };

      setTurns((current) => [nextTurn, ...current].slice(0, 12));
      setHistory((current) => {
        const nextHistory = [
          ...current,
          { role: "user", content: nextTurn.userText },
          { role: "assistant", content: nextTurn.assistantText },
        ];
        return nextHistory.slice(-12);
      });
      setStatus("Reponse prete");
      setHealth((current) => ({ ...(current || {}), patient: payload.patient }));
    } catch (requestError) {
      setError(requestError.message || "Erreur inconnue.");
      setStatus("Erreur");
    } finally {
      setIsSending(false);
    }
  }

  async function toggleRecording() {
    if (isRecording) {
      await stopRecording();
      return;
    }
    await startRecording();
  }

  async function startRecording() {
    try {
      stopAssistantPlayback();
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new BrowserAudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const gain = audioContext.createGain();
      const chunks = [];

      processor.onaudioprocess = (event) => {
        const channel = event.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(channel));
      };

      source.connect(processor);
      gain.gain.value = 0;
      processor.connect(gain);
      gain.connect(audioContext.destination);

      recorderRef.current = {
        stream,
        audioContext,
        source,
        processor,
        gain,
        chunks,
      };
      setIsRecording(true);
      setStatus("Ecoute en cours");
    } catch (captureError) {
      setError(captureError.message || "Impossible d'acceder au microphone.");
      setStatus("Erreur micro");
    }
  }

  async function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder) {
      setIsRecording(false);
      return;
    }

    recorder.processor.disconnect();
    recorder.source.disconnect();
    recorder.gain.disconnect();
    recorder.stream.getTracks().forEach((track) => track.stop());
    const sampleRate = recorder.audioContext.sampleRate;
    await recorder.audioContext.close();
    recorderRef.current = null;
    setIsRecording(false);

    const samples = mergeChunks(recorder.chunks);
    if (samples.length === 0) {
      setError("Aucun signal audio capture.");
      setStatus("Pret a ecouter");
      return;
    }

    const wavArrayBuffer = encodeWav(samples, sampleRate);
    const audioBase64 = arrayBufferToBase64(wavArrayBuffer);
    await sendTurn({ mode: "voice", audioBase64 });
  }

  function clearConversation() {
    stopAssistantPlayback();
    latestAutoPlayIdRef.current = "";
    setTurns([]);
    setHistory([]);
    setError("");
    setQuestion("");
    setStatus("Pret a ecouter");
  }

  return (
    <div className="app-shell">
      <aside className="config-panel">
        <div className="panel-header">
          <p className="eyebrow">Runtime Controls</p>
          <h2>Configuration</h2>
        </div>

        <label className="field">
          <span>API runtime</span>
          <input
            value={settings.apiBaseUrl}
            onChange={(event) => updateSetting(setSettings, "apiBaseUrl", event.target.value)}
          />
        </label>

        <label className="field field-large">
          <span>Snapshot patient JSON</span>
          <textarea
            rows="16"
            value={settings.snapshotText}
            onChange={(event) => updateSetting(setSettings, "snapshotText", event.target.value)}
          />
        </label>

        <div className="field-grid">
          <label className="field">
            <span>LLM base URL</span>
            <input
              value={settings.llmBaseUrl}
              onChange={(event) => updateSetting(setSettings, "llmBaseUrl", event.target.value)}
            />
          </label>
          <label className="field">
            <span>Modele LLM</span>
            <input
              list="llm-model-options"
              value={settings.llmModel}
              onChange={(event) => updateSetting(setSettings, "llmModel", event.target.value)}
              placeholder={llmModels[0] || "Ex: gemma3:4b"}
            />
            <datalist id="llm-model-options">
              {llmModels.map((modelName) => (
                <option key={modelName} value={modelName} />
              ))}
            </datalist>
            {llmModels.length > 0 ? (
              <small>
                Modeles detectes: {llmModels.join(", ")}
              </small>
            ) : llmModelsError ? (
              <small>Detection Ollama indisponible: {llmModelsError}</small>
            ) : (
              <small>Utilise l'identifiant exact expose par le serveur LLM.</small>
            )}
          </label>
          <label className="field">
            <span>Whisper</span>
            <input
              value={settings.whisperModel}
              onChange={(event) => updateSetting(setSettings, "whisperModel", event.target.value)}
            />
          </label>
          <label className="field">
            <span>TTS speaker</span>
            <input
              value={settings.ttsSpeaker}
              onChange={(event) => updateSetting(setSettings, "ttsSpeaker", event.target.value)}
            />
          </label>
        </div>

        <div className="button-row">
          <button className="secondary-button" type="button" onClick={clearConversation}>
            Effacer
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <section className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">Memento React Runtime</p>
            <h1>Memento</h1>
            <p>
              Clique la sphere pour parler a l&apos;assistant. Le navigateur enregistre en WAV,
              l&apos;API Python transcrit avec Whisper, recentre avec la memoire, puis renvoie une
              reponse et un rendu vocal.
            </p>
          </div>

          <div className="orb-stage">
            <button
              ref={orbRef}
              type="button"
              className={`orb-button ${isRecording ? "is-recording" : ""} ${
                isSending ? "is-thinking" : ""
              } ${isSpeaking ? "is-speaking" : ""}`}
              onClick={toggleRecording}
              disabled={isSending}
              style={{ "--speech-level": speechLevel.toFixed(3) }}
            >
              <span className="orb-ring orb-ring-a" />
              <span className="orb-ring orb-ring-b" />
              <span className="orb-core" />
              <span className="orb-spectrum" />
              <span className="orb-pulse orb-pulse-a" />
              <span className="orb-pulse orb-pulse-b" />
              <span className="orb-content">
                <strong>{patientName}</strong>
                <small>{isRecording ? "Relacher pour envoyer" : status}</small>
              </span>
            </button>
          </div>

          <div className="status-grid">
            <div className="status-card">
              <span>API</span>
              <strong>{health ? "Connectee" : "Hors ligne"}</strong>
            </div>
            <div className="status-card">
              <span>Modele LLM</span>
              <strong>{settings.llmModel || "A choisir"}</strong>
            </div>
            <div className="status-card">
              <span>Mode</span>
              <strong>{isRecording ? "Ecoute" : isSpeaking ? "Parle" : isSending ? "Reflechit" : "Pret"}</strong>
            </div>
          </div>
        </section>

        <section className="composer-card">
          <div className="composer-header">
            <h2>Message texte</h2>
            <p>Fallback rapide si tu ne veux pas utiliser le micro.</p>
          </div>

          <div className="composer-row">
            <input
              className="composer-input"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ex: Qui vient dimanche ?"
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitTextTurn();
                }
              }}
            />
            <button
              className="primary-button"
              type="button"
              onClick={() => {
                void submitTextTurn();
              }}
              disabled={isSending}
            >
              Envoyer
            </button>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="content-grid">
          <div className="conversation-panel">
            <div className="panel-header">
              <p className="eyebrow">Conversation</p>
              <h2>Derniers tours</h2>
            </div>

            {turns.length === 0 ? (
              <div className="empty-state">
                Aucun tour encore. Utilise la sphere ou le champ texte.
              </div>
            ) : (
              turns.map((turn, index) => (
                <article
                  className="turn-card"
                  key={turn.playbackId || `${turn.userText}-${turn.assistantText}-${index}`}
                >
                  <div className="bubble user">
                    <span className="bubble-label">Vous</span>
                    <p>{turn.userText}</p>
                  </div>

                  <div className="bubble assistant">
                    <span className="bubble-label">Assistant</span>
                    <p>{turn.assistantText}</p>
                  </div>

                  <div className="metrics-row">
                    <div>
                      <span>Transcription</span>
                      <strong>{formatMetric(turn.transcriptLatencyMs)}</strong>
                    </div>
                    <div>
                      <span>Generation</span>
                      <strong>{formatMetric(turn.generationLatencyMs)}</strong>
                    </div>
                    <div>
                      <span>Synthese</span>
                      <strong>{formatMetric(turn.synthesisLatencyMs)}</strong>
                    </div>
                  </div>

                  {turn.audioUrl ? <audio controls src={turn.audioUrl} /> : null}
                  {turn.ttsError ? <p className="inline-warning">{turn.ttsError}</p> : null}

                  <div className="memory-zone">
                    <p className="memory-title">Memoire mobilisee</p>
                    <div className="memory-chip-row">
                      {turn.retrievedMemories.length === 0 ? (
                        <span className="memory-chip muted">Aucun souvenir remonte</span>
                      ) : (
                        turn.retrievedMemories.map((memory) => (
                          <span
                            className="memory-chip"
                            key={`${memory.sourceLabel}-${memory.sourceDisplayName}`}
                          >
                            {memory.sourceLabel}: {memory.sourceDisplayName}
                          </span>
                        ))
                      )}
                    </div>
                  </div>
                </article>
              ))
            )}
          </div>

          <div className="patient-panel">
            <div className="panel-header">
              <p className="eyebrow">Patient</p>
              <h2>Contexte actif</h2>
            </div>

            {patient ? (
              <>
                <div className="patient-card">
                  <span>Nom affiche</span>
                  <strong>{patient.displayName || patient.display_name}</strong>
                </div>
                <div className="patient-card">
                  <span>Nom prefere</span>
                  <strong>{patient.preferredName || patient.preferred_name || "n/a"}</strong>
                </div>

                <div className="list-block">
                  <p>Reperes rassurants</p>
                  {(patient.anchors || []).map((anchor) => (
                    <span className="line-chip" key={anchor}>
                      {anchor}
                    </span>
                  ))}
                </div>

                <div className="list-block">
                  <p>Notes de soin</p>
                  {(patient.careNotes || patient.care_notes || []).map((note) => (
                    <span className="line-chip" key={note}>
                      {note}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">Snapshot invalide ou non charge.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function updateSetting(setSettings, key, value) {
  setSettings((current) => ({
    ...current,
    [key]: value,
  }));
}

function buildOllamaTagsUrl(baseUrl) {
  const normalized = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!normalized) {
    return "";
  }
  const withoutChatPath = normalized.replace(/\/chat\/completions$/i, "");
  const withoutV1 = withoutChatPath.replace(/\/v1$/i, "");
  return `${withoutV1}/api/tags`;
}

function extractModelNames(payload) {
  if (!payload || !Array.isArray(payload.models)) {
    return [];
  }
  return payload.models
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }
      return String(item.name || item.model || "").trim();
    })
    .filter(Boolean);
}

function createPlaybackId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatMetric(value) {
  return value == null ? "n/a" : `${Math.round(value)} ms`;
}

function mergeChunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += bytesPerSample;
  }

  return buffer;
}

function writeString(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.byteLength; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return window.btoa(binary);
}

export default App;
