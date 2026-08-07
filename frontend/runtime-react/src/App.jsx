import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
      notes: "Passe le dimanche pour le déjeuner.",
      emotional_significance: 0.98,
    },
    {
      person_id: "lucas",
      name: "Lucas",
      relationship_to_patient: "son petit-fils",
      notes: "Appelle en vidéo le mercredi soir.",
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
      title: "Déjeuner du dimanche",
      schedule: "dimanche midi",
      description: "Claire vient partager le repas avec Rose.",
      cue: "Mettre la nappe claire sur la table.",
      support_strategy: "Rappeler que Claire arrive après la matinée.",
      place_id: "home",
    },
  ],
  episodes: [
    {
      episode_id: "ep_family_lunch",
      title: "Repas de famille",
      narrative: "Rose aime les déjeuners calmes avec Claire dans le salon.",
      happened_on: "2026-03-30",
      people_ids: ["claire"],
      place_id: "home",
      emotions: [
        {
          label: "apaisement",
          valence: 0.9,
          intensity: 0.7,
          notes: "La présence de Claire rassure Rose.",
        },
      ],
      tags: ["famille", "repère", "dimanche"],
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
  const [showConfig, setShowConfig] = useState(false);
  const [turns, setTurns] = useState([]);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState("Prêt à écouter");
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [health, setHealth] = useState(null);
  const [llmModels, setLlmModels] = useState([]);
  const [llmModelsError, setLlmModelsError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechLevel, setSpeechLevel] = useState(0);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [copiedTurnId, setCopiedTurnId] = useState(null);

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
  const recordingTimerRef = useRef(null);
  const conversationEndRef = useRef(null);
  const isRecordingRef = useRef(false);
  const isSendingRef = useRef(false);

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

  useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);
  useEffect(() => { isSendingRef.current = isSending; }, [isSending]);

  useEffect(() => {
    const controller = new AbortController();
    async function fetchHealth() {
      try {
        const response = await fetch(`${settings.apiBaseUrl}/api/health`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
        const response = await fetch(tagsUrl, { signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const models = extractModelNames(payload);
        setLlmModels(models);
        setLlmModelsError("");
        if (models.length > 0) {
          setSettings((current) => {
            const normalized = current.llmModel.trim();
            if (normalized && normalized !== LEGACY_LLM_MODEL_LABEL) return current;
            return { ...current, llmModel: models[0] };
          });
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        setLlmModels([]);
        setLlmModelsError(err.message || "Impossible de charger les modèles LLM.");
      }
    }
    fetchModels();
    return () => controller.abort();
  }, [settings.llmBaseUrl]);

  useEffect(() => {
    return () => { stopAssistantPlayback(); };
  }, []);

  useEffect(() => {
    const latestTurn = turns[turns.length - 1];
    if (!latestTurn?.audioUrl || latestTurn.playbackId === latestAutoPlayIdRef.current) return;
    latestAutoPlayIdRef.current = latestTurn.playbackId;
    void playAssistantAudio(latestTurn.audioUrl);
  }, [turns]);

  useEffect(() => {
    speechLevelRef.current = speechLevel;
  }, [speechLevel]);

  useEffect(() => {
    const orbNode = orbRef.current;
    if (!orbNode) return undefined;
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

  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0);
      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } else {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      setRecordingSeconds(0);
    }
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    };
  }, [isRecording]);

  useEffect(() => {
    if (turns.length > 0 && conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [turns.length]);

  // Space = push-to-talk (hold = record, release = send)
  useEffect(() => {
    let spaceDown = false;

    function handleKeyDown(event) {
      if (event.code !== "Space" || event.repeat) return;
      if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
      if (isSendingRef.current) return;
      event.preventDefault();
      if (!isRecordingRef.current && !spaceDown) {
        spaceDown = true;
        void startRecording();
      }
    }

    function handleKeyUp(event) {
      if (event.code !== "Space") return;
      if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
      if (!spaceDown) return;
      event.preventDefault();
      spaceDown = false;
      if (isRecordingRef.current) {
        void stopRecording();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

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
    if (context) void context.close().catch(() => {});
    setIsSpeaking(false);
    setSpeechLevel(0);
  }

  function startSpeechMeter() {
    const analyser = playbackAnalyserRef.current;
    const data = playbackDataRef.current;
    if (!analyser || !data) return;
    const tick = () => {
      analyser.getByteFrequencyData(data);
      const average = data.reduce((sum, v) => sum + v, 0) / Math.max(1, data.length);
      const normalized = Math.min(1, average / 160);
      setSpeechLevel((c) => c * 0.58 + normalized * 0.42);
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
        if (context.state === "suspended") await context.resume();
      }
      audio.onplay = () => {
        setIsSpeaking(true);
        setStatus("Parole en cours…");
        startSpeechMeter();
      };
      audio.onpause = () => {
        if (audio.ended) return;
        if (playbackFrameRef.current) {
          cancelAnimationFrame(playbackFrameRef.current);
          playbackFrameRef.current = 0;
        }
        setIsSpeaking(false);
        setSpeechLevel(0);
      };
      audio.onended = () => { stopAssistantPlayback(); setStatus("Prêt à écouter"); };
      audio.onerror = () => { stopAssistantPlayback(); };
      await audio.play();
    } catch (err) {
      stopAssistantPlayback();
      setError(err.message || "Lecture audio impossible.");
    }
  }

  async function submitTextTurn() {
    if (!question.trim()) {
      setError("Le message texte est vide.");
      return;
    }
    await sendTurn({ mode: "text", userText: question.trim() });
    setQuestion("");
  }

  async function sendTurn({ mode, userText = "", audioBase64 = "" }) {
    const normalizedModel = settings.llmModel.trim();
    if (!normalizedModel) {
      setError("Choisissez un modèle LLM avant d'envoyer.");
      setStatus("Configuration incomplète");
      return;
    }
    if (llmModels.length > 0 && !llmModels.includes(normalizedModel)) {
      setError(
        `Modèle introuvable sur le serveur : ${normalizedModel}. Disponibles : ${llmModels.join(", ")}.`
      );
      setStatus("Configuration invalide");
      return;
    }
    stopAssistantPlayback();
    setIsSending(true);
    setError("");
    setStatus(mode === "voice" ? "Transcription en cours…" : "Génération en cours…");
    try {
      const snapshot = JSON.parse(settings.snapshotText);
      const response = await fetch(`${settings.apiBaseUrl}/api/runtime/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode, userText, audioBase64, snapshot,
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
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      const nextTurn = {
        ...payload.turn,
        playbackId: createPlaybackId(),
        audioUrl:
          payload.turn.audioBase64 && payload.turn.audioMimeType
            ? `data:${payload.turn.audioMimeType};base64,${payload.turn.audioBase64}`
            : null,
      };
      setTurns((current) => [...current, nextTurn].slice(-12));
      setHistory((current) => {
        const nextHistory = [
          ...current,
          { role: "user", content: nextTurn.userText },
          { role: "assistant", content: nextTurn.assistantText },
        ];
        return nextHistory.slice(-12);
      });
      setStatus("Réponse prête");
      setHealth((current) => ({ ...(current || {}), patient: payload.patient }));
    } catch (err) {
      setError(err.message || "Erreur inconnue.");
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
      recorderRef.current = { stream, audioContext, source, processor, gain, chunks };
      setIsRecording(true);
      setStatus("Écoute en cours…");
    } catch (err) {
      setError(err.message || "Impossible d'accéder au microphone.");
      setStatus("Erreur micro");
    }
  }

  async function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder) { setIsRecording(false); return; }
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
      setError("Aucun signal audio capturé.");
      setStatus("Prêt à écouter");
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
    setStatus("Prêt à écouter");
  }

  async function copyToClipboard(text, turnId) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedTurnId(turnId);
      setTimeout(() => setCopiedTurnId(null), 2000);
    } catch {
      // clipboard unavailable
    }
  }

  const apiConnected = Boolean(health?.status === "ok");

  const orbLabel = isRecording
    ? formatDuration(recordingSeconds)
    : isSending
      ? "…"
      : isSpeaking
        ? "♪"
        : "●";

  const orbSub = isRecording
    ? "Relâcher pour envoyer"
    : status;

  return (
    <div className="app-shell">
      <button
        className="config-toggle"
        type="button"
        onClick={() => setShowConfig((v) => !v)}
        title={showConfig ? "Fermer la configuration" : "Ouvrir la configuration"}
        aria-label="Paramètres"
      >
        {showConfig ? "✕" : "⚙"}
      </button>

      <aside className={`config-panel ${showConfig ? "is-open" : ""}`}>
        <div className="panel-header">
          <p className="eyebrow">Paramètres</p>
          <h2>Configuration</h2>
        </div>

        <label className="field">
          <span>URL de l'API runtime</span>
          <input
            value={settings.apiBaseUrl}
            onChange={(e) => updateSetting(setSettings, "apiBaseUrl", e.target.value)}
          />
          <small style={{ color: apiConnected ? "var(--cyan)" : "var(--pink)" }}>
            {apiConnected ? "API connectée" : "API hors ligne"}
          </small>
        </label>

        <label className="field field-large">
          <span>Snapshot patient (JSON)</span>
          <textarea
            rows="14"
            value={settings.snapshotText}
            onChange={(e) => updateSetting(setSettings, "snapshotText", e.target.value)}
          />
        </label>

        <div className="field-grid">
          <label className="field">
            <span>URL du serveur LLM</span>
            <input
              value={settings.llmBaseUrl}
              onChange={(e) => updateSetting(setSettings, "llmBaseUrl", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Modèle LLM</span>
            <input
              list="llm-model-options"
              value={settings.llmModel}
              onChange={(e) => updateSetting(setSettings, "llmModel", e.target.value)}
              placeholder={llmModels[0] || "Ex : gemma3:4b"}
            />
            <datalist id="llm-model-options">
              {llmModels.map((m) => <option key={m} value={m} />)}
            </datalist>
            {llmModels.length > 0 ? (
              <small>Disponibles : {llmModels.join(", ")}</small>
            ) : llmModelsError ? (
              <small>Ollama hors ligne : {llmModelsError}</small>
            ) : (
              <small>Entrez l'identifiant exact exposé par le serveur LLM.</small>
            )}
          </label>
          <label className="field">
            <span>Modèle Whisper</span>
            <input
              value={settings.whisperModel}
              onChange={(e) => updateSetting(setSettings, "whisperModel", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Voix TTS</span>
            <input
              value={settings.ttsSpeaker}
              onChange={(e) => updateSetting(setSettings, "ttsSpeaker", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Clé API LLM (optionnel)</span>
            <input
              type="password"
              value={settings.llmApiKey}
              onChange={(e) => updateSetting(setSettings, "llmApiKey", e.target.value)}
            />
          </label>
          <label className="field">
            <span>Température LLM</span>
            <input
              type="number"
              min="0"
              max="2"
              step="0.05"
              value={settings.temperature}
              onChange={(e) => updateSetting(setSettings, "temperature", e.target.value)}
            />
          </label>
        </div>

        <div className="button-row">
          <button className="secondary-button full-width" type="button" onClick={clearConversation}>
            Effacer la conversation
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <section className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">Memento · Prothèse cognitive vocale</p>
            <h1>{patientName}</h1>
            <p>
              Maintenez <kbd>Espace</kbd> pour parler, relâchez pour envoyer —
              ou cliquez la sphère.
            </p>
          </div>

          <div className="orb-stage">
            <button
              ref={orbRef}
              type="button"
              className={`orb-button${isRecording ? " is-recording" : ""}${isSending ? " is-thinking" : ""}${isSpeaking ? " is-speaking" : ""}`}
              onClick={toggleRecording}
              disabled={isSending}
              style={{ "--speech-level": speechLevel.toFixed(3) }}
              aria-label={isRecording ? "Arrêter l'enregistrement" : "Démarrer l'enregistrement"}
            >
              <span className="orb-ring orb-ring-a" />
              <span className="orb-ring orb-ring-b" />
              <span className="orb-core" />
              <span className="orb-spectrum" />
              <span className="orb-pulse orb-pulse-a" />
              <span className="orb-pulse orb-pulse-b" />
              <span className="orb-content">
                <strong className="orb-label">{orbLabel}</strong>
                <small>{orbSub}</small>
              </span>
            </button>
          </div>

          <div className="status-grid">
            <div className={`status-card${apiConnected ? " status-ok" : " status-offline"}`}>
              <span>API</span>
              <strong>{apiConnected ? "Connectée" : "Hors ligne"}</strong>
            </div>
            <div className="status-card">
              <span>Modèle</span>
              <strong className="status-model">{settings.llmModel || "—"}</strong>
            </div>
            <div className="status-card">
              <span>Mode</span>
              <strong>
                {isRecording ? "Écoute" : isSpeaking ? "Parole" : isSending ? "Réflexion" : "Prêt"}
              </strong>
            </div>
          </div>
        </section>

        <section className="composer-card">
          <div className="composer-header">
            <h2>Saisie manuelle</h2>
            <p>Alternative au micro — posez une question par écrit.</p>
          </div>
          <div className="composer-row">
            <input
              className="composer-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ex : Qui vient dimanche ?"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submitTextTurn();
                }
              }}
            />
            <button
              className="primary-button"
              type="button"
              onClick={() => void submitTextTurn()}
              disabled={isSending}
            >
              {isSending ? "…" : "Envoyer"}
            </button>
          </div>
          {error ? <div className="error-banner" role="alert">{error}</div> : null}
        </section>

        <section className="content-grid">
          <div className="conversation-panel">
            <div className="panel-header">
              <p className="eyebrow">Conversation</p>
              <h2>Échanges récents</h2>
              {turns.length > 0 && (
                <button className="clear-link" type="button" onClick={clearConversation}>
                  Tout effacer
                </button>
              )}
            </div>

            {isSending && (
              <div className="thinking-indicator" aria-live="polite">
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-label">{status}</span>
              </div>
            )}

            {turns.length === 0 && !isSending ? (
              <div className="empty-state">
                Aucun échange pour l'instant. Utilisez la sphère ou la saisie manuelle.
              </div>
            ) : (
              turns.map((turn, index) => (
                <article
                  className="turn-card"
                  key={turn.playbackId || `turn-${index}`}
                >
                  <div className="bubble user">
                    <span className="bubble-label">
                      {turn.source === "voice" ? "🎙 Vous" : "✎ Vous"}
                    </span>
                    <p>{turn.userText}</p>
                  </div>

                  <div className="bubble assistant">
                    <div className="bubble-label-row">
                      <span className="bubble-label">Memento</span>
                      <button
                        className={`copy-button${copiedTurnId === turn.playbackId ? " copied" : ""}`}
                        type="button"
                        onClick={() => void copyToClipboard(turn.assistantText, turn.playbackId)}
                        title="Copier la réponse"
                      >
                        {copiedTurnId === turn.playbackId ? "Copié ✓" : "Copier"}
                      </button>
                    </div>
                    <p>{turn.assistantText}</p>
                    {turn.audioUrl ? (
                      <audio controls src={turn.audioUrl} className="turn-audio" />
                    ) : null}
                  </div>

                  {turn.guardApplied && (
                    <div className="guard-notice" role="note">
                      ⚠ Contexte insuffisant — réponse de sécurité appliquée.
                    </div>
                  )}

                  {turn.ttsError ? (
                    <p className="inline-warning">{turn.ttsError}</p>
                  ) : null}

                  <details className="turn-details">
                    <summary>Métriques &amp; mémoire mobilisée</summary>
                    <div className="metrics-row">
                      <div>
                        <span>Transcription</span>
                        <strong>{formatMetric(turn.transcriptLatencyMs)}</strong>
                      </div>
                      <div>
                        <span>Génération</span>
                        <strong>{formatMetric(turn.generationLatencyMs)}</strong>
                      </div>
                      <div>
                        <span>Synthèse</span>
                        <strong>{formatMetric(turn.synthesisLatencyMs)}</strong>
                      </div>
                    </div>
                    <div className="memory-zone">
                      <p className="memory-title">Mémoire mobilisée</p>
                      <div className="memory-chip-row">
                        {turn.retrievedMemories.length === 0 ? (
                          <span className="memory-chip muted">Aucun souvenir remonté</span>
                        ) : (
                          turn.retrievedMemories.map((memory) => (
                            <span
                              className="memory-chip"
                              key={`${memory.sourceLabel}-${memory.sourceDisplayName}`}
                              title={memory.summary}
                            >
                              {memory.sourceLabel} · {memory.sourceDisplayName}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  </details>
                </article>
              ))
            )}
            <div ref={conversationEndRef} />
          </div>

          <div className="patient-panel">
            <div className="panel-header">
              <p className="eyebrow">Patient</p>
              <h2>Contexte actif</h2>
            </div>

            {patient ? (
              <>
                <div className="patient-identity">
                  <div className="patient-avatar">
                    {(
                      patient.preferredName ||
                      patient.preferred_name ||
                      patient.displayName ||
                      patient.display_name ||
                      "?"
                    )[0].toUpperCase()}
                  </div>
                  <div className="patient-identity-text">
                    <div className="patient-name">
                      {patient.displayName || patient.display_name}
                    </div>
                    <div className="patient-alias">
                      {patient.preferredName || patient.preferred_name || "—"}
                    </div>
                  </div>
                </div>

                <div className="list-block">
                  <p>Repères rassurants</p>
                  {(patient.anchors || []).length > 0 ? (
                    (patient.anchors || []).map((anchor) => (
                      <span className="line-chip" key={anchor}>{anchor}</span>
                    ))
                  ) : (
                    <span className="memory-chip muted">Aucun repère défini</span>
                  )}
                </div>

                <div className="list-block">
                  <p>Notes de soin</p>
                  {(patient.careNotes || patient.care_notes || []).length > 0 ? (
                    (patient.careNotes || patient.care_notes || []).map((note) => (
                      <span className="line-chip care-note" key={note}>{note}</span>
                    ))
                  ) : (
                    <span className="memory-chip muted">Aucune note de soin</span>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">Snapshot invalide ou non chargé.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function updateSetting(setSettings, key, value) {
  setSettings((current) => ({ ...current, [key]: value }));
}

function buildOllamaTagsUrl(baseUrl) {
  const normalized = String(baseUrl || "").trim().replace(/\/+$/, "");
  if (!normalized) return "";
  const withoutChatPath = normalized.replace(/\/chat\/completions$/i, "");
  const withoutV1 = withoutChatPath.replace(/\/v1$/i, "");
  return `${withoutV1}/api/tags`;
}

function extractModelNames(payload) {
  if (!payload || !Array.isArray(payload.models)) return [];
  return payload.models
    .map((item) => {
      if (!item || typeof item !== "object") return "";
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

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
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
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += bytesPerSample;
  }
  return buffer;
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export default App;
