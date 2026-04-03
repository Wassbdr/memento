import { useEffect, useMemo, useRef, useState } from "react";

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
  llmModel: "Ministral 3 8B",
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
  const [isRecording, setIsRecording] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const recorderRef = useRef(null);

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
          llmModel: settings.llmModel,
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
              value={settings.llmModel}
              onChange={(event) => updateSetting(setSettings, "llmModel", event.target.value)}
            />
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
            <h1>Une sphere centrale, sobre, et vivante.</h1>
            <p>
              Clique la sphere pour parler a l&apos;assistant. Le navigateur enregistre en WAV,
              l&apos;API Python transcrit avec Whisper, recentre avec la memoire, puis renvoie une
              reponse et un rendu vocal.
            </p>
          </div>

          <div className="orb-stage">
            <button
              type="button"
              className={`orb-button ${isRecording ? "is-recording" : ""} ${
                isSending ? "is-busy" : ""
              }`}
              onClick={toggleRecording}
              disabled={isSending}
            >
              <span className="orb-ring orb-ring-a" />
              <span className="orb-ring orb-ring-b" />
              <span className="orb-core" />
              <span className="orb-spectrum" />
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
              <strong>{settings.llmModel}</strong>
            </div>
            <div className="status-card">
              <span>Mode</span>
              <strong>{isRecording ? "Ecoute" : "Pret"}</strong>
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
                <article className="turn-card" key={`${turn.userText}-${index}`}>
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
