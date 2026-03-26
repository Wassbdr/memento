import { useCallback, useEffect, useRef, useState } from "react";

/**
 * WebcamDetector — Memento Guardian
 *
 * Composant de detection faciale en temps reel.
 * Capture des frames webcam et les envoie au backend pour detection/identification.
 */
export default function WebcamDetector({ onPersonDetected, onAlert, onMemoriesFound }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [lastDetection, setLastDetection] = useState(null);
  const [detectedPersons, setDetectedPersons] = useState([]);
  const [error, setError] = useState(null);

  // Cooldown pour eviter les alertes repetees
  const detectionCooldown = useRef(new Map());

  const startWebcam = useCallback(async () => {
    setError(null);
    setIsLoading(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);
      }
    } catch (err) {
      console.error("Webcam access denied:", err);
      setError("Acces webcam refuse. Verifiez les permissions.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsActive(false);
    setDetectedPersons([]);
    setLastDetection(null);
  }, []);

  const captureAndDetect = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Verifie que la video est prete
    if (video.readyState !== 4) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    // Encode en JPEG base64
    const imageData = canvas.toDataURL("image/jpeg", 0.7);

    try {
      const response = await fetch("/api/v1/vision/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: imageData }),
      });

      const data = await response.json();

      if (data.status === "success") {
        setLastDetection(new Date().toISOString());
        setDetectedPersons(data.persons || []);
        setError(null);

        // Notifie le parent des personnes identifiees
        const identified = (data.persons || []).filter((p) => p.identified);
        if (identified.length > 0) {
          onPersonDetected?.(identified);
        }

        // Notifie des memoires trouvees
        if (data.memory_matches && data.memory_matches.length > 0) {
          onMemoriesFound?.(data.memory_matches);
        }

        // Phrase proactive (LLM) prioritaire sur les simples rappels tâches
        const proactiveText = data.proactive_suggestion || data.proactive_alert;
        if (proactiveText) {
          const alertKey = proactiveText.slice(0, 50);
          const lastAlertTime = detectionCooldown.current.get(alertKey) || 0;
          const now = Date.now();

          // Cooldown de 30 secondes pour eviter le spam
          if (now - lastAlertTime > 30000) {
            detectionCooldown.current.set(alertKey, now);
            onAlert?.(proactiveText);
          }
        }
      } else if (data.proactive_alert) {
        // Message d'erreur du service (ex: face_recognition non installe)
        setError(data.proactive_alert);
      }
    } catch (err) {
      console.error("Detection failed:", err);
      // Ne pas afficher l'erreur pour les timeouts temporaires
    }
  }, [isActive, onPersonDetected, onAlert, onMemoriesFound]);

  // Detection toutes les 2 secondes quand actif
  useEffect(() => {
    if (!isActive) return;

    // Premiere detection apres 1 seconde
    const initialTimeout = setTimeout(captureAndDetect, 1000);

    // Puis toutes les 2 secondes
    const intervalId = setInterval(captureAndDetect, 2000);

    return () => {
      clearTimeout(initialTimeout);
      clearInterval(intervalId);
    };
  }, [isActive, captureAndDetect]);

  // Cleanup a la destruction du composant
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div
      className="webcam-detector rounded-2xl overflow-hidden"
      style={{
        background: "rgba(0,0,0,0.3)",
        border: "1px solid rgba(139,92,246,0.3)",
      }}
    >
      {/* Header */}
      <div className="p-3 flex items-center justify-between border-b border-white/10">
        <h3 className="text-sm font-semibold text-white/90 flex items-center gap-2">
          <span>Détection faciale</span>
          {isActive && (
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          )}
        </h3>
        <button
          onClick={isActive ? stopWebcam : startWebcam}
          disabled={isLoading}
          className={`px-3 py-1 text-xs rounded-lg transition-all ${
            isActive
              ? "bg-red-500/30 text-red-300 hover:bg-red-500/40"
              : "bg-emerald-500/30 text-emerald-300 hover:bg-emerald-500/40"
          } disabled:opacity-50`}
        >
          {isLoading ? "…" : isActive ? "Arrêter" : "Démarrer"}
        </button>
      </div>

      {/* Video container */}
      <div className="relative aspect-video bg-black/50">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Overlay des personnes detectees */}
        {detectedPersons.map((person, idx) => {
          const videoWidth = videoRef.current?.videoWidth || 640;
          const videoHeight = videoRef.current?.videoHeight || 480;

          return (
            <div
              key={idx}
              className="absolute text-xs px-2 py-1 rounded font-medium"
              style={{
                left: `${(person.bounding_box.left / videoWidth) * 100}%`,
                top: `${(person.bounding_box.top / videoHeight) * 100}%`,
                background: person.identified
                  ? "rgba(16,185,129,0.85)"
                  : "rgba(251,191,36,0.85)",
                color: "white",
                transform: "translateY(-100%)",
              }}
            >
              {person.identified
                ? `[IDENTIFIÉ : ${String(person.name ?? "").toUpperCase()}]`
                : person.name}
              {person.identified && ` (${Math.round(person.confidence * 100)}%)`}
            </div>
          );
        })}

        {/* Message quand inactif */}
        {!isActive && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-white/50 text-sm">
            Caméra inactive
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-white/70 text-sm">
            Initialisation…
          </div>
        )}
      </div>

      {/* Footer avec status */}
      <div className="p-2 text-xs text-white/50 flex items-center justify-between">
        <span>
          {error ? (
            <span className="text-amber-400">{error}</span>
          ) : detectedPersons.length > 0 ? (
            <span className="text-emerald-400">
              {detectedPersons.length} visage(s) detecte(s)
            </span>
          ) : isActive ? (
            "Recherche de visages…"
          ) : (
            "Cliquez sur Démarrer"
          )}
        </span>
        {lastDetection && (
          <span>
            {new Date(lastDetection).toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
}
