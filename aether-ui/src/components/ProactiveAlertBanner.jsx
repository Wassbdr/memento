import { useEffect, useState } from "react";

/**
 * ProactiveAlertBanner — Memento Guardian
 *
 * Banniere d'alerte proactive qui s'affiche quand une personne
 * est reconnue et qu'une tache/rappel est associee.
 */
export default function ProactiveAlertBanner({ alert, onDismiss }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (alert) {
      setIsVisible(true);
    }
  }, [alert]);

  const handleDismiss = () => {
    setIsVisible(false);
    setTimeout(() => onDismiss?.(), 300);
  };

  if (!alert) return null;

  return (
    <div
      className={`fixed top-20 left-1/2 transform -translate-x-1/2 z-50
                  max-w-lg w-[calc(100%-2rem)] rounded-xl shadow-2xl
                  transition-all duration-300 ${
                    isVisible
                      ? "opacity-100 translate-y-0"
                      : "opacity-0 -translate-y-4"
                  }`}
      style={{
        background:
          "linear-gradient(135deg, rgba(139,92,246,0.95), rgba(99,102,241,0.95))",
        border: "2px solid rgba(255,255,255,0.3)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="p-4 flex items-start gap-3">
        {/* Icon pulse */}
        <div className="relative flex-shrink-0">
          <span className="text-2xl">🧠</span>
          <span
            className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-400"
            style={{
              animation: "pulse 1.5s infinite",
            }}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="text-white font-medium text-sm leading-relaxed">
            {alert}
          </p>
          <p className="text-white/70 text-xs mt-1.5">
            Memento Guardian — rappel proactif
          </p>
        </div>

        {/* Close button */}
        <button
          onClick={handleDismiss}
          className="flex-shrink-0 text-white/70 hover:text-white text-xl
                     w-8 h-8 flex items-center justify-center rounded-lg
                     hover:bg-white/10 transition-colors"
        >
          x
        </button>
      </div>

      {/* Progress bar pour auto-dismiss */}
      <div className="h-1 bg-white/20 rounded-b-xl overflow-hidden">
        <div
          className="h-full bg-white/50"
          style={{
            animation: "shrink 10s linear forwards",
          }}
        />
      </div>

      <style>{`
        @keyframes shrink {
          from { width: 100%; }
          to { width: 0%; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
        }
      `}</style>
    </div>
  );
}
