import { BellRing, X } from "lucide-react";
import { useEffect } from "react";

export default function EventAlertStrip({ alertEvent, onClose }) {
  useEffect(() => {
    if (!alertEvent) return undefined;
    const t = setTimeout(onClose, 6000);
    return () => clearTimeout(t);
  }, [alertEvent, onClose]);

  if (!alertEvent) return null;

  return (
    <div
      className="relative shrink-0 overflow-hidden animate-slide-down"
      style={{
        background:  "linear-gradient(90deg,rgba(13,148,136,0.12) 0%,rgba(6,182,212,0.08) 50%,rgba(99,102,241,0.09) 100%)",
        borderBottom:"1px solid rgba(20,184,166,0.20)",
      }}
    >
      {/* Animated top shine */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{
          background: "linear-gradient(90deg,transparent 0%,rgba(45,212,191,0.55) 50%,transparent 100%)",
        }}
      />

      <div className="flex items-center gap-3 px-6 py-2.5">
        {/* Icon badge */}
        <div
          className="flex items-center justify-center w-6 h-6 rounded-full shrink-0"
          style={{
            background: "rgba(20,184,166,0.18)",
            border:     "1px solid rgba(20,184,166,0.35)",
          }}
        >
          <BellRing className="w-3 h-3 text-teal-300" />
        </div>

        <span className="flex-1 min-w-0 text-xs text-teal-200 font-medium truncate">
          {alertEvent.message}
        </span>

        <button
          onClick={onClose}
          aria-label="Fermer"
          className="shrink-0 w-5 h-5 flex items-center justify-center rounded-md text-teal-500 hover:text-teal-200 transition-colors"
          style={{ border: "1px solid rgba(20,184,166,0.20)" }}
        >
          <X className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}
