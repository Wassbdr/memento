import { BellRing, Clock } from "lucide-react";

function formatTime(ts) {
  try {
    return new Intl.DateTimeFormat("fr-FR", { hour: "2-digit", minute: "2-digit" }).format(
      new Date(ts),
    );
  } catch {
    return "";
  }
}

export default function AlertCenter({ alerts }) {
  return (
    <section
      className="shrink-0 rounded-xl overflow-hidden"
      style={{
        border:     "1px solid rgba(255,255,255,0.04)",
        background: "rgba(255,255,255,0.012)",
        backdropFilter: "blur(20px)",
        minHeight:  132,
      }}
    >
      {/* Header */}
      <div
        className="px-4 py-2.5 flex items-center gap-2"
        style={{
          borderBottom: "1px solid rgba(255,255,255,0.04)",
          background:   "rgba(0,0,0,0.15)",
        }}
      >
        <BellRing className="w-3.5 h-3.5 text-teal-400" />
        <span className="text-xs font-semibold text-zinc-300 tracking-wide">Centre d’alertes</span>
        {alerts.length > 0 && (
          <span
            className="ml-1.5 px-1.5 py-0.5 rounded text-[9px] font-bold text-teal-400 tracking-wider"
            style={{
              background: "rgba(20,184,166,0.10)",
              border:     "1px solid rgba(20,184,166,0.22)",
            }}
          >
            {alerts.length}
          </span>
        )}
      </div>

      {/* Timeline list */}
      <div className="max-h-44 overflow-y-auto p-3 space-y-1.5">
        {alerts.length === 0 ? (
          <p className="text-xs text-zinc-700 text-center py-4">Aucune alerte pour l’instant</p>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className="flex gap-3 rounded-lg px-3 py-2 animate-fade-in"
              style={{
                background:  "rgba(20,184,166,0.04)",
                borderLeft:  "2px solid rgba(45,212,191,0.32)",
              }}
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs text-teal-200 leading-relaxed">{alert.message}</p>
                <div className="flex items-center gap-1 mt-1">
                  <Clock className="w-2.5 h-2.5 text-zinc-700" />
                  <span className="text-[9px] text-zinc-700">{formatTime(alert.timestamp)}</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
