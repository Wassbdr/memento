import { AlertCircle, CheckCircle, Loader2, RefreshCw, Sparkles } from "lucide-react";

export default function ActionBar({
  actionStatus,
  connected,
  onDemoReset,
  onSimulateWhisper,
  demoBusy,
}) {
  return (
    <header
      className="relative z-20 shrink-0 flex items-center justify-between gap-4 px-6 py-3"
      style={{
        borderBottom:   "1px solid rgba(255,255,255,0.04)",
        background:     "rgba(4,8,16,0.88)",
        backdropFilter: "blur(24px)",
      }}
    >
      {/* ── Brand ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 shrink-0">
        {/* Glow logo mark */}
        <div className="relative w-8 h-8 shrink-0">
          <div
            className="absolute inset-0 rounded-xl blur-sm"
            style={{ background: "linear-gradient(135deg,#6366f1,#7c3aed)", opacity: 0.7 }}
          />
          <div
            className="relative w-8 h-8 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg,#6366f1,#7c3aed)" }}
          >
            {/* Custom brain-circuit icon */}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.44-3.66z"/>
              <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.44-3.66z"/>
            </svg>
          </div>
        </div>

        <div>
          <h1
            className="text-sm font-bold tracking-[0.22em] uppercase"
            style={{
              background:             "linear-gradient(135deg,#a5b4fc 0%,#818cf8 45%,#c084fc 100%)",
              WebkitBackgroundClip:   "text",
              WebkitTextFillColor:    "transparent",
              backgroundClip:         "text",
            }}
          >
            Memento
          </h1>
          <p className="text-[9px] text-zinc-600 tracking-[0.32em] uppercase">Mémoire augmentée</p>
        </div>
      </div>

      {/* ── Right side ────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <button
          onClick={onSimulateWhisper}
          disabled={demoBusy}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold tracking-wider uppercase transition"
          style={{
            border: "1px solid rgba(45,212,191,0.28)",
            color: demoBusy ? "rgba(115,115,115,1)" : "rgba(153,246,228,1)",
            background: "rgba(13,148,136,0.10)",
          }}
        >
          <Sparkles className="w-3 h-3" />
          Simuler chuchotement
        </button>

        <button
          onClick={onDemoReset}
          disabled={demoBusy}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold tracking-wider uppercase transition"
          style={{
            border: "1px solid rgba(129,140,248,0.28)",
            color: demoBusy ? "rgba(115,115,115,1)" : "rgba(199,210,254,1)",
            background: "rgba(99,102,241,0.10)",
          }}
        >
          <RefreshCw className="w-3 h-3" />
          Réinitialiser
        </button>

        {/* Connection pulse dot */}
        <div className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{
              background: connected ? "#34d399" : "#f59e0b",
              boxShadow:  connected
                ? "0 0 7px rgba(52,211,153,0.9)"
                : undefined,
              animation:  connected ? undefined : "statusBlink 2s steps(2) infinite",
            }}
          />
          <span
            className="text-[10px] tracking-[0.22em] uppercase font-medium"
            style={{ color: connected ? "#34d399" : "#f59e0b" }}
          >
            {connected ? "En direct" : "Reconnexion"}
          </span>
        </div>

        {/* Status toast (slide-down) */}
        {actionStatus && (
          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium animate-slide-down"
            style={{
              border:         actionStatus.loading
                ? "1px solid rgba(255,255,255,0.07)"
                : actionStatus.ok
                ? "1px solid rgba(52,211,153,0.30)"
                : "1px solid rgba(248,113,113,0.30)",
              background:     actionStatus.loading
                ? "rgba(0,0,0,0.50)"
                : actionStatus.ok
                ? "rgba(16,185,129,0.10)"
                : "rgba(239,68,68,0.10)",
              color:          actionStatus.loading
                ? "rgba(161,161,170,1)"
                : actionStatus.ok
                ? "rgba(110,231,183,1)"
                : "rgba(252,165,165,1)",
              backdropFilter: "blur(12px)",
            }}
          >
            {actionStatus.loading ? (
              <Loader2 className="w-3 h-3 animate-spin shrink-0" />
            ) : actionStatus.ok ? (
              <CheckCircle className="w-3 h-3 shrink-0" />
            ) : (
              <AlertCircle className="w-3 h-3 shrink-0" />
            )}
            {actionStatus.loading ? "Traitement…" : actionStatus.message}
          </div>
        )}
      </div>
    </header>
  );
}
