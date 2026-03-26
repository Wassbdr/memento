import { useEffect, useRef } from "react";
import { MessageSquare, Terminal, Trash2 } from "lucide-react";

// ── Message classification ────────────────────────────────────────────────
function classifyMessage(role, text) {
  if (role === "assistant" && typeof text === "string") {
    const lower = text.toLowerCase();
    if (
      lower.includes("action proactive memento") ||
      lower.includes("aether proactive action") ||
      lower.includes("alerte système") ||
      lower.includes("system alert:") ||
      lower.includes("chuchotement proactif injecté") ||
      lower.includes("proactive whisper injected") ||
      lower.includes("directive interne memento") ||
      lower.includes("aether internal directive")
    ) {
      return "proactive";
    }
    if (text.startsWith("[")) return "system";
  }
  return role === "user" ? "user" : "assistant";
}

// Per-type color palette — using hex/rgba for inline styles so Tailwind JIT
// doesn't have to enumerate dynamic classes.
function getSystemPalette(text) {
  const l = text.toLowerCase();
  if (l.includes("whisper") || l.includes("chuchotement"))
    return { text: "rgba(251,191,36,0.95)", border: "rgba(251,191,36,0.18)", bg: "rgba(120,83,0,0.14)" };
  if (l.includes("vapi"))
    return { text: "rgba(192,167,250,0.95)", border: "rgba(139,92,246,0.20)", bg: "rgba(88,28,135,0.14)" };
  if (l.includes("vision"))
    return { text: "rgba(125,211,252,0.95)", border: "rgba(14,165,233,0.18)", bg: "rgba(7,89,133,0.13)" };
  return   { text: "rgba(94,234,212,0.95)",  border: "rgba(20,184,166,0.18)", bg: "rgba(13,148,136,0.10)" };
}

// ── Bubble components ─────────────────────────────────────────────────────

function SystemBubble({ text }) {
  const pal = getSystemPalette(text);
  return (
    <div className="flex justify-center my-3 animate-fade-in">
      <div
        className="inline-flex items-start gap-2 max-w-[92%] px-3.5 py-2 rounded-lg text-[11px] font-mono leading-relaxed"
        style={{
          color:          pal.text,
          border:         `1px solid ${pal.border}`,
          background:     pal.bg,
          backdropFilter: "blur(8px)",
        }}
      >
        <Terminal className="w-3 h-3 mt-0.5 shrink-0 opacity-55" />
        <span>{text}</span>
      </div>
    </div>
  );
}

function UserBubble({ text }) {
  return (
    <div className="flex justify-end mb-3 animate-slide-up">
      <div
        className="max-w-[72%] px-4 py-2.5 rounded-2xl rounded-tr-sm text-sm leading-relaxed text-zinc-100"
        style={{
          background: "rgba(99,102,241,0.18)",
          border:     "1px solid rgba(99,102,241,0.22)",
        }}
      >
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ text }) {
  return (
    <div className="flex justify-start mb-3 animate-slide-up">
      <div
        className="relative max-w-[72%] px-4 py-2.5 pl-5 rounded-2xl rounded-tl-sm text-sm leading-relaxed text-zinc-300 overflow-hidden"
        style={{
          background: "rgba(255,255,255,0.025)",
          border:     "1px solid rgba(255,255,255,0.04)",
        }}
      >
        {/* Gradient left accent bar */}
        <span
          className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full"
          style={{
            background: "linear-gradient(180deg,rgba(99,102,241,0) 0%,rgba(99,102,241,0.75) 50%,rgba(99,102,241,0) 100%)",
          }}
        />
        {text}
      </div>
    </div>
  );
}

function ProactiveBubble({ text }) {
  return (
    <div className="flex justify-start mb-3 animate-slide-up">
      <div
        className="relative max-w-[80%] px-4 py-3 rounded-xl overflow-hidden"
        style={{
          background: "rgba(251,191,36,0.10)",
          border: "1px solid rgba(251,191,36,0.42)",
          boxShadow: "0 0 0 1px rgba(251,191,36,0.18), 0 0 18px rgba(251,191,36,0.14)",
          animation: "glowPulse 1.8s ease-in-out infinite",
        }}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <span
            className="inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase"
            style={{
              color: "rgba(254,243,199,1)",
              background: "rgba(180,83,9,0.38)",
              border: "1px solid rgba(251,191,36,0.38)",
            }}
          >
            Memento
          </span>
        </div>
        <p className="text-sm leading-relaxed" style={{ color: "rgba(254,240,138,1)" }}>
          {text}
        </p>
      </div>
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────

export default function TranscriptPanel({ messages, onClear }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  return (
    <section
      className="flex-1 min-w-0 flex flex-col rounded-xl overflow-hidden"
      style={{
        border:         "1px solid rgba(255,255,255,0.045)",
        background:     "rgba(255,255,255,0.016)",
        backdropFilter: "blur(20px)",
      }}
    >
      {/* Header */}
      <div
        className="shrink-0 flex items-center gap-2.5 px-4 py-3"
        style={{
          borderBottom: "1px solid rgba(255,255,255,0.04)",
          background:   "rgba(0,0,0,0.22)",
        }}
      >
        {/* Live dot */}
        <span className="relative flex h-2 w-2 shrink-0">
          <span className="animate-ping absolute inset-0 rounded-full bg-emerald-400 opacity-45" />
          <span
            className="relative rounded-full h-2 w-2"
            style={{ background: "#34d399", boxShadow: "0 0 7px rgba(52,211,153,0.85)" }}
          />
        </span>
        <MessageSquare className="w-3.5 h-3.5 text-zinc-600" />
        <span className="text-xs font-semibold text-zinc-300 tracking-wide">Flux neural</span>
        <span className="ml-auto text-[10px] text-zinc-600 font-mono tracking-wider">
          {messages.length} entrées
        </span>
        <button
          onClick={onClear}
          className="ml-2 inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[10px] font-semibold tracking-wider uppercase text-zinc-500 hover:text-zinc-200 transition-colors duration-150"
          style={{ border: "1px solid rgba(255,255,255,0.06)" }}
        >
          <Trash2 className="w-3 h-3" />
          Effacer
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 pt-4 pb-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-2">
            <MessageSquare className="w-7 h-7 text-zinc-800" />
            <p className="text-xs text-zinc-700 text-center">Aucune activité pour l’instant</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const type = classifyMessage(msg.role, msg.text);
            if (type === "system")    return <SystemBubble    key={idx} text={msg.text} />;
            if (type === "proactive") return <ProactiveBubble key={idx} text={msg.text} />;
            if (type === "user")      return <UserBubble      key={idx} text={msg.text} />;
            return                           <AssistantBubble key={idx} text={msg.text} />;
          })
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
}
