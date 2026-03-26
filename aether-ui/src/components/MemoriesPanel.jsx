import { Brain, Clock } from "lucide-react";

const CATEGORY_LABELS_FR = {
  voice: "Voix",
  family: "Famille",
  memory: "Mémoire",
  alert: "Alerte",
  vision: "Vision",
  preference: "Préférence",
  contact: "Contact",
  meeting: "Réunion",
  task: "Tâche",
  work: "Travail",
  general: "Général",
};

// ── Per-category accent color (hex) ──────────────────────────────────────
// Used for inline styles so Tailwind JIT sees static class strings elsewhere.
function getAccentHex(cat) {
  switch (cat) {
    case "voice":      return "#818cf8";
    case "memory":     return "#34d399";
    case "alert":      return "#f43f5e";
    case "vision":     return "#a78bfa";
    case "preference": return "#38bdf8";
    case "contact":    return "#fbbf24";
    case "meeting":    return "#2dd4bf";
    case "task":       return "#fb923c";
    case "work":       return "#6366f1";
    default:           return "#71717a";
  }
}

function getBadgePalette(cat) {
  const hex = getAccentHex(cat);
  return { color: hex, bg: `${hex}18`, border: `${hex}33` };
}

function formatTimestamp(ts) {
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      month:  "short",
      day:    "numeric",
      hour:   "2-digit",
      minute: "2-digit",
    }).format(new Date(ts));
  } catch {
    return ts ?? "";
  }
}

// ── Memory card ───────────────────────────────────────────────────────────

function MemoryCard({ memory }) {
  const cat = (memory.category ?? "general").toLowerCase();
  const catLabel = CATEGORY_LABELS_FR[cat] ?? memory.category ?? "general";
  const accent = getAccentHex(cat);
  const badge  = getBadgePalette(cat);

  return (
    <article
      className="group relative rounded-xl p-3.5 overflow-hidden cursor-default animate-fade-in"
      style={{
        background:     "rgba(255,255,255,0.018)",
        border:         "1px solid rgba(255,255,255,0.055)",
        backdropFilter: "blur(12px)",
        transition:     "border-color 0.25s ease, transform 0.2s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = `${accent}40`;
        e.currentTarget.style.transform   = "translateY(-1px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.055)";
        e.currentTarget.style.transform   = "translateY(0)";
      }}
    >
      {/* Glowing top accent line */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{
          background: `linear-gradient(90deg,transparent,${accent}60,transparent)`,
        }}
      />

      {/* Hover glow */}
      <div
        className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-300"
        style={{
          background: `radial-gradient(ellipse at 50% 0%,${accent}10 0%,transparent 70%)`,
        }}
      />

      {/* Category badge */}
      <div className="relative flex items-center gap-2 mb-2.5">
        <span
          className="inline-flex items-center gap-1 text-[9px] font-bold px-2 py-0.5 rounded-full tracking-[0.16em] uppercase"
          style={{ color: badge.color, background: badge.bg, border: `1px solid ${badge.border}` }}
        >
          <span
            className="w-1 h-1 rounded-full shrink-0"
            style={{ background: badge.color }}
          />
          {catLabel}
        </span>
      </div>

      {/* Memory text */}
      <p className="relative text-[13px] text-zinc-200 leading-relaxed line-clamp-4">
        {memory.text}
      </p>

      {/* Timestamp */}
      <div className="relative flex items-center gap-1.5 mt-3">
        <Clock className="w-2.5 h-2.5 text-zinc-700" />
        <span className="text-[10px] text-zinc-600">
          {formatTimestamp(memory.timestamp)}
        </span>
      </div>
    </article>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────

export default function MemoriesPanel({ memories }) {
  return (
    <aside
      className="w-full lg:w-[400px] shrink-0 flex flex-col rounded-xl overflow-hidden"
      style={{
        border:         "1px solid rgba(255,255,255,0.045)",
        background:     "rgba(255,255,255,0.014)",
        backdropFilter: "blur(20px)",
        minHeight:      280,
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
        {/* Live indigo dot */}
        <span className="relative flex h-2 w-2 shrink-0">
          <span
            className="animate-ping absolute inset-0 rounded-full opacity-50"
            style={{ background: "#818cf8" }}
          />
          <span
            className="relative rounded-full h-2 w-2"
            style={{ background: "#818cf8", boxShadow: "0 0 7px rgba(129,140,248,0.85)" }}
          />
        </span>
        <Brain className="w-3.5 h-3.5 text-zinc-600" />
        <span className="text-xs font-semibold text-zinc-300 tracking-wide">
          Graphe de connaissances
        </span>
        <span className="ml-auto text-[10px] text-zinc-600 font-mono tracking-wider">
          {memories.length} nœuds
        </span>
      </div>

      {/* Cards — scrollable */}
      <div className="flex-1 overflow-y-auto p-3">
        {memories.length === 0 ? (
          /* Empty state — orbital rings */
          <div className="h-full flex flex-col items-center justify-center gap-5 py-14">
            <div className="relative w-20 h-20">
              {/* Outer orbit */}
              <div
                className="absolute inset-0 rounded-full animate-spin-slow"
                style={{ border: "1px dashed rgba(99,102,241,0.18)" }}
              />
              {/* Inner orbit */}
              <div
                className="absolute inset-3 rounded-full animate-spin-reverse"
                style={{ border: "1px dashed rgba(139,92,246,0.14)" }}
              />
              {/* Core */}
              <div
                className="absolute inset-6 rounded-full flex items-center justify-center"
                style={{
                  background: "rgba(99,102,241,0.06)",
                  border:     "1px solid rgba(99,102,241,0.15)",
                }}
              >
                <Brain className="w-4 h-4 text-zinc-700" />
              </div>
            </div>
            <p className="text-zinc-600 text-xs text-center max-w-[180px] leading-relaxed">
              Parlez pour enrichir votre graphe de mémoire
            </p>
          </div>
        ) : (
          /* 2-column responsive card grid */
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-2.5">
            {memories.map((memory) => (
              <MemoryCard key={memory.id} memory={memory} />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}

