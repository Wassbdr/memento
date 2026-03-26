import { Clock, Sparkles } from "lucide-react";
import VoiceOrb from "./VoiceOrb";

function formatShortTime(ts) {
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      hour:   "2-digit",
      minute: "2-digit",
    }).format(new Date(ts));
  } catch {
    return "";
  }
}

export default function UserTab({
  connected,
  listening,
  speechSupported,
  onVoiceCapture,
  memories,
}) {
  const quickItems = memories.slice(0, 6);

  return (
    <div className="h-full overflow-y-auto flex flex-col items-center">
      {/* ── Hero section ───────────────────────────────────────────── */}
      <div className="w-full max-w-xl mx-auto pt-10 pb-6 px-4 flex flex-col items-center text-center">
        {/* Eyebrow */}
        <div className="flex items-center gap-2 mb-5">
          <Sparkles className="w-3 h-3 text-indigo-400" />
          <span
            className="text-[10px] tracking-[0.32em] uppercase font-semibold"
            style={{ color: "rgba(129,140,248,0.85)" }}
          >
            Interface vocale
          </span>
          <Sparkles className="w-3 h-3 text-indigo-400" />
        </div>

        {/* Headline with gradient */}
        <h2
          className="text-3xl md:text-[2.6rem] font-bold leading-[1.15] mb-3"
          style={{
            background:           "linear-gradient(135deg,#e0e7ff 0%,#a5b4fc 38%,#c4b5fd 72%,#f0abfc 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor:  "transparent",
            backgroundClip:       "text",
          }}
        >
          Une phrase suffit.
          <br />
          Memento s’en souvient.
        </h2>
        <p className="text-sm text-zinc-500 max-w-xs leading-relaxed">
          Vos pensées, capturées et restituées au bon moment.
        </p>

        {/* ── Voice Orb ─────────────────────────────────────────────── */}
        <div className="mt-8 mb-2">
          <VoiceOrb
            listening={listening}
            speechSupported={speechSupported}
            onClick={onVoiceCapture}
          />
        </div>

        {/* State label */}
        <p
          className="mt-3 text-[11px] tracking-[0.18em] uppercase font-medium transition-all duration-300"
          style={{
            color: !speechSupported
              ? "rgba(82,82,91,1)"
              : listening
              ? "rgba(251,113,133,1)"
              : "rgba(113,113,122,1)",
          }}
        >
          {!speechSupported
            ? "Voix indisponible"
            : listening
            ? "● Enregistrement\u2026"
            : "Appuyez pour parler"}
        </p>

        {!connected && (
          <p className="mt-1 text-[11px] text-amber-500/80">Reconnexion au serveur Memento\u2026</p>
        )}
      </div>

      {/* ── Memory constellation ────────────────────────────────────── */}
      <div className="w-full max-w-xl px-4 pb-10">
        {quickItems.length > 0 ? (
          <>
            {/* Divider */}
            <div className="flex items-center gap-3 mb-4">
              <div
                className="h-px flex-1"
                style={{ background: "linear-gradient(90deg,transparent,rgba(99,102,241,0.22))" }}
              />
              <span className="text-[9px] text-zinc-600 tracking-[0.28em] uppercase shrink-0">
                Graphe de mémoire
              </span>
              <div
                className="h-px flex-1"
                style={{ background: "linear-gradient(270deg,transparent,rgba(99,102,241,0.22))" }}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {quickItems.map((item, i) => (
                <article
                  key={item.id}
                  className="group relative rounded-xl p-4 overflow-hidden cursor-default animate-slide-up"
                  style={{
                    background:           "rgba(255,255,255,0.018)",
                    border:               "1px solid rgba(255,255,255,0.06)",
                    backdropFilter:       "blur(12px)",
                    animationDelay:       `${i * 55}ms`,
                    animationFillMode:    "both",
                    transition:           "border-color 0.3s ease, transform 0.2s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "rgba(99,102,241,0.30)";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  {/* Hover glow overlay */}
                  <div
                    className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-300"
                    style={{
                      background: "radial-gradient(ellipse at 50% 0%,rgba(99,102,241,0.09) 0%,transparent 70%)",
                    }}
                  />
                  {/* Top glimmer line on hover */}
                  <div
                    className="absolute top-0 left-6 right-6 h-px opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                    style={{
                      background: "linear-gradient(90deg,transparent,rgba(129,140,248,0.45),transparent)",
                    }}
                  />

                  <p className="relative text-sm text-zinc-200 leading-relaxed line-clamp-3">
                    {item.text}
                  </p>
                  <div className="relative flex items-center gap-1.5 mt-3">
                    <Clock className="w-3 h-3 text-zinc-700" />
                    <span className="text-[10px] text-zinc-600">
                      {formatShortTime(item.timestamp)}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          </>
        ) : (
          <p className="text-center text-xs text-zinc-700 mt-2">
            Enregistrez votre premier souvenir ci-dessus
          </p>
        )}
      </div>
    </div>
  );
}
