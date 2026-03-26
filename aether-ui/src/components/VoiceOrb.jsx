/**
 * VoiceOrb — the hero voice-capture interaction.
 *
 * Visual layers (bottom → top):
 *   1. Ambient radial glow blob (blur)
 *   2. Three expanding ring pings (listening only)
 *   3. Outer dashed ring — slow rotation
 *   4. Inner dashed ring — counter-rotation
 *   5. Solid accent ring with box-shadow glow
 *   6. Core button: gradient orb with mic icon / waveform bars
 */
export default function VoiceOrb({ listening, speechSupported, onClick }) {
  const SIZE = 220;

  return (
    <div
      className="relative flex items-center justify-center select-none"
      style={{ width: SIZE, height: SIZE }}
    >
      {/* ── 1. Ambient glow blob ──────────────────────────────────── */}
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          width:     listening ? 250 : 148,
          height:    listening ? 250 : 148,
          left: "50%",
          top: "50%",
          transform: "translate(-50%,-50%)",
          background: listening
            ? "radial-gradient(circle, rgba(244,63,94,0.30) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(99,102,241,0.17) 0%, transparent 70%)",
          filter:     "blur(30px)",
          transition: "all 1s ease",
        }}
      />

      {/* ── 2. Expand pings (listening only) ─────────────────────── */}
      {listening && (
        <>
          <span className="absolute inset-0 rounded-full border border-rose-400/55 animate-ring-ping-1" />
          <span className="absolute inset-0 rounded-full border border-rose-400/30 animate-ring-ping-2" />
          <span className="absolute inset-0 rounded-full border border-rose-400/12 animate-ring-ping-3" />
        </>
      )}

      {/* ── 3. Outer dashed ring — clockwise ─────────────────────── */}
      <div
        className="absolute rounded-full animate-spin-slow"
        style={{
          inset:       4,
          border:      "1px dashed",
          borderColor: listening
            ? "rgba(244,63,94,0.30)"
            : "rgba(99,102,241,0.20)",
          transition:  "border-color 0.7s ease",
        }}
      />

      {/* ── 4. Inner dashed ring — counter-clockwise ─────────────── */}
      <div
        className="absolute rounded-full animate-spin-reverse"
        style={{
          inset:       20,
          border:      "1px dashed",
          borderColor: listening
            ? "rgba(251,113,133,0.22)"
            : "rgba(139,92,246,0.16)",
          transition:  "border-color 0.7s ease",
        }}
      />

      {/* ── 5. Solid accent ring ──────────────────────────────────── */}
      <div
        className="absolute rounded-full"
        style={{
          inset:       34,
          border:      "1px solid",
          borderColor: listening
            ? "rgba(244,63,94,0.55)"
            : "rgba(99,102,241,0.28)",
          boxShadow:   listening
            ? "0 0 24px 0 rgba(244,63,94,0.26), inset 0 0 18px 0 rgba(244,63,94,0.06)"
            : "0 0 18px 0 rgba(99,102,241,0.12)",
          transition:  "all 0.7s ease",
        }}
      />

      {/* ── 6. Core orb button ───────────────────────────────────── */}
      <button
        onClick={onClick}
        disabled={!speechSupported}
        aria-label={listening ? "Arrêter l’enregistrement" : "Démarrer la capture vocale"}
        className="relative z-10 rounded-full flex items-center justify-center focus:outline-none"
        style={{
          width:      90,
          height:     90,
          cursor:     speechSupported ? "pointer" : "not-allowed",
          background: !speechSupported
            ? "rgba(39,39,42,0.80)"
            : listening
            ? "linear-gradient(135deg, #f43f5e 0%, #e11d48 100%)"
            : "linear-gradient(135deg, #6366f1 0%, #7c3aed 100%)",
          boxShadow:  !speechSupported
            ? "none"
            : listening
            ? "0 0 44px rgba(244,63,94,0.60), 0 4px 28px rgba(244,63,94,0.40)"
            : "0 0 30px rgba(99,102,241,0.46), 0 4px 20px rgba(99,102,241,0.32)",
          transition: "background 0.4s ease, box-shadow 0.4s ease, transform 0.15s ease",
        }}
        onMouseEnter={(e) => {
          if (speechSupported) e.currentTarget.style.transform = "scale(1.07)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "scale(1)";
        }}
        onMouseDown={(e) => {
          if (speechSupported) e.currentTarget.style.transform = "scale(0.93)";
        }}
        onMouseUp={(e) => {
          if (speechSupported) e.currentTarget.style.transform = "scale(1.04)";
        }}
      >
        {listening ? (
          /* ── Waveform bars ───────────────────────────────── */
          <div className="flex items-end gap-[3.5px]" style={{ height: 28 }}>
            {[11, 20, 26, 22, 16, 24, 11].map((h, i) => (
              <div
                key={i}
                className="rounded-full bg-white/88"
                style={{
                  width:           3,
                  height:          h,
                  transformOrigin: "center",
                  animation:       `waveform 0.85s ease-in-out ${i * 0.11}s infinite`,
                }}
              />
            ))}
          </div>
        ) : (
          /* ── Mic icon SVG ────────────────────────────────── */
          <svg
            width="34"
            height="34"
            viewBox="0 0 24 24"
            fill="none"
            stroke="rgba(255,255,255,0.92)"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <line x1="12" y1="19" x2="12" y2="22" />
            <line x1="8"  y1="22" x2="16" y2="22" />
          </svg>
        )}
      </button>
    </div>
  );
}
