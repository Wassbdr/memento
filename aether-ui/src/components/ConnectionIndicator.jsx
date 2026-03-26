export default function ConnectionIndicator() {
  return (
    <div
      className="shrink-0 flex items-center justify-center gap-2 px-6 py-1.5 text-[11px] font-medium tracking-wide"
      style={{
        background:  "rgba(120,53,15,0.14)",
        borderBottom:"1px solid rgba(251,146,60,0.14)",
        color:       "rgba(251,146,60,0.80)",
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full shrink-0 animate-status-blink"
        style={{ background: "#f59e0b", boxShadow: "0 0 6px rgba(245,158,11,0.8)" }}
      />
      Reconnexion à Memento…
    </div>
  );
}
