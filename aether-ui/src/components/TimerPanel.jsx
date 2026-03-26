import { Clock3, Plus, Timer, X } from "lucide-react";
import { useMemo, useState } from "react";

function formatRemaining(remainingMs) {
  const totalSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export default function TimerPanel({ timers, nowMs, onStartTimer, onCancelTimer }) {
  const [minutesInput, setMinutesInput] = useState("1");
  const [labelInput, setLabelInput] = useState("Petite pause");

  const sortedTimers = useMemo(() => {
    return [...timers].sort((a, b) => a.endsAt - b.endsAt);
  }, [timers]);

  const submit = (event) => {
    event.preventDefault();
    const minutes = Number(minutesInput);
    const seconds = Math.round(minutes * 60);
    onStartTimer(seconds, labelInput);
  };

  return (
    <section
      className="w-full rounded-xl p-4"
      style={{
        background: "rgba(255,255,255,0.02)",
        border: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Timer className="w-4 h-4 text-cyan-300" />
        <h3 className="text-sm font-semibold text-zinc-200">Minuteur de rappel</h3>
      </div>

      <form onSubmit={submit} className="flex flex-col sm:flex-row gap-2.5">
        <label className="flex items-center gap-2 rounded-lg px-3 py-2 bg-zinc-900/60 border border-zinc-700/70">
          <Clock3 className="w-3.5 h-3.5 text-zinc-500" />
          <input
            type="number"
            min="0.1"
            max="1440"
            step="0.1"
            value={minutesInput}
            onChange={(e) => setMinutesInput(e.target.value)}
            className="w-20 bg-transparent text-xs text-zinc-100 outline-none"
            aria-label="Durée en minutes"
          />
          <span className="text-[11px] text-zinc-500">min</span>
        </label>

        <input
          type="text"
          value={labelInput}
          onChange={(e) => setLabelInput(e.target.value)}
          placeholder="Message du rappel"
          maxLength={100}
          className="flex-1 rounded-lg px-3 py-2 text-xs text-zinc-100 bg-zinc-900/60 border border-zinc-700/70 outline-none focus:border-cyan-400/50"
        />

        <button
          type="submit"
          className="inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-xs font-semibold tracking-wide text-cyan-200"
          style={{
            background: "rgba(6,182,212,0.12)",
            border: "1px solid rgba(34,211,238,0.28)",
          }}
        >
          <Plus className="w-3.5 h-3.5" />
          Démarrer
        </button>
      </form>

      <div className="mt-3 space-y-2 max-h-36 overflow-y-auto">
        {sortedTimers.length === 0 ? (
          <p className="text-[11px] text-zinc-600">Aucun minuteur actif</p>
        ) : (
          sortedTimers.map((timer) => {
            const remainingMs = timer.endsAt - nowMs;
            return (
              <div
                key={timer.id}
                className="flex items-center gap-3 px-3 py-2 rounded-lg"
                style={{
                  background: "rgba(255,255,255,0.018)",
                  border: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                <span className="text-xs font-mono text-cyan-300 shrink-0">
                  {formatRemaining(remainingMs)}
                </span>
                <span className="text-xs text-zinc-300 truncate flex-1">{timer.label}</span>
                <button
                  type="button"
                  onClick={() => onCancelTimer(timer.id)}
                  className="shrink-0 rounded-md p-1 text-zinc-500 hover:text-rose-300"
                  aria-label="Annuler le minuteur"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}