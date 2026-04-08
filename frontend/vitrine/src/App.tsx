import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Activity, Brain, Mic, MicOff, type LucideProps } from "lucide-react";
import OrganicSphere, { type SphereState } from "./components/OrganicSphere";
import AlertFeed from "./components/AlertFeed";
import MemoryGraph from "./components/MemoryGraph";

// ─── Navigation ───────────────────────────────────────────────────────────────

type Panel = "soul" | "feed" | "brain";

const NAV: { id: Panel; label: string; Icon: React.FC<LucideProps> }[] = [
  { id: "soul", label: "L'Âme", Icon: Sparkles },
  { id: "feed", label: "Fil de Soin", Icon: Activity },
  { id: "brain", label: "Le Cerveau", Icon: Brain },
];

const PANEL_ORDER: Panel[] = ["soul", "feed", "brain"];

// ─── Panel transition variants ────────────────────────────────────────────────

const panelVariants = {
  enter: (dir: number) => ({
    opacity: 0,
    x: dir > 0 ? 55 : -55,
    scale: 0.975,
  }),
  center: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: { duration: 0.42, ease: [0.25, 0.1, 0.25, 1] },
  },
  exit: (dir: number) => ({
    opacity: 0,
    x: dir > 0 ? -55 : 55,
    scale: 0.975,
    transition: { duration: 0.28, ease: [0.25, 0.1, 0.25, 1] },
  }),
};

// ─── Soul Panel ───────────────────────────────────────────────────────────────

function SoulPanel({
  sphereState,
  onToggle,
}: {
  sphereState: SphereState;
  onToggle: () => void;
}) {

  return (
    <div className="w-full h-full flex flex-col items-center justify-center relative">
      {/* Patient label */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.35 }}
        className="absolute top-8 left-1/2 -translate-x-1/2 text-center pointer-events-none z-10"
      >
        <p className="text-[10px] text-white/20 uppercase tracking-[0.4em] mb-1">
          Memento · Profil
        </p>
        <p className="text-sm font-display font-medium text-white/50">
          Mamie Rose
        </p>
      </motion.div>

      {/* Sphere — fills available space */}
      <div className="w-full flex-1 min-h-0 relative">
        <OrganicSphere state={sphereState} />
      </div>

      {/* Bottom CTA */}
      <motion.div
        initial={{ opacity: 0, y: 22 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55, duration: 0.5 }}
        className="absolute inset-x-0 bottom-10 flex flex-col items-center gap-5 text-center"
      >
        <p className="text-sm text-white/40 font-light tracking-wide text-center">
        </p>

        <button
          onClick={onToggle}
          className={`relative group flex items-center gap-2.5 px-7 py-3.5 rounded-2xl transition-all duration-300 backdrop-blur-md ${
            sphereState === "listening"
              ? "bg-violet-500/20 border border-violet-400/45 shadow-lg shadow-violet-500/15"
              : "bg-white/[0.05] border border-white/[0.09] hover:bg-white/[0.09] hover:border-white/[0.16]"
          }`}
        >
          {sphereState === "listening" ? (
            <MicOff className="w-4 h-4 text-violet-300" />
          ) : (
            <Mic className="w-4 h-4 text-white/50 group-hover:text-white/75 transition-colors" />
          )}
          <span
            className={`text-sm font-medium ${
              sphereState === "listening"
                ? "text-violet-200"
                : "text-white/50 group-hover:text-white/75 transition-colors"
            }`}
          >
            {sphereState === "listening"
              ? "Arrêter l'écoute"
              : "Parler à Memento"}
          </span>

          {/* Pulsing ring when listening */}
          {sphereState === "listening" && (
            <motion.span
              className="absolute inset-0 rounded-2xl border border-violet-400/30 pointer-events-none"
              animate={{ scale: [1, 1.05, 1], opacity: [0.4, 0.75, 0.4] }}
              transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
            />
          )}
        </button>
      </motion.div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [panel, setPanel] = useState<Panel>("soul");
  const [dir, setDir] = useState(1);
  const [sphereState, setSphereState] = useState<SphereState>("idle");

  const navigate = (next: Panel) => {
    const ci = PANEL_ORDER.indexOf(panel);
    const ni = PANEL_ORDER.indexOf(next);
    setDir(ni >= ci ? 1 : -1);
    setPanel(next);
  };

  const toggleListening = () =>
    setSphereState((s) => (s === "idle" ? "listening" : "idle"));

  return (
    <div className="w-screen h-screen bg-[#050810] overflow-hidden flex relative">
      {/* ── Ambient background glow ── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[700px] h-[420px] bg-violet-700/7 rounded-full blur-[110px]" />
        <div className="absolute bottom-0 left-0 w-[450px] h-[320px] bg-blue-700/5 rounded-full blur-[90px]" />
        <div className="absolute bottom-0 right-0 w-[380px] h-[280px] bg-teal-500/4 rounded-full blur-[90px]" />
      </div>

      {/* ── Sidebar ── */}
      <aside className="relative z-20 flex flex-col items-center py-9 px-3 w-[68px] flex-shrink-0 border-r border-white/[0.06]">
        {/* Logo mark */}
        <div className="mb-9 w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-indigo-700 flex items-center justify-center shadow-lg shadow-violet-600/25">
          <span className="text-white text-xs font-display font-bold tracking-tight">
            M
          </span>
        </div>

        {/* Nav items */}
        <nav className="flex flex-col gap-1.5 flex-1">
          {NAV.map(({ id, label, Icon }) => {
            const active = panel === id;
            return (
              <div key={id} className="relative group">
                <button
                  onClick={() => navigate(id)}
                  className={`relative w-11 h-11 rounded-xl flex items-center justify-center transition-colors duration-200 ${
                    active
                      ? "text-violet-300"
                      : "text-white/28 hover:text-white/55"
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="nav-bg"
                      className="absolute inset-0 rounded-xl bg-violet-500/18 border border-violet-500/28"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  )}
                  <Icon
                    className="w-4 h-4 relative z-10"
                    strokeWidth={active ? 2 : 1.5}
                  />
                </button>

                {/* Tooltip */}
                <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 pointer-events-none z-50 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                  <div className="bg-[#1a1f35] text-white/75 text-xs px-2.5 py-1.5 rounded-lg whitespace-nowrap border border-white/10 shadow-xl">
                    {label}
                  </div>
                </div>
              </div>
            );
          })}
        </nav>

        {/* Live indicator */}
        <div className="mt-auto flex flex-col items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-teal-400 shadow-sm shadow-teal-400/60 animate-pulse" />
          <span
            className="text-[8px] text-white/18 uppercase tracking-widest"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Live
          </span>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 relative overflow-hidden">
        <AnimatePresence custom={dir} mode="wait">
          <motion.div
            key={panel}
            custom={dir}
            variants={panelVariants}
            initial="enter"
            animate="center"
            exit="exit"
            className="absolute inset-0"
          >
            {panel === "soul" && (
              <SoulPanel sphereState={sphereState} onToggle={toggleListening} />
            )}
            {panel === "feed" && <AlertFeed />}
            {panel === "brain" && <MemoryGraph />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
